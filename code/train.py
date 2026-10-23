"""Train on train references, select on validation, never load test references.

The finite search grid below is recorded in runs/experiments.json. All model
weights, scalers, and PCA transformations are fitted on training inputs only.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '4'
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import platform
import time

import numpy as np
import scipy
import sklearn
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from common import DATA, load_inputs, parse_answer, read_jsonl
from features import extract_features, embedding_features, majority

SEED = 7506


def labels(bank, split):
    if split not in ('train', 'validation'):
        raise ValueError('Training program cannot read test references')
    gold = read_jsonl(DATA/f'gsm8k_{split}.jsonl')
    assert [r['id'] for r in gold] == [r['id'] for r in bank]
    return np.array([[parse_answer(c['text']) is not None and
                      parse_answer(c['text']) == parse_answer(g['answer'])
                      for c in r['candidates']] for r, g in zip(bank, gold)])


def hard_mask(bank):
    return np.array([len({a for c in r['candidates'] if (a := parse_answer(c['text'])) is not None}) >= 2 for r in bank])


def accuracy(scores, y, valid, mask=None):
    chosen = np.where(valid, scores, -1e9).argmax(axis=1)
    correct = y[np.arange(len(y)), chosen]
    return float(correct.mean() if mask is None else correct[mask].mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path('runs'))
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    tick = time.perf_counter()
    banks, arrays, blocks, valid, y, hard, emb = {}, {}, {}, {}, {}, {}, {}
    for split in ('train', 'validation'):
        banks[split], arrays[split] = load_inputs(split)
        blocks[split], valid[split] = extract_features(banks[split])
        y[split] = labels(banks[split], split)
        hard[split] = hard_mask(banks[split])
        emb[split] = embedding_features(arrays[split])
        print(split, {'n': len(y[split]), 'hard': int(hard[split].sum()),
                      'informative': int((~y[split].all(axis=1)).sum()),
                      'candidate_accuracy': y[split].mean()}, flush=True)
    # Groups with all four answers correct have no ranking supervision.
    informative = ~y['train'].all(axis=1)
    fitrows = np.repeat(informative, 4)
    target = y['train'].ravel()[fitrows]
    records, models, vscores = [], {}, {}
    def record(name, scores, seconds=0., **extra):
        entry = {'name': name, 'validation_accuracy': accuracy(scores, y['validation'], valid['validation']),
                 'validation_hard_accuracy': accuracy(scores, y['validation'], valid['validation'], hard['validation']),
                 'fit_seconds': seconds, **extra}
        records.append(entry)
        vscores[name] = scores
        print(json.dumps(entry), flush=True)
    n = len(y['validation'])
    first = np.zeros((n,4)); first[:,0] = 1
    record('first_parseable', first)
    modal = np.zeros((n,4)); modal[np.arange(n), majority(banks['validation'])] = 1
    record('majority', modal)
    # PCA is trained only on informative training groups, without validation.
    pca = PCA(n_components=32, svd_solver='randomized', random_state=SEED)
    pca.fit(emb['train'][1].reshape(-1,1024)[fitrows])
    p = {s: pca.transform(emb[s][1].reshape(-1,1024)) for s in banks}
    feature_sets = {
        'answer_only': ('answer',),
        'style_only': ('style',),
        'individual': ('answer','style','arithmetic'),
        'comparison': ('answer','style','arithmetic','comparison'),
        'no_arithmetic': ('answer','style','comparison'),
        'comparison_pca': ('answer','style','arithmetic','comparison'),
    }
    for name, keys in feature_sets.items():
        x = {s: np.concatenate([blocks[s][k] for k in keys], axis=-1).reshape(len(y[s])*4,-1) for s in banks}
        if name.endswith('_pca'):
            x = {s: np.concatenate([x[s], p[s]], axis=1) for s in banks}
        # Small fixed search; ablations use the same settings as the primary model.
        settings = [(7,10.), (15,10.), (15,1.)] if name in ('comparison','comparison_pca') else [(7,10.)]
        for leaves, l2 in settings:
            key = f'{name}_leaves{leaves}_l2{l2:g}'
            start = time.perf_counter()
            model = HistGradientBoostingClassifier(max_iter=160, learning_rate=.06,
                max_leaf_nodes=leaves, min_samples_leaf=30, l2_regularization=l2,
                early_stopping=False, random_state=SEED)
            model.fit(x['train'][fitrows], target)
            scores = model.decision_function(x['validation']).reshape(-1,4)
            models[key] = {'kind':'hist', 'model': model, 'blocks': keys,
                           'pca': pca if name.endswith('_pca') else None}
            record(key, scores, time.perf_counter()-start)
    # Linear embedding verifier: absolute trace representation and within-group residual.
    for kind in ('embedding','embedding_centered','embedding_hand'):
        x = {s: emb[s][1 if kind == 'embedding_centered' else 0].reshape(-1,1024) for s in banks}
        if kind == 'embedding_hand':
            x = {s: np.concatenate([x[s], *[blocks[s][k].reshape(len(y[s])*4,-1) for k in ('answer','style','arithmetic','comparison')]],axis=1) for s in banks}
        scaler = StandardScaler().fit(x['train'][fitrows])
        scaled = {s: scaler.transform(x[s]) for s in banks}
        for c in (.001, .01, .1):
            key = f'{kind}_C{c:g}'
            start = time.perf_counter()
            model = LogisticRegression(C=c, max_iter=400, solver='lbfgs', random_state=SEED)
            model.fit(scaled['train'][fitrows], target)
            scores = model.decision_function(scaled['validation']).reshape(-1,4)
            models[key] = {'kind':kind, 'model':model,'scaler':scaler}
            record(key, scores, time.perf_counter()-start)
    # Choose each family by hard-validation accuracy, then total accuracy, then simplicity.
    def best(names):
        return max((r for r in records if r['name'] in names),
                   key=lambda r: (r['validation_hard_accuracy'],r['validation_accuracy']))['name']
    tree = best([name for name in models if name.startswith('comparison')])
    # Best individual accuracy does not imply the best complementary ensemble.
    # Retain one train-fitted verifier from each predeclared linear family.
    for family in ('embedding','embedding_centered','embedding_hand'):
        linear = best([f'{family}_C{c:g}' for c in (.001,.01,.1)])
        for alpha in (0., .25, .5, .75, 1.):
            record(f'fusion_{family}_{alpha:g}', (1-alpha)*vscores[tree]+alpha*vscores[linear],
                   tree=tree, linear=linear, alpha=alpha)
    selected_name = best([r['name'] for r in records if r['name'].startswith('fusion_')])
    selected = next(r for r in records if r['name'] == selected_name)
    alpha = selected['alpha']
    linear = selected['linear']
    bundle = {'version':1,'seed':SEED,'alpha':alpha,
              'tree': models[tree] if alpha < 1 else None,
              'linear':models[linear] if alpha > 0 else None,
              'selected': selected, 'protocol':'dase7506-mp2-open-hard-verifier-v5'}
    checkpoint = out/'model_bundle'
    checkpoint.mkdir(exist_ok=True)
    with (checkpoint/'model.pkl').open('wb') as f:
        pickle.dump(bundle, f, protocol=5)
    metadata = {'seed':SEED,'selection_metric':'hard validation accuracy, then overall validation accuracy',
                'search':records,'selected':selected,'total_training_seconds':time.perf_counter()-tick,
                'feature_extraction_included':True,'train_informative_groups':int(informative.sum()),
                'split_counts':{s:{'total':len(y[s]),'hard':int(hard[s].sum())} for s in banks},
                'versions':{'python':platform.python_version(),'numpy':np.__version__,
                            'scipy':scipy.__version__,'scikit-learn':sklearn.__version__},
                'hardware':{'platform':platform.platform(),'cpu':os.environ.get('PROCESSOR_IDENTIFIER'),
                            'logical_cpus':os.cpu_count(),'threads':4},
                'checkpoint_sha256':hashlib.sha256((checkpoint/'model.pkl').read_bytes()).hexdigest(),
                'checkpoint_bytes':(checkpoint/'model.pkl').stat().st_size}
    (out/'experiments.json').write_text(json.dumps(metadata,indent=2)+'\n')
    np.savez(out/'validation_analysis.npz',labels=y['validation'], valid=valid['validation'],hard=hard['validation'],
             **{k:v for k,v in vscores.items()})
    print('SELECTED', json.dumps(selected), flush=True)
    print('TOTAL_SECONDS',metadata['total_training_seconds'],flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
