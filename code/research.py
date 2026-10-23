"""Prospectively bounded round-2 research. Never reads test data or results."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '4'
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import platform
import shutil
import time

import numpy as np
import torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from common import load_inputs
from train import labels, hard_mask
from features import extract_features, embedding_features
from round2_model import input_features, normalized_scores, new_member_scores, legacy_scores

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)


def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')


def metrics(scores,y,valid,hard):
    choice = np.where(valid,scores,-1e9).argmax(axis=1)
    correct = y[np.arange(len(y)),choice]
    return {'accuracy':float(correct.mean()),'hard_accuracy':float(correct[hard].mean()),
            'correct':int(correct.sum()),'hard_correct':int(correct[hard].sum()),
            'n':len(y),'hard_n':int(hard.sum())}


def scale_scores(scores):
    return max(float((scores-scores.mean(axis=1,keepdims=True)).std()),1e-6)


def prep(x):
    scaler = StandardScaler().fit(x.reshape(-1,x.shape[-1]))
    return scaler.mean_.astype(np.float32),scaler.scale_.astype(np.float32)


def pair_linear(x,y,xe,c):
    mean,scale = prep(x)
    z = np.clip((x-mean)/scale,-8,8)
    diffs,weights = [],[]
    for row,label in zip(z,y):
        positive,negative = np.flatnonzero(label),np.flatnonzero(~label)
        for a in positive:
            for b in negative:
                d = row[a]-row[b]
                diffs.extend((d,-d))
                weights.extend((1/(len(positive)*len(negative)),)*2)
    model = LogisticRegression(C=c,fit_intercept=False,max_iter=500,solver='lbfgs',random_state=7506)
    model.fit(np.asarray(diffs),np.tile([1,0],len(diffs)//2),sample_weight=np.asarray(weights))
    result = {'mean':mean,'scale':scale,'w0':model.coef_.T.astype(np.float32),'b0':np.zeros(1,dtype=np.float32)}
    train_scores = (z @ result['w0']).squeeze(-1)
    result['score_scale'] = np.array(scale_scores(train_scores),dtype=np.float32)
    return result,new_member_scores(xe,result)


def neural(x,y,xe,family,decay,seed,epochs,plan):
    torch.manual_seed(seed)
    mean,scale = prep(x)
    z = np.clip((x-mean)/scale,-8,8).astype(np.float32)
    ze = np.clip((xe-mean)/scale,-8,8).astype(np.float32)
    tx,ty,te = torch.from_numpy(z),torch.from_numpy(y),torch.from_numpy(ze)
    model = nn.Sequential(nn.Linear(z.shape[-1],plan['hidden_units']),nn.ReLU(),
                          nn.Dropout(plan['dropout']),nn.Linear(plan['hidden_units'],1))
    optimizer = torch.optim.AdamW(model.parameters(),lr=plan['learning_rate'],weight_decay=decay)
    generator = torch.Generator().manual_seed(seed+123)
    scores,exported = {},{}
    for epoch in range(1,max(epochs)+1):
        model.train()
        order = torch.randperm(len(tx),generator=generator)
        for ids in order.split(plan['batch_groups']):
            s = model(tx[ids]).squeeze(-1)
            target = ty[ids]
            if family == 'mlp_bce':
                loss = nn.functional.binary_cross_entropy_with_logits(s,target.float())
            elif family == 'mlp_pairwise':
                mask = target[:,:,None] & ~target[:,None,:]
                penalties = nn.functional.softplus(s[:,None,:]-s[:,:,None])
                loss = ((penalties*mask).sum(dim=(1,2))/mask.sum(dim=(1,2))).mean()
            else:
                loss = (torch.logsumexp(s,dim=1)-torch.logsumexp(s.masked_fill(~target,-torch.inf),dim=1)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
        if epoch in epochs:
            model.eval()
            with torch.no_grad():
                st = model(tx).squeeze(-1).numpy()
                se = model(te).squeeze(-1).numpy()
            weights = {'mean':mean,'scale':scale,
                       'w0':model[0].weight.detach().numpy().T.copy(),'b0':model[0].bias.detach().numpy().copy(),
                       'w1':model[3].weight.detach().numpy().T.copy(),'b1':model[3].bias.detach().numpy().copy(),
                       'score_scale':np.array(scale_scores(st),dtype=np.float32)}
            # NumPy export must reproduce training-library scores without torch at inference.
            np.testing.assert_allclose(new_member_scores(xe[:32],weights),normalized_scores(se[:32],weights['score_scale']),rtol=2e-4,atol=2e-4)
            scores[epoch] = normalized_scores(se,weights['score_scale'])
            exported[epoch] = weights
    return exported,scores


def refit_legacy(blocks,emb,y,fit):
    pca = PCA(n_components=32,svd_solver='randomized',random_state=7506).fit(emb[1][fit].reshape(-1,1024))
    x = np.concatenate([blocks[k] for k in ('answer','style','arithmetic','comparison')],axis=-1)
    x = np.concatenate([x,pca.transform(emb[1].reshape(-1,1024)).reshape(len(y),4,32)],axis=-1)
    tree = HistGradientBoostingClassifier(max_iter=160,learning_rate=.06,max_leaf_nodes=15,min_samples_leaf=30,
                l2_regularization=10.,early_stopping=False,random_state=7506).fit(x[fit].reshape(-1,x.shape[-1]),y[fit].ravel())
    scaler = StandardScaler().fit(emb[0][fit].reshape(-1,1024))
    linear = LogisticRegression(C=.1,max_iter=400,solver='lbfgs',random_state=7506).fit(scaler.transform(emb[0][fit].reshape(-1,1024)),y[fit].ravel())
    return {'alpha':.75,'tree':{'kind':'hist','model':tree,'blocks':('answer','style','arithmetic','comparison'),'pca':pca},
            'linear':{'kind':'embedding','model':linear,'scaler':scaler}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('research_runs'))
    args = parser.parse_args()
    out = args.output
    if (out/'selection.json').exists():
        raise SystemExit('Completed run exists; use another --output directory for reproduction.')
    out.mkdir(parents=True,exist_ok=True)
    plan = json.loads(Path('research_plan.json').read_text())
    begin = time.perf_counter()
    dump(out/'started.json',{'utc':datetime.now(timezone.utc).isoformat(),
        'plan_sha256':hashlib.sha256(Path('research_plan.json').read_bytes()).hexdigest(),
        'source_sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in ('research.py','round2_model.py','features.py')},
        'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','threads':4})
    bank,arrays = load_inputs('train')
    blocks,valid = extract_features(bank); emb = embedding_features(arrays)
    y = labels(bank,'train'); hard = hard_mask(bank)
    fit_idx,hold_idx = train_test_split(np.arange(len(bank)),test_size=plan['internal_holdout_fraction'],
        random_state=plan['split_seed'],stratify=y.sum(axis=1))
    fit_idx = fit_idx[~y[fit_idx].all(axis=1)]
    full_idx = np.flatnonzero(~y.all(axis=1))
    np.savez(out/'train_partition.npz',fit=fit_idx,holdout=hold_idx,full_fit=full_idx)
    inner_baseline = refit_legacy(blocks,emb,y,fit_idx)
    inner_old_raw = legacy_scores(inner_baseline,blocks,emb,len(bank))
    inner_old_scale = scale_scores(inner_old_raw[fit_idx])
    inner_old = normalized_scores(inner_old_raw[hold_idx],inner_old_scale)
    hm = lambda s: metrics(s,y[hold_idx],valid[hold_idx],hard[hold_idx])
    old_hold_metric = hm(inner_old)
    print('INTERNAL_BASELINE',json.dumps(old_hold_metric),flush=True)
    xs = {False:input_features(blocks,emb),True:input_features(blocks,emb,True)}
    hold_scores,inner_records = {},[]
    for c in plan['pairwise_linear_C']:
        tick=time.perf_counter(); key=f'pair_linear_C{c:g}'
        _,score = pair_linear(xs[False][fit_idx],y[fit_idx],xs[False][hold_idx],c)
        hold_scores[key]=score
        row={'family':'pair_linear','C':c,'name':key,'seconds':time.perf_counter()-tick,**hm(score)}
        inner_records.append(row); print(json.dumps(row),flush=True)
    for family in plan['neural_families']:
        context = family == 'context_listwise'
        for decay in plan['weight_decay']:
            for seed in plan['seeds']:
                tick=time.perf_counter()
                _,scores=neural(xs[context][fit_idx],y[fit_idx],xs[context][hold_idx],family,decay,seed,plan['epoch_checkpoints'],plan)
                seconds=time.perf_counter()-tick
                for epoch,score in scores.items():
                    key=f'{family}_wd{decay:g}_seed{seed}_ep{epoch}'
                    hold_scores[key]=score
                    inner_records.append({'name':key,'family':family,'decay':decay,'seed':seed,'epoch':epoch,
                                          'trajectory_seconds':seconds,**hm(score)})
                print(f'INNER {family} wd={decay} seed={seed} seconds={seconds:.2f} hard_by_epoch='+str({ep:round(hm(s)['hard_accuracy'],4) for ep,s in scores.items()}),flush=True)
                dump(out/'inner_results.json',inner_records)
    best_linear = max((r for r in inner_records if r['family']=='pair_linear'),key=lambda r:(r['hard_accuracy'],r['accuracy']))
    chosen=[{'family':'pair_linear','C':best_linear['C'],'inner_keys':[best_linear['name']]}]
    for family in plan['neural_families']:
        candidates=[]
        for decay in plan['weight_decay']:
            for epoch in plan['epoch_checkpoints']:
                rows=[r for r in inner_records if r['family']==family and r['decay']==decay and r['epoch']==epoch]
                candidates.append({'family':family,'decay':decay,'epoch':epoch,
                    'mean_hard':float(np.mean([r['hard_accuracy'] for r in rows])),
                    'mean_all':float(np.mean([r['accuracy'] for r in rows])), 'inner_keys':[r['name'] for r in rows]})
        chosen.append(max(candidates,key=lambda r:(r['mean_hard'],r['mean_all'])))
    dump(out/'inner_selection.json',chosen)
    # Official validation inputs/references are opened only after inner selection.
    vb,va=load_inputs('validation'); vf,vvalid=extract_features(vb); ve=embedding_features(va)
    vy=labels(vb,'validation'); vh=hard_mask(vb)
    vx={False:input_features(vf,ve),True:input_features(vf,ve,True)}
    with Path('../history/round1/model_bundle/model.pkl').open('rb') as f:
        old = pickle.load(f)
    old_scale=scale_scores(legacy_scores(old,blocks,emb,len(bank))[full_idx])
    old_scores=normalized_scores(legacy_scores(old,vf,ve,len(vb)),old_scale)
    vm=lambda s:metrics(s,vy,vvalid,vh)
    old_metric=vm(old_scores)
    print('OFFICIAL_VALIDATION_BASELINE',json.dumps(old_metric),flush=True)
    all_val={'baseline':old_scores}; final_records=[]; candidates=[]
    for config in chosen:
        family=config['family']; context=family=='context_listwise'; seed_scores=[]; files=[]
        family_dir=out/'checkpoints'/family;family_dir.mkdir(parents=True,exist_ok=True)
        seeds=plan['seeds'] if family!='pair_linear' else [7506]
        for seed in seeds:
            tick=time.perf_counter()
            if family=='pair_linear':
                weights,score=pair_linear(xs[False][full_idx],y[full_idx],vx[False],config['C'])
            else:
                weights_by_epoch,scores_by_epoch=neural(xs[context][full_idx],y[full_idx],vx[context],family,
                    config['decay'],seed,[config['epoch']],plan)
                weights=weights_by_epoch[config['epoch']];score=scores_by_epoch[config['epoch']]
            filename=f'seed{seed}.npz';np.savez(family_dir/filename,**weights);files.append(filename)
            seed_scores.append(score);all_val[f'{family}_seed{seed}']=score
            record={'family':family,'seed':seed,'seconds':time.perf_counter()-tick,**vm(score)}
            final_records.append(record);print('REFIT',json.dumps(record),flush=True)
        average=np.mean(seed_scores,axis=0); all_val[family]=average
        hold_average=np.mean([hold_scores[k] for k in config['inner_keys']],axis=0)
        for alpha in plan['fusion_weights_new_model']:
            mixed=(1-alpha)*old_scores+alpha*average
            internal=(1-alpha)*inner_old+alpha*hold_average
            seed_metrics=[vm((1-alpha)*old_scores+alpha*s)['hard_accuracy'] for s in seed_scores]
            improved_seeds=sum(v>old_metric['hard_accuracy'] for v in seed_metrics)
            eligible=(vm(mixed)['hard_accuracy']>old_metric['hard_accuracy'] and
                      hm(internal)['hard_accuracy']>=old_hold_metric['hard_accuracy'] and
                      improved_seeds >= (2 if len(seed_scores)==3 else 1))
            key=f'{family}_fusion{alpha:g}';all_val[key]=mixed
            row={'name':key,'family':family,'new_weight':alpha,'context':context,'members':files,
                 'inner':hm(internal),'seed_fusion_hard':seed_metrics,'improved_seeds':improved_seeds,
                 'eligible':eligible,'member_bytes':sum((family_dir/f).stat().st_size for f in files),**vm(mixed)}
            candidates.append(row)
        dump(out/'refit_results.json',final_records);dump(out/'fusion_results.json',candidates)
    eligible=[r for r in candidates if r['eligible']]
    selected=max(eligible,key=lambda r:(r['hard_accuracy'],r['accuracy'],-r['member_bytes'])) if eligible else None
    assets=out/'selected_assets';assets.mkdir(exist_ok=True)
    if selected is not None:
        conf={'version':2,'family':selected['family'],'context':selected['context'],
              'new_weight':selected['new_weight'],'legacy_scale':old_scale,'members':selected['members'],
              'seed':plan['seeds'],'protocol':'dase7506-mp2-open-hard-verifier-v5'}
        for filename in selected['members']:
            shutil.copy2(out/'checkpoints'/selected['family']/filename,assets/filename)
        if selected['new_weight']<1:
            shutil.copy2('../history/round1/model_bundle/model.pkl',assets/'legacy.pkl')
    else:
        conf={'version':2,'family':'round1_retained','context':False,'new_weight':0.,'legacy_scale':old_scale,
              'members':[],'protocol':'dase7506-mp2-open-hard-verifier-v5'}
        shutil.copy2('../history/round1/model_bundle/model.pkl',assets/'legacy.pkl')
    dump(assets/'config.json',conf)
    asset_bytes=sum(p.stat().st_size for p in assets.iterdir())
    assert asset_bytes<=plan['constraints']['asset_limit_bytes']
    result={'selected':selected,'baseline_validation':old_metric,'baseline_internal_holdout':old_hold_metric,
            'total_seconds':time.perf_counter()-begin,'asset_bytes':asset_bytes,
            'fit_groups_inner':len(fit_idx),'holdout_groups':len(hold_idx),'fit_groups_full':len(full_idx),
            'completed_utc':datetime.now(timezone.utc).isoformat(),
            'prior_test_exposure_disclosed':True,'test_used':False}
    np.savez(out/'validation_scores.npz',labels=vy,valid=vvalid,hard=vh,**all_val)
    np.savez(out/'inner_scores.npz',labels=y[hold_idx],valid=valid[hold_idx],hard=hard[hold_idx],baseline=inner_old,**hold_scores)
    dump(out/'selection.json',result)
    print('SELECTED',json.dumps(result),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):
        main()
