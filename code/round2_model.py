"""NumPy-only learned scoring layers; sklearn is used only for the legacy member.

Inference never imports torch or any training/reference-reading code.
"""
import json
import pickle
from pathlib import Path
import numpy as np
from features import extract_features, embedding_features
from selector import component_scores


def input_features(blocks, embeddings, context=False):
    pieces = [embeddings[0]]
    if context:
        pieces.append(embeddings[1])
    pieces.extend(blocks[k] for k in ('answer','style','arithmetic','comparison'))
    return np.concatenate(pieces,axis=-1).astype(np.float32)


def normalized_scores(scores,scale):
    return (scores-scores.mean(axis=1,keepdims=True))/max(float(scale),1e-8)


def new_member_scores(x,weights):
    z = np.clip((x-weights['mean'])/weights['scale'],-8,8)
    z = z @ weights['w0'] + weights['b0']
    if 'w1' in weights:
        z = np.maximum(z,0) @ weights['w1'] + weights['b1']
    return normalized_scores(z.squeeze(-1),float(weights['score_scale']))


def legacy_scores(bundle,blocks,emb,n):
    result = np.zeros((n,4))
    if bundle['tree'] is not None:
        result += (1-bundle['alpha'])*component_scores(bundle['tree'],blocks,emb,n)
    if bundle['linear'] is not None:
        result += bundle['alpha']*component_scores(bundle['linear'],blocks,emb,n)
    return result


def score_candidates(bank,arrays,checkpoint):
    if not bank:
        return np.empty((0,4))
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint/'config.json').read_text())
    blocks, valid = extract_features(bank)
    emb = embedding_features(arrays)
    scores = np.zeros((len(bank),4),dtype=np.float64)
    alpha = config['new_weight']
    if alpha < 1:
        with (checkpoint/'legacy.pkl').open('rb') as f:
            legacy = pickle.load(f)
        scores += (1-alpha)*normalized_scores(legacy_scores(legacy,blocks,emb,len(bank)),config['legacy_scale'])
    if alpha > 0:
        x = input_features(blocks,emb,config['context'])
        for filename in config['members']:
            with np.load(checkpoint/filename,allow_pickle=False) as weights:
                scores += alpha/len(config['members'])*new_member_scores(x,weights)
    return np.where(valid,scores,-1e9)


def predict(bank,arrays,checkpoint):
    return score_candidates(bank,arrays,checkpoint).argmax(axis=1).tolist()
