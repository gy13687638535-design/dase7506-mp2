"""Frozen offline selection. Input-only: no access to reference files."""
from pathlib import Path
import pickle

import numpy as np
from features import extract_features, embedding_features


def component_scores(component, blocks, embeddings, n):
    kind = component['kind']
    if kind == 'hist':
        x = np.concatenate([blocks[k] for k in component['blocks']],axis=-1).reshape(n*4,-1)
        if component['pca'] is not None:
            x = np.concatenate([x,component['pca'].transform(embeddings[1].reshape(n*4,-1))],axis=1)
    else:
        x = embeddings[1 if kind == 'embedding_centered' else 0].reshape(n*4,-1)
        if kind == 'embedding_hand':
            x = np.concatenate([x,*[blocks[k].reshape(n*4,-1) for k in ('answer','style','arithmetic','comparison')]],axis=1)
        x = component['scaler'].transform(x)
    return component['model'].decision_function(x).reshape(n,4)


def score_candidates(bank, arrays, checkpoint):
    if not bank:
        return np.empty((0,4))
    path = Path(checkpoint) if checkpoint is not None else Path(__file__).parent/'runs'/'model_bundle'
    with (path/'model.pkl').open('rb') as f:
        bundle = pickle.load(f)
    blocks, valid = extract_features(bank)
    emb = embedding_features(arrays)
    scores = np.zeros((len(bank),4))
    if bundle['tree'] is not None:
        scores += (1-bundle['alpha'])*component_scores(bundle['tree'],blocks,emb,len(bank))
    if bundle['linear'] is not None:
        scores += bundle['alpha']*component_scores(bundle['linear'],blocks,emb,len(bank))
    return np.where(valid, scores, -1e9)


def predict(bank, arrays, checkpoint):
    return score_candidates(bank, arrays, checkpoint).argmax(axis=1).tolist()
