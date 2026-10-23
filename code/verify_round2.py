"""Validation-input-only contract, independence and export checks."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='4'
from copy import deepcopy
import json
import argparse
from pathlib import Path
import sys
import numpy as np
from common import load_inputs,validate_choices
from features import equation_stats,safe_arithmetic
from round2_model import predict,score_candidates


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,default=Path('runs/model_bundle'))
    checkpoint=parser.parse_args().checkpoint
    bank,arrays=load_inputs('validation')
    scores=score_candidates(bank,arrays,checkpoint)
    decisions=scores.argmax(axis=1).tolist()
    validate_choices(bank,[r['id'] for r in bank],decisions)
    selection=json.loads(Path('research_runs/selection.json').read_text())
    with np.load('research_runs/validation_scores.npz') as archive:
        expected=archive[selection['selected']['name'] if selection['selected'] else 'baseline']
        expected=np.where(scores>-1e8,expected,-1e9)
        assert decisions==expected.argmax(axis=1).tolist()
        np.testing.assert_allclose(scores,expected,atol=3e-5,rtol=3e-5)
    bank=bank[:80];arrays={'embeddings':arrays['embeddings'][:80]};scores=scores[:80];decisions=decisions[:80]
    assert predict([],{'embeddings':np.empty((0,4,1024),dtype=np.float32)},checkpoint)==[]
    changed=deepcopy(bank[::-1])
    for i,row in enumerate(changed):row['id']=f'fresh-unseen-question-{i}'
    assert predict(changed,{'embeddings':arrays['embeddings'][::-1]},checkpoint)==decisions[::-1]
    batched=[]
    for start in range(0,len(bank),7):
        batched+=predict(bank[start:start+7],{'embeddings':arrays['embeddings'][start:start+7]},checkpoint)
    assert batched==decisions
    permutation=[2,0,3,1];changed=deepcopy(bank)
    for row in changed:
        old=row['candidates'];row['candidates']=[dict(old[j],index=i) for i,j in enumerate(permutation)]
    perm_scores=score_candidates(changed,{'embeddings':arrays['embeddings'][:,permutation]},checkpoint)
    np.testing.assert_allclose(perm_scores,scores[:,permutation],atol=3e-5,rtol=3e-5)
    # Audit hooks inspect actual file/network accesses, not just the official loader.
    enabled=[True];asset_root=checkpoint.resolve()
    def audit(event,args):
        if not enabled[0]:return
        if event in ('socket.connect','socket.getaddrinfo'):
            raise AssertionError('Network use during prediction')
        if event=='open' and isinstance(args[0],(str,bytes)):
            path=Path(os.fsdecode(args[0])).resolve()
            if path.suffix=='.jsonl':raise AssertionError('Bank/reference reads during prediction')
            if path.suffix in ('.npz','.pkl','.json') and asset_root not in path.parents:
                raise AssertionError(f'Non-asset data read: {path}')
    sys.addaudithook(audit)
    try:assert predict(bank,arrays,checkpoint)==decisions
    finally:enabled[0]=False
    assert 'torch' not in sys.modules
    assert safe_arithmetic('(12-4)/2')==4
    assert equation_stats(r'\frac{12}{3} = 4')[0]==1
    for expression in ("__import__('os').getcwd()",'2**100000','open(1)'):
        try:safe_arithmetic(expression)
        except ValueError:pass
        else:raise AssertionError('Executable expression accepted')
    asset_bytes=sum(p.stat().st_size for p in checkpoint.iterdir())
    array_bytes=0
    for path in checkpoint.glob('*.npz'):
        with np.load(path,allow_pickle=False) as archive:array_bytes+=sum(archive[k].nbytes for k in archive.files)
    assert asset_bytes<=16*2**20
    result={'status':'passed','full_validation_export_matching':True,'contract_test_questions':80,
            'checks':['empty inputs','new IDs','reordered questions','candidate permutation','batch consistency',
                      'no reference/bank/output-cache file reads','no network calls','no torch inference import',
                      'bounded arithmetic','official prediction types'],'asset_bytes':asset_bytes,
            'uncompressed_npz_array_bytes':array_bytes}
    Path('runs').mkdir(exist_ok=True)
    Path('runs/verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
