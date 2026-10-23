"""Audit frozen files and create delivery archives without model evaluation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()
def read(path): return json.loads(path.read_text(encoding='utf-8'))
def dump(path, obj): path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def included(path):
    r=path.relative_to(ROOT)
    if any(p in {'.git','.venv','__pycache__','tmp'} for p in r.parts): return False
    s=r.as_posix()
    if any(s.startswith(p) for p in ('code/research_runs/checkpoints/','code/research_runs/selected_assets/','code/runs/preflight/')): return False
    return path.suffix not in {'.pyc','.zip'} and s!='DELIVERY_MANIFEST.json'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=ROOT.parent)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    freeze=read(ROOT/'code/runs/freeze.json')
    for name, expected in freeze['files'].items():
        assert digest(ROOT/'code'/name)==expected, f'Frozen file changed: {name}'
    upstream=read(ROOT/'code/PACKAGE_MANIFEST.json')['files']
    for name, expected in upstream.items():
        if name!='code/submission.py':
            assert digest(ROOT/name)==expected, f'Original package file changed: {name}'
    pages=len(PdfReader(ROOT/'output/pdf/report.pdf').pages)
    assert pages<=10
    required=sorted((ROOT/'code/runs/model_bundle').iterdir())
    assets=sum(p.stat().st_size for p in required)
    historical=sum(p.stat().st_size for p in (ROOT/'history/round1/model_bundle').iterdir())
    assert assets==freeze['asset_bytes']==1411186
    assert assets+historical<=16*1024**2
    for p in required:
        if p.suffix=='.npz':
            with zipfile.ZipFile(p) as z:
                assert all(i.compress_type==zipfile.ZIP_STORED for i in z.infolist())
    score=read(ROOT/'code/runs/test-score.json')
    resources=read(ROOT/'code/runs/test-resources.json')
    assert (score['correct'],score['n'])==(378,467)
    assert abs(score['accuracy']-100*378/467)<1e-12
    assert resources['threads']==4 and resources['total_seconds']<30
    metadata={'status':'local_artifacts_ready_not_submitted','student_id':None,'public_code_url':None,
        'public_checkpoint_url':None,'github_submission_issue_url':None,'public_commit':None,
        'accuracy':score['accuracy'],'correct':score['correct'],'n':score['n'],
        'method':'mlp_listwise_fusion0.5','method_freeze_sha256':digest(ROOT/'code/runs/freeze.json'),
        'report':'output/pdf/report.pdf','checkpoint_directory':'code/runs/model_bundle',
        'required_final_step':'Publish matching code/checkpoint links, supply student ID, and create the GitHub issue.',
        'deadline':'2026-10-31T23:59:00+08:00'}
    dump(ROOT/'SUBMISSION_METADATA.json',metadata)
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and included(p))
    entries={p.relative_to(ROOT).as_posix():{'sha256':digest(p),'bytes':p.stat().st_size} for p in files}
    delivery_id=hashlib.sha256(json.dumps(entries,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    manifest={'version':'round2-final','created_utc':datetime.now(timezone.utc).isoformat(),
        'delivery_content_sha256':delivery_id,'files':entries,
        'audit':{'frozen_hashes_match':True,'upstream_unchanged_except_submission_stub':True,
            'report_pages':pages,'inference_asset_bytes':assets,'historical_asset_bytes':historical,
            'conservative_assets_bytes':assets+historical,'asset_limit_bytes':16*1024**2,
            'local_complete_evaluation_seconds':resources['total_seconds'],'threads':4,
            'reference_cpu_eligibility':'pending instructor reproduction',
            'visual_review':'All eight pages inspected; chart annotation corrected and rechecked.',
            'inference_verification':'code/runs/verification.json',
            'test_reexecution_for_packaging':False},
        'omitted_from_delivery':['virtual environments','temporary rendering/reproduction files',
            'nonselected candidate weights','duplicate selected_assets and preflight outputs'],
        'lightweight_archive_omits':'Original code/data/**/*.jsonl and *.npz only; restore these from official data.',
        'manifest_self_hash':'Recorded by the external archive manifest to avoid circularity.'}
    dump(ROOT/'DELIVERY_MANIFEST.json',manifest)
    files.append(ROOT/'DELIVERY_MANIFEST.json')
    light=[p for p in files if not (p.relative_to(ROOT).as_posix().startswith('code/data/') and p.suffix in {'.jsonl','.npz'})]
    specs=[('mp2-round2-complete.zip',files,False),('mp2-round2-code-checkpoint.zip',light,False),('mp2-round2-checkpoint.zip',required,True)]
    archives=[]
    for name,paths,flat in specs:
        dest=args.output_dir/name
        with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in paths:
                arc=p.name if flat else 'mp2-round2/'+p.relative_to(ROOT).as_posix()
                z.write(p,arc)
        with zipfile.ZipFile(dest) as z:
            assert z.testzip() is None
            for p in paths:
                arc=p.name if flat else 'mp2-round2/'+p.relative_to(ROOT).as_posix()
                assert hashlib.sha256(z.read(arc)).hexdigest()==digest(p), arc
        archives.append({'file':name,'bytes':dest.stat().st_size,'sha256':digest(dest),'entries':len(paths),'zip_crc_and_content_hashes_verified':True})
    result={'delivery_content_sha256':delivery_id,'delivery_manifest_sha256':digest(ROOT/'DELIVERY_MANIFEST.json'),
        'archives':archives,'audit':manifest['audit'],'submission_status':metadata['status']}
    dump(args.output_dir/'mp2-round2-archives.manifest.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
