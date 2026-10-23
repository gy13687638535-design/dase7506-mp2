"""Externally time the untouched official prediction and scoring processes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--split',choices=('validation','test'),default='validation')
    parser.add_argument('--checkpoint',type=Path,default=Path('runs/model_bundle'))
    parser.add_argument('--output-dir',type=Path,default=Path('runs'))
    args=parser.parse_args()
    if args.split=='test':
        freeze=json.loads(Path('runs/freeze.json').read_text())
        for name,digest in freeze['files'].items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest, f'Frozen file changed: {name}'
        assert args.checkpoint.resolve()==Path('runs/model_bundle').resolve(), 'Use the frozen assets for final testing'
    args.output_dir.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[name]='4'
    predictions=args.output_dir/f'{args.split}-predictions.json'
    score=args.output_dir/f'{args.split}-score.json'
    commands=[['predict.py','--method','submission.py','--checkpoint',str(args.checkpoint),
               '--split',args.split,'--output',str(predictions)],
              ['score_predictions.py','--split',args.split,'--predictions',str(predictions),'--output',str(score)]]
    durations=[]
    for command in commands:
        start=time.perf_counter();subprocess.run([sys.executable,*command],env=env,check=True)
        durations.append(time.perf_counter()-start)
    cpu=platform.processor()
    if sys.platform=='win32':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            cpu=winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
    result={'split':args.split,'prediction_process_seconds':durations[0],
            'scoring_process_seconds':durations[1],'total_seconds':sum(durations),'threads':4,
            'cpu':cpu,'platform':platform.platform(),'python':platform.python_version(),
            'asset_bytes':sum(p.stat().st_size for p in args.checkpoint.iterdir()),
            'timing_includes':'fresh process startup, imports, assets and inputs, all ensemble members, independent scoring',
            'eligibility':'Local measurement only; instructor reference CPU is authoritative.'}
    (args.output_dir/f'{args.split}-resources.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
