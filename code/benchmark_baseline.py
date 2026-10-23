"""Fresh-process round-1 validation timing on the same local environment."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    env=dict(os.environ)
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='4'
    durations=[]
    commands=[['predict.py','--method','selector.py','--checkpoint','../history/round1/model_bundle',
        '--split','validation','--output','runs/baseline-validation-predictions.json'],
        ['score_predictions.py','--split','validation','--predictions','runs/baseline-validation-predictions.json',
        '--output','runs/baseline-validation-score.json']]
    for cmd in commands:
        tick=time.perf_counter();subprocess.run([sys.executable,*cmd],env=env,check=True);durations.append(time.perf_counter()-tick)
    result={'prediction_process_seconds':durations[0],'scoring_process_seconds':durations[1],
            'total_seconds':sum(durations),'threads':4,'split':'validation','method':'frozen_round1'}
    Path('runs/baseline-validation-resources.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
