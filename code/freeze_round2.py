"""Promote by development evidence and freeze BEFORE final test scoring."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil


def main():
    selection=json.loads(Path('research_runs/selection.json').read_text())
    assert selection['selected'] is not None,'No promotion: retain original artifact; do not rescore test.'
    assert json.loads(Path('runs/verification.json').read_text())['status']=='passed'
    manifest=Path('runs/freeze.json')
    assert not manifest.exists(),'Already frozen; preserve the original freeze record.'
    target=Path('runs/model_bundle');target.mkdir(parents=True,exist_ok=True)
    for path in Path('research_runs/selected_assets').iterdir():shutil.copy2(path,target/path.name)
    source=['submission.py','round2_model.py','selector.py','features.py','research.py','research_plan.json',
            'train.py','requirements-method.txt','requirements-training.txt','common.py','predict.py','score_predictions.py','check_data.py']
    files=[Path(p) for p in source]+sorted(target.iterdir())
    # Verify all protected original scoring code and packaged data.
    upstream=json.loads(Path('PACKAGE_MANIFEST.json').read_text())
    for name in ('common.py','predict.py','score_predictions.py','check_data.py','requirements.txt'):
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==upstream['files']['code/'+name]
    result={'version':2,'utc':datetime.now(timezone.utc).isoformat(),'selected':selection['selected'],
            'selection_basis':'training internal holdout and official validation only; prior test aggregate exposure disclosed',
            'test_policy':'One final test score after freeze; no switching methods based on test results.',
            'asset_bytes':sum(p.stat().st_size for p in target.iterdir()),
            'files':{p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    assert result['asset_bytes']<=16*2**20
    manifest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
