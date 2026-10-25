"""Export a new immutable local shadow release, never overwrite a release."""
import hashlib
import importlib.metadata
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from quality import ROOT, load_data


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_release():
    releases=ROOT/'releases';releases.mkdir(exist_ok=True)
    lock=json.loads((ROOT/'results/selection_lock.json').read_text(encoding='utf-8'))
    version='secom-v1-'+sha(ROOT/'models/champion.joblib')[:12]
    target=releases/version
    if target.exists():
        raise FileExistsError(f'Immutable release already exists: {version}')
    staging=Path(tempfile.mkdtemp(prefix='.staging-',dir=releases))
    try:
        for src,name in [('models/champion.joblib','model.joblib'),('models/model_card.json','model_card.json'),
                         ('results/selection_lock.json','selection_lock.json'),('results/protocol.json','protocol.json'),
                         ('results/holdout_metrics.csv','holdout_metrics.csv'),('requirements-lock.txt','requirements-lock.txt')]:
            shutil.copy2(ROOT/src,staging/name)
        x,_,_=load_data()
        split=__import__('pandas').read_csv(ROOT/'results/split_manifest.csv')
        dev=split[split.partition=='development'].row_id.to_numpy()
        a=x.iloc[dev].to_numpy()
        low=np.nanquantile(a,.01,axis=0);high=np.nanquantile(a,.99,axis=0)
        (staging/'reference.json').write_text(json.dumps({'missing_rate':np.isnan(a).mean(axis=0).tolist(),
             'p01':low.tolist(),'p99':high.tolist()},allow_nan=False),encoding='utf-8')
        files={p.name:sha(p) for p in staging.iterdir()}
        manifest=dict(version=version,created_at=datetime.now(timezone.utc).isoformat(),
           mode='shadow_only',production_approved=False,model=lock['champion'],files=files,
           schema_version='secom-590-v1',python='3.12',packages={n:importlib.metadata.version(n) for n in
                     ['scikit-learn','numpy','pandas','lightgbm','joblib']},
           gate_reason='Temporal holdout AP 0.0666; only 2/17 failures at 10% inspection. No fab validation.')
        (staging/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        staging.rename(target)
        pointer={'version':version,'manifest_sha256':sha(target/'manifest.json')}
        tmp=releases/'active.tmp';tmp.write_text(json.dumps(pointer,indent=2),encoding='utf-8')
        os.replace(tmp,releases/'active.json')
        return pointer
    except Exception:
        # Retain failed staging for diagnosis, never activate it.
        raise


if __name__=='__main__': print(json.dumps(export_release(),indent=2))
