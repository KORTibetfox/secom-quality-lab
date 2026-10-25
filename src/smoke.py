"""Live HTTP smoke test and measured local latency; sample data only."""
import json
import time
import urllib.request
import urllib.error
import numpy as np
import pandas as pd
from quality import ROOT


def main():
    token=(ROOT/'state/.api_token').read_text().strip()
    values=json.loads(pd.read_csv(ROOT/'data/demo_sensors.csv').to_json(orient='values'))
    times=[];responses=[]
    for i in range(30):
        payload={'request_id':f'local-smoke-{i:02d}','inspection_fraction':.1,
            'units':[{'unit_id':f'demo-{j:04d}','sensors':row} for j,row in enumerate(values)]}
        req=urllib.request.Request('http://127.0.0.1:8766/v1/rank',data=json.dumps(payload,allow_nan=False).encode(),
            headers={'Content-Type':'application/json','X-API-Key':token})
        start=time.perf_counter()
        with urllib.request.urlopen(req,timeout=15) as response:body=json.load(response)
        times.append((time.perf_counter()-start)*1000);responses.append(body)
        assert body['mode']=='shadow_only' and not body['production_approved']
        assert len(body['units'])==20 and body['inspected']==2
    report=dict(requests=30,rows_per_request=20,latency_scope='end-to-end localhost HTTP incl validation, inference, SQLite commit; serial load',
      cold_first_ms=times[0],warm_p50_ms=float(np.median(times[1:])),warm_p95_ms=float(np.quantile(times[1:],.95)),
      max_ms=max(times),replays=sum(r['replayed'] for r in responses),
      caveat='Single local run, not a production SLA or concurrency capacity test.')
    (ROOT/'results/operations_smoke.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    # Delayed-label demo explicitly uses original holdout labels and demo IDs, not new production data.
    split=pd.read_csv(ROOT/'results/split_manifest.csv');labels=split[split.partition=='future_holdout'].head(20).y.tolist()
    pd.DataFrame({'request_id':['local-smoke-00']*20,'unit_id':[f'demo-{i:04d}' for i in range(20)],'label':labels}).to_csv(ROOT/'data/demo_feedback.csv',index=False)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
