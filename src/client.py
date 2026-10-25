"""Example real HTTP client, stable IDs and request ID required for retries."""
import argparse
import json
import urllib.request
from pathlib import Path
import pandas as pd
from quality import ROOT


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,default=ROOT/'data/demo_sensors.csv')
    p.add_argument('--output',type=Path,default=ROOT/'results/api_demo.json')
    p.add_argument('--request-id',default='demo-v2-001')
    p.add_argument('--port',type=int,default=8766)
    args=p.parse_args()
    df=pd.read_csv(args.input)
    expected=[f'sensor_{i:03d}' for i in range(590)]
    if list(df.columns)!=expected:raise ValueError('CSV column names/order must match sensor_000 through sensor_589')
    # Use actual persistent MES IDs in an integration; these are demo-only IDs.
    rows=json.loads(df.to_json(orient='values'))
    payload={'request_id':args.request_id,'inspection_fraction':.1,
             'units':[{'unit_id':f'demo-{i:04d}','sensors':row} for i,row in enumerate(rows)]}
    token=(ROOT/'state/.api_token').read_text(encoding='utf-8').strip()
    request=urllib.request.Request(f'http://127.0.0.1:{args.port}/v1/rank',
      data=json.dumps(payload,allow_nan=False).encode(),headers={'Content-Type':'application/json','X-API-Key':token})
    with urllib.request.urlopen(request,timeout=15) as response:result=json.load(response)
    args.output.parent.mkdir(exist_ok=True,parents=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(f"Shadow response: {len(result['units'])} units, {result['inspected']} priorities, replayed={result['replayed']}")


if __name__=='__main__':main()
