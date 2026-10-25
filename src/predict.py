"""Local batch inference with strict schema checks; no data leaves this PC."""
import argparse
import json
import math
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from quality import ROOT, ranked_indices


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True, help='CSV containing sensor_000 ... sensor_589')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--budget', type=float, default=.1)
    args=p.parse_args()
    if not 0<=args.budget<=1: p.error('budget must be within [0,1]')
    x=pd.read_csv(args.input)
    schema=json.loads((ROOT/'models/model_card.json').read_text(encoding='utf-8'))['original_features']
    if list(x.columns)!=schema: raise ValueError('CSV sensor columns/order must exactly match model_card schema')
    if len(x)==0: raise ValueError('No rows to score')
    a=x.to_numpy(dtype=float)
    if np.isinf(a).any(): raise ValueError('Infinite value in input')
    model=joblib.load(ROOT/'models/champion.joblib')
    scores=model.predict_proba(x)[:,1]
    order=ranked_indices(scores,range(len(x)))
    selected=np.zeros(len(x),dtype=bool)
    selected[order[:math.ceil(len(x)*args.budget)]]=True
    rank=np.empty(len(x),dtype=int); rank[order]=np.arange(1,len(x)+1)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'row_id':range(len(x)),'risk_score':scores,'inspection_rank':rank,
                  'inspect':selected}).to_csv(args.output,index=False)
    print(f'Scored {len(x)} rows; inspect {int(selected.sum())}. Scores are uncalibrated, not probabilities.')


if __name__=='__main__': main()
