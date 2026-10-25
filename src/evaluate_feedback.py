"""Join delayed final-test labels to retained predictions without retraining."""
import argparse
import json
import sqlite3
from pathlib import Path
import pandas as pd
from sklearn.metrics import average_precision_score
from quality import ROOT


def evaluate(csv,db_path):
    frame=pd.read_csv(csv,dtype={'request_id':str,'unit_id':str})
    if set(frame.columns)!={'request_id','unit_id','label'}:raise ValueError('Expected request_id,unit_id,label')
    if frame.empty or frame.isna().any().any() or not frame.label.isin([0,1]).all():raise ValueError('Nonempty complete labels 0/1 required')
    if frame.duplicated(['request_id','unit_id']).any():raise ValueError('Duplicate feedback keys')
    if not Path(db_path).exists():raise ValueError('Prediction database does not exist')
    db=sqlite3.connect(f'file:{Path(db_path).as_posix()}?mode=ro',uri=True)
    try:stored=[json.loads(r[0]) for r in db.execute('SELECT result FROM requests')]
    finally:db.close()
    records=[dict(request_id=b['request_id'],version=b['version'],**u) for b in stored for u in b['units']]
    if not records:raise ValueError('No retained predictions')
    predictions=pd.DataFrame(records)
    joined=frame.merge(predictions,on=['request_id','unit_id'],how='left',validate='one_to_one',indicator=True)
    if not (joined['_merge']=='both').all():raise ValueError('Unknown/expired prediction keys; no silent dropping')
    reports=[]
    for version,group in joined.groupby('version'):
        selected=group.shadow_priority.astype(bool)
        positives=int(group.label.sum());hits=int(group.loc[selected,'label'].sum())
        reports.append(dict(version=version,labeled_units=len(group),
           retained_units=int((predictions.version==version).sum()),
           label_coverage=len(group)/int((predictions.version==version).sum()),failures=positives,
           shadow_priority_units=int(selected.sum()),hits=hits,missed=positives-hits,
           recall=hits/positives if positives else None,
           precision=hits/int(selected.sum()) if selected.sum() else None,
           average_precision=float(average_precision_score(group.label,group.risk_score)) if group.label.nunique()==2 else None))
    return dict(mode='shadow_only',reports=reports,
         caveat='Metrics cover matched labeled units only. Delayed/missing labels can bias results. Shadow priorities are not actual inspections or yield improvements.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--db',type=Path,default=ROOT/'state/operations.sqlite3')
    p.add_argument('--output',type=Path,default=ROOT/'results/feedback_report.json');args=p.parse_args()
    report=evaluate(args.labels,args.db)
    args.output.parent.mkdir(exist_ok=True,parents=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report,indent=2))
