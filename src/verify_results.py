"""Reconcile delivered metrics to original labels and persisted predictions."""
import json
import math
import hashlib
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from quality import ROOT, load_data, temporal_split


def main():
    x,y,t=load_data();dev,test=temporal_split(t)
    r=ROOT/'results'
    protocol=json.loads((r/'protocol.json').read_text(encoding='utf-8'))
    lock=json.loads((r/'selection_lock.json').read_text(encoding='utf-8'))
    pred=pd.read_csv(r/'holdout_predictions.csv')
    np.testing.assert_array_equal(pred.row_id.to_numpy(),test)
    np.testing.assert_array_equal(pred.y.to_numpy(),y[test])
    assert t.iloc[dev].max()<t.iloc[test].min()
    for filename,digest in protocol['source_sha256'].items():
        assert hashlib.sha256((ROOT/'data/raw'/filename).read_bytes()).hexdigest()==digest
    cv=pd.read_csv(r/'cv_summary.csv')
    assert lock['champion']==cv[cv.model!='dummy'].sort_values('mean_repeat_ap',ascending=False).iloc[0].model
    metrics=pd.read_csv(r/'holdout_metrics.csv')
    for row in metrics.itertuples():
        np.testing.assert_allclose(average_precision_score(pred.y,pred[row.model]),row.average_precision,atol=1e-12)
    for row in pd.read_csv(r/'inspection_budgets.csv').itertuples():
        count=math.ceil(row.budget*len(test))
        ties=[hashlib.sha256(f'secom-v1:{i}'.encode()).hexdigest() for i in test]
        order=sorted(range(len(test)),key=lambda i:(-pred.iloc[i][row.model],ties[i]))
        hits=int(y[test[np.array(order[:count])]].sum())
        assert count==row.inspected and hits==row.hits
        assert row.hits+row.missed==int(y[test].sum())
        assert row.hits+row.false_alarms==row.inspected
    scores=pred[lock['champion']].to_numpy()
    for row in pd.read_csv(r/'cost_scenarios.csv').itertuples():
        decision=scores>=row.threshold
        fp=int(np.sum(decision & (y[test]==0)));fn=int(np.sum(~decision & (y[test]==1)))
        assert fp==row.fp and fn==row.fn
        assert fp+row.fn_cost*fn==row.cost_units
    model=joblib.load(ROOT/'models/champion.joblib')
    np.testing.assert_allclose(model.predict_proba(x.iloc[test])[:,1],scores,atol=1e-12)
    support=model.named_steps['sensors'].support_
    np.testing.assert_allclose(model.named_steps['imputer'].statistics_,np.nanmedian(x.iloc[dev].to_numpy()[:,support],axis=0))
    # Persist exact deterministic development fold membership for inspection.
    records=[]
    for seed in protocol['seeds']:
        for fold,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(dev,y[dev])):
            assert not set(dev[tr])&set(test)
            records.extend({'seed':seed,'validation_fold':fold,'row_id':int(dev[i])} for i in va)
    pd.DataFrame(records).to_csv(r/'development_fold_manifest.csv',index=False)
    for row in pd.read_csv(r/'development_walk_forward.csv').itertuples():
        assert pd.Timestamp(row.train_end)<pd.Timestamp(row.validation_start)
        assert pd.Timestamp(row.validation_end)<t.iloc[test].min()
    receipt=dict(status='passed',checks=['source hashes','temporal separation','original labels',
       'development-only selection','all model AP values','all 48 budget rows','all cost scenarios',
       'serialized model predictions','training-only imputation medians','walk-forward separation'],
       caveat='Checks establish arithmetic/protocol consistency, not production validity or causal attribution.')
    (r/'validation.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
