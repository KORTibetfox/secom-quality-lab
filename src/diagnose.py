"""Post-evaluation diagnostics. Never changes v1 champion or thresholds."""
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import TimeSeriesSplit
from quality import ROOT, load_data, temporal_split, pipeline, score_metrics, budget_metrics
from train import candidates


def main():
    x,y,t=load_data();dev,test=temporal_split(t)
    rows=[]
    for fold,(tr,va) in enumerate(TimeSeriesSplit(n_splits=3).split(dev)):
        train_ids=dev[tr];valid_ids=dev[va]
        # Exclude simultaneous test timestamps from training.
        train_ids=train_ids[np.asarray(t.iloc[train_ids] < t.iloc[valid_ids].min())]
        for name,estimator in candidates(2026).items():
            model=pipeline(estimator).fit(x.iloc[train_ids],y[train_ids])
            p=model.predict_proba(x.iloc[valid_ids])[:,1]
            bm=budget_metrics(y[valid_ids],p,valid_ids,.1)
            row=dict(model=name,fold=fold,train_rows=len(train_ids),validation_rows=len(valid_ids),
                     validation_failures=int(y[valid_ids].sum()),
                     train_end=str(t.iloc[train_ids].max()),validation_start=str(t.iloc[valid_ids].min()),
                     validation_end=str(t.iloc[valid_ids].max()),**score_metrics(y[valid_ids],p),
                     recall_at_10=bm['recall'])
            rows.append(row)
            print(f'walk-forward {fold} {name} AP={row["average_precision"]:.3f}',flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'results/development_walk_forward.csv',index=False)
    drift=[]
    for name in x.columns:
        a=x.iloc[dev][name];b=x.iloc[test][name]
        missing_delta=float(b.isna().mean()-a.isna().mean())
        aa=a.dropna();bb=b.dropna()
        ks=float(ks_2samp(aa,bb).statistic) if len(aa)>1 and len(bb)>1 else None
        std=float(aa.std())
        shift=float((bb.mean()-aa.mean())/std) if std>0 and len(bb) else None
        drift.append(dict(sensor=name,development_missing=float(a.isna().mean()),
                          holdout_missing=float(b.isna().mean()),missing_delta=missing_delta,
                          ks_statistic=ks,standardized_mean_shift=shift))
    pd.DataFrame(drift).to_csv(ROOT/'results/sensor_distribution_shift.csv',index=False)
    summary=dict(purpose='Post-evaluation descriptive diagnosis only. v1 selection stays locked.',
                 development_failure_rate=float(y[dev].mean()),holdout_failure_rate=float(y[test].mean()),
                 sensors_with_ks_ge_02=sum(d['ks_statistic'] is not None and d['ks_statistic']>=.2 for d in drift),
                 sensors_missing_delta_ge_10pp=sum(abs(d['missing_delta'])>=.1 for d in drift),
                 causal_limit='Observed feature distribution changes cannot establish why generalization failed; repeated/lot dependence is unknown.')
    (ROOT/'results/diagnostic_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(summary,flush=True)


if __name__=='__main__':main()
