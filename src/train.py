"""Fixed v1 protocol. Every reported result is generated from local raw data."""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MLFLOW_ENABLE_ASYNC_LOGGING', 'false')
os.environ.setdefault('MLFLOW_ENABLE_TELEMETRY', 'false')
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
from scipy.stats import hypergeom
from quality import (ROOT, load_data, temporal_split, pipeline, score_metrics,
                     budget_metrics, choose_threshold, threshold_metrics)

SEEDS = [17, 42, 2026]
BUDGETS = [.05, .1, .2, .3, .5, 1.]
FN_COSTS = [5, 20, 100]


def candidates(seed):
    return {
        'dummy': DummyClassifier(strategy='prior'),
        'logistic_c01': LogisticRegression(C=.1, class_weight='balanced', max_iter=2000, random_state=seed),
        'logistic_c1': LogisticRegression(C=1., class_weight='balanced', max_iter=2000, random_state=seed),
        'random_forest': RandomForestClassifier(n_estimators=250, max_features='sqrt', min_samples_leaf=4,
                             class_weight='balanced_subsample', n_jobs=4, random_state=seed),
        'lgbm_7': LGBMClassifier(n_estimators=200, learning_rate=.03, num_leaves=7, min_child_samples=30,
                             reg_lambda=5., verbosity=-1, n_jobs=4, random_state=seed),
        'lgbm_15': LGBMClassifier(n_estimators=200, learning_rate=.03, num_leaves=15, min_child_samples=30,
                             reg_lambda=5., verbosity=-1, n_jobs=4, random_state=seed),
        'lgbm_7_balanced': LGBMClassifier(n_estimators=200, learning_rate=.03, num_leaves=7, min_child_samples=30,
                             reg_lambda=5., class_weight='balanced', verbosity=-1, n_jobs=4, random_state=seed),
    }


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    if args.seed != 2026:
        raise ValueError('v1 protocol uses seed=2026; create a new protocol before changing it.')
    result = ROOT / 'results'
    result.mkdir(exist_ok=True)
    x, y, timestamps = load_data()
    dev, test = temporal_split(timestamps)
    xd, yd = x.iloc[dev], y[dev]
    xt, yt = x.iloc[test], y[test]
    audit = dict(rows=len(x), sensors=x.shape[1], failures=int(y.sum()),
                 passes=int((y==0).sum()), missing_values=int(x.isna().sum().sum()),
                 missing_fraction=float(x.isna().to_numpy().mean()),
                 constant_sensors=int((x.nunique()<=1).sum()),
                 all_missing_sensors=int(x.isna().all().sum()),
                 duplicate_sensor_rows=int(x.duplicated().sum()),
                 duplicate_timestamps=int(timestamps.duplicated().sum()),
                 start=str(timestamps.min()), end=str(timestamps.max()))
    write_json(result/'data_audit.json', audit)
    missing = pd.DataFrame({'sensor':x.columns,'missing_fraction':x.isna().mean().values,
                            'unique_nonmissing':x.nunique().values})
    missing.to_csv(result/'sensor_audit.csv', index=False)
    splits = pd.DataFrame({'row_id':range(len(y)), 'timestamp':timestamps.astype(str), 'y':y,
                            'partition':['development' if i in set(dev) else 'future_holdout' for i in range(len(y))]})
    splits.to_csv(result/'split_manifest.csv', index=False)
    monthly = splits.assign(month=timestamps.dt.to_period('M').astype(str)).groupby('month').agg(
        samples=('y','size'),failures=('y','sum'))
    monthly['failure_rate'] = monthly.failures/monthly.samples
    monthly.to_csv(result/'monthly_quality.csv')
    manifest = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data/raw').iterdir()}
    protocol = dict(version='v1-fixed-2026-10-22', primary_selection='development mean repeated-OOF average precision',
                    seeds=SEEDS, folds=5, budgets=BUDGETS, fn_costs=FN_COSTS,
                    temporal_boundary=str(timestamps.iloc[test[0]]),
                    development_rows=len(dev), development_failures=int(yd.sum()),
                    holdout_rows=len(test), holdout_failures=int(yt.sum()),
                    candidates=list(candidates(2026)),
                    audit_note='Global audit is descriptive only; selection/imputation fit inside training folds.',
                    caveat='Repeated OOF selects candidate; development scores have model-selection optimism. Holdout is final evaluation.',
                    source_sha256=manifest)
    write_json(result/'protocol.json', protocol)
    print('AUDIT',json.dumps(audit),flush=True)
    # Pure file-backed local experiment tracking. No external MLflow server.
    import mlflow
    os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'
    mlflow.set_tracking_uri((ROOT/'mlruns').as_uri())
    mlflow.set_experiment('SECOM-quality-v1')
    names = list(candidates(2026))
    oof = {name:[] for name in names}
    cv_rows = []
    for seed in SEEDS:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for name in names:
            scores = np.full(len(dev),np.nan)
            start = time.perf_counter()
            for fold, (tr, va) in enumerate(cv.split(xd, yd)):
                model = pipeline(candidates(seed)[name])
                model.fit(xd.iloc[tr], yd[tr])
                p = model.predict_proba(xd.iloc[va])[:,1]
                scores[va] = p
                metrics = score_metrics(yd[va],p)
                cv_rows.append(dict(model=name,seed=seed,fold=fold,**metrics))
            assert np.isfinite(scores).all()
            oof[name].append(scores)
            elapsed = time.perf_counter()-start
            metrics = score_metrics(yd,scores)
            with mlflow.start_run(run_name=f'{name}-seed-{seed}'):
                mlflow.log_params({'model':name,'seed':seed,'folds':5,'partition':'development_oof',
                                   'preprocessing':'train-only filter+median+missing-indicator+scale'})
                mlflow.log_metrics({**metrics,'duration_s':elapsed})
                mlflow.log_artifact(str(result/'protocol.json'))
            print(f'CV {name} seed={seed}: AP={metrics["average_precision"]:.4f}, {elapsed:.1f}s',flush=True)
    pd.DataFrame(cv_rows).to_csv(result/'cv_folds.csv',index=False)
    cv_summary = []
    oof_table = pd.DataFrame({'row_id':dev,'y':yd})
    for name in names:
        repeat_ap = [score_metrics(yd,p)['average_precision'] for p in oof[name]]
        avg = np.mean(oof[name],axis=0)
        oof_table[name] = avg
        row = dict(model=name, mean_repeat_ap=float(np.mean(repeat_ap)),
                   sd_repeat_ap=float(np.std(repeat_ap,ddof=1)), **score_metrics(yd,avg))
        row.update({f'recall_at_{int(b*100)}':budget_metrics(yd,avg,dev,b)['recall'] for b in [.1,.2]})
        cv_summary.append(row)
    summary = pd.DataFrame(cv_summary).sort_values('mean_repeat_ap',ascending=False)
    summary.to_csv(result/'cv_summary.csv',index=False)
    oof_table.to_csv(result/'development_oof.csv',index=False)
    champion = summary.loc[summary.model != 'dummy'].iloc[0]['model']
    # Persist selection BEFORE evaluating any holdout model.
    selection = dict(champion=champion, selected_by='development mean_repeat_ap',
                     locked_at=datetime.now(timezone.utc).isoformat(), thresholds={})
    champion_oof = oof_table[champion].to_numpy()
    for ratio in FN_COSTS:
        threshold, cost = choose_threshold(yd,champion_oof,ratio)
        selection['thresholds'][str(ratio)] = dict(threshold=threshold,development_cost=cost)
    write_json(result/'selection_lock.json',selection)
    test_table = pd.DataFrame({'row_id':test,'timestamp':timestamps.iloc[test].astype(str).to_numpy(),'y':yt})
    holdout_rows, budget_rows = [], []
    final = None
    for name in names:
        start = time.perf_counter()
        model = pipeline(candidates(2026)[name]).fit(xd,yd)
        p = model.predict_proba(xt)[:,1]
        test_table[name] = p
        holdout_rows.append(dict(model=name, selected=name==champion, **score_metrics(yt,p),
                                 fit_predict_s=time.perf_counter()-start))
        for b in BUDGETS:
            bm = budget_metrics(yt,p,test,b)
            lo,hi = hypergeom.ppf([.025,.975],len(yt),int(yt.sum()),bm['inspected'])
            budget_rows.append(dict(model=name, **bm,random_95_low_hits=int(lo),random_95_high_hits=int(hi)))
        if name == champion:
            final = model
            joblib.dump(model,ROOT/'models/champion.joblib')
    # Label-free anomaly comparator: fit exclusively on NORMAL development samples.
    # Supplemental only: not part of champion selection, scores are NOT probabilities.
    anomaly = pipeline(IsolationForest(n_estimators=250, contamination='auto',random_state=2026,n_jobs=4))
    anomaly.fit(xd.iloc[np.flatnonzero(yd==0)])
    anomaly_scores = -anomaly.score_samples(xt)
    test_table['isolation_forest'] = anomaly_scores
    holdout_rows.append(dict(model='isolation_forest',selected=False,**score_metrics(yt,anomaly_scores),fit_predict_s=None))
    for b in BUDGETS:
        bm=budget_metrics(yt,anomaly_scores,test,b)
        lo,hi=hypergeom.ppf([.025,.975],len(yt),int(yt.sum()),bm['inspected'])
        budget_rows.append(dict(model='isolation_forest',**bm,random_95_low_hits=int(lo),random_95_high_hits=int(hi)))
    test_table.to_csv(result/'holdout_predictions.csv',index=False)
    pd.DataFrame(holdout_rows).to_csv(result/'holdout_metrics.csv',index=False)
    pd.DataFrame(budget_rows).to_csv(result/'inspection_budgets.csv',index=False)
    cost_rows = []
    cp = test_table[champion].to_numpy()
    for ratio in FN_COSTS:
        row = threshold_metrics(yt,cp,selection['thresholds'][str(ratio)]['threshold'],ratio)
        row['inspect_none_cost'] = int(yt.sum())*ratio
        row['inspect_all_cost'] = int((yt==0).sum())
        cost_rows.append(row)
    pd.DataFrame(cost_rows).to_csv(result/'cost_scenarios.csv',index=False)
    # Conditional fixed-model bootstrap; does not capture training/model-selection uncertainty.
    rng=np.random.default_rng(2026)
    boot=[]
    for _ in range(2000):
        idx=rng.integers(0,len(yt),len(yt))
        if len(np.unique(yt[idx]))<2: continue
        ids=np.arange(len(idx)) # deterministic secondary order for duplicate bootstrap observations
        boot.append([score_metrics(yt[idx],cp[idx])['average_precision'],
                     budget_metrics(yt[idx],cp[idx],ids,.1)['recall']])
    bounds=np.quantile(boot,[.025,.975],axis=0)
    write_json(result/'uncertainty.json',dict(method='2000 iid row bootstrap draws, fixed fitted champion; 95% percentile interval',
               caveat='Conditional on fitted model and this short historical window; not training uncertainty or production confidence.',
               valid_draws=len(boot), average_precision=bounds[:,0].tolist(),recall_at_10=bounds[:,1].tolist()))
    # Interpretation uses development data only, not holdout labels.
    # Single held-out dev fold; this is supplemental, not a causal ranking.
    tr,va=next(StratifiedKFold(5,shuffle=True,random_state=2026).split(xd,yd))
    explain_model=pipeline(candidates(2026)[champion]).fit(xd.iloc[tr],yd[tr])
    importance=permutation_importance(explain_model,xd.iloc[va],yd[va],scoring='average_precision',
                                    n_repeats=3,random_state=2026,n_jobs=1)
    pd.DataFrame({'sensor':x.columns,'ap_drop_mean':importance.importances_mean,
                   'ap_drop_sd':importance.importances_std}).sort_values('ap_drop_mean',ascending=False).to_csv(
                       result/'development_permutation_importance.csv',index=False)
    names_out=final.named_steps['imputer'].get_feature_names_out(
        x.columns[final.named_steps['sensors'].support_])
    write_json(ROOT/'models/model_card.json',dict(model=champion,training_rows=len(dev),
        original_features=list(x.columns),transformed_features=list(names_out),
        score_semantics='Uncalibrated ranking score; never claim calibrated defect probability.',
        intended_use='Historical offline inspection prioritization proof of concept.',
        excluded_use='Production release decisions, lifetime/reliability prediction, causal process optimization.',
        known_limits=['2008 anonymized measurements','17 holdout failures','no equipment/lot IDs',
                      'feature availability before final test is not established','no external-fab validation'],
        thresholds=selection['thresholds']))
    versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','scikit-learn','scipy','lightgbm','mlflow-skinny','matplotlib','joblib']}
    write_json(result/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),packages=versions))
    with mlflow.start_run(run_name=f'FINAL-{champion}'):
        mlflow.log_params({'model':champion,'partition':'future_holdout','selection':'locked development AP'})
        mlflow.log_metrics(score_metrics(yt,cp))
        for filename in ['selection_lock.json','holdout_metrics.csv','inspection_budgets.csv','uncertainty.json']:
            mlflow.log_artifact(str(result/filename))
    print('COMPLETE champion='+champion,flush=True)
    print(pd.DataFrame(holdout_rows).to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
