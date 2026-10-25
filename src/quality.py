"""Leakage-aware SECOM loading, preprocessing, ranking, and scoring."""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]


def load_data(folder=None):
    folder = Path(folder or ROOT / 'data/raw')
    x = pd.read_csv(folder / 'secom.data', sep=r'\s+', header=None)
    labels = pd.read_csv(folder / 'secom_labels.data', sep=r'\s+', header=None)
    if x.shape != (1567, 590) or len(labels) != len(x):
        raise ValueError(f'Unexpected original SECOM shape: {x.shape}, labels={len(labels)}')
    if not set(labels[0].unique()) <= {-1, 1}:
        raise ValueError('Unknown label; expected -1=pass, +1=fail')
    x.columns = [f'sensor_{i:03d}' for i in range(x.shape[1])]
    y = (labels[0] == 1).astype(int).to_numpy()
    timestamps = pd.to_datetime(labels[1], format='%d/%m/%Y %H:%M:%S')
    if timestamps.isna().any() or np.isinf(x.to_numpy()).any():
        raise ValueError('Invalid timestamp or infinite value')
    return x, y, timestamps


def temporal_split(timestamps, fraction=.8):
    """Keep all equal boundary timestamps together in the later partition."""
    order = np.argsort(np.asarray(timestamps), kind='stable')
    cut = int(len(order) * fraction)
    while cut > 0 and timestamps.iloc[order[cut-1]] == timestamps.iloc[order[cut]]:
        cut -= 1
    if cut == 0 or cut == len(order):
        raise ValueError('Empty temporal partition')
    return order[:cut], order[cut:]


class SensorFilter(TransformerMixin, BaseEstimator):
    """All selection statistics are learned ONLY inside each training fold."""
    def __init__(self, max_missing=.5):
        self.max_missing = max_missing

    def fit(self, X, y=None):
        a = np.asarray(X, dtype=float)
        missing = np.isnan(a).mean(axis=0)
        unique = np.array([len(np.unique(col[~np.isnan(col)])) for col in a.T])
        self.support_ = (missing <= self.max_missing) & (unique > 1)
        self.n_features_in_ = a.shape[1]
        if not self.support_.any():
            raise ValueError('No usable training sensors')
        return self

    def transform(self, X):
        a = np.asarray(X, dtype=float)
        if a.shape[1] != self.n_features_in_:
            raise ValueError('Sensor schema changed')
        return a[:, self.support_]


def pipeline(estimator):
    return Pipeline([
        ('sensors', SensorFilter()),
        ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
        ('scale', StandardScaler()),
        ('model', estimator),
    ])


def ranked_indices(scores, ids):
    """Deterministic score-tie order independent of outcome and time order."""
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite model score')
    tie = np.array([hashlib.sha256(f'secom-v1:{i}'.encode()).hexdigest() for i in ids])
    return np.lexsort((tie, -scores))


def budget_metrics(y, scores, ids, fraction):
    y = np.asarray(y)
    n = len(y)
    k = min(n, max(0, math.ceil(n * fraction)))
    chosen = ranked_indices(scores, ids)[:k]
    hits = int(y[chosen].sum())
    positives = int(y.sum())
    expected = k * positives / n
    return dict(budget=fraction, inspected=k, actual_fraction=k/n, hits=hits,
                missed=positives-hits, false_alarms=k-hits,
                recall=hits/positives if positives else 0.,
                precision=hits/k if k else 0., random_expected_hits=expected,
                lift=hits/expected if expected else 0.)


def score_metrics(y, scores):
    return dict(average_precision=float(average_precision_score(y, scores)),
                roc_auc=float(roc_auc_score(y, scores)), prevalence=float(np.mean(y)))


def choose_threshold(y, scores, fn_cost):
    """OOF empirical cost; not Bayes probability threshold (uncalibrated scores)."""
    thresholds = np.r_[np.nextafter(np.max(scores), np.inf), np.unique(scores)]
    costs = []
    for t in thresholds:
        tn, fp, fn, tp = confusion_matrix(y, scores >= t, labels=[0, 1]).ravel()
        costs.append(fp + fn_cost * fn)
    # Lowest cost, then largest threshold (fewer inspections at a tie).
    best = min(range(len(costs)), key=lambda i: (costs[i], -thresholds[i]))
    return float(thresholds[best]), float(costs[best])


def threshold_metrics(y, scores, threshold, fn_cost):
    tn, fp, fn, tp = confusion_matrix(y, scores >= threshold, labels=[0, 1]).ravel()
    return dict(threshold=threshold, fn_cost=fn_cost, tn=int(tn), fp=int(fp),
                fn=int(fn), tp=int(tp), cost_units=int(fp + fn*fn_cost),
                inspected=int(tp+fp))
