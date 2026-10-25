"""Strict local inference with immutable releases and transactional idempotency."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import math
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from quality import ROOT, ranked_indices
from release import sha


class ConflictError(ValueError): pass
class BusyError(RuntimeError): pass


class Runtime:
    def __init__(self, releases=None, state=None):
        releases=Path(releases or ROOT/'releases')
        self.state=Path(state or ROOT/'state');self.state.mkdir(parents=True,exist_ok=True)
        pointer=json.loads((releases/'active.json').read_text(encoding='utf-8'))
        version=pointer['version']
        if not version.replace('-','').isalnum(): raise ValueError('Invalid release version')
        folder=releases/version
        if sha(folder/'manifest.json')!=pointer['manifest_sha256']: raise ValueError('Manifest integrity failed')
        self.manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        if self.manifest['mode']!='shadow_only' or self.manifest['production_approved']:
            raise ValueError('This runtime only accepts shadow-only artifacts')
        for name,digest in self.manifest['files'].items():
            if Path(name).name!=name or sha(folder/name)!=digest: raise ValueError('Artifact integrity failed')
        for name,version in self.manifest['packages'].items():
            if importlib.metadata.version(name)!=version: raise ValueError(f'Runtime package mismatch: {name}')
        # joblib is executable serialization: only load trusted local release files after checking them.
        self.model=joblib.load(folder/'model.joblib')
        card=json.loads((folder/'model_card.json').read_text(encoding='utf-8'))
        self.schema=card['original_features']
        self.reference=json.loads((folder/'reference.json').read_text(encoding='utf-8'))
        self.version=self.manifest['version']
        self.gate=threading.BoundedSemaphore(2)
        self.db=self.state/'operations.sqlite3'
        with self.transaction() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE IF NOT EXISTS requests (key TEXT PRIMARY KEY, digest TEXT NOT NULL, created TEXT NOT NULL, result TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, created TEXT NOT NULL, kind TEXT NOT NULL, duration_ms REAL, version TEXT)')

    def connect(self):
        conn=sqlite3.connect(self.db,timeout=2)
        conn.execute('PRAGMA synchronous=FULL')
        return conn

    @contextmanager
    def transaction(self):
        conn=self.connect()
        try:
            with conn:yield conn
        finally:conn.close()

    def score(self, request):
        if not self.gate.acquire(blocking=False): raise BusyError('Inference capacity reached; retry later')
        try: return self._score(request)
        finally:self.gate.release()

    def _score(self, request):
        # The same strict validation is used by HTTP and batch clients.
        from api import RankRequest
        request=RankRequest.model_validate(request).model_dump()
        key=request.pop('request_id')
        digest=hashlib.sha256(json.dumps({'request':request,'version':self.version},sort_keys=True,allow_nan=False).encode()).hexdigest()
        start=time.perf_counter()
        db=self.connect()
        try:
            db.execute('BEGIN IMMEDIATE')
            old=db.execute('SELECT digest,result FROM requests WHERE key=?',(key,)).fetchone()
            if old:
                if old[0]!=digest: raise ConflictError('Request ID already used with a different payload or release')
                result=json.loads(old[1]);result['replayed']=True
                db.execute('INSERT INTO events(created,kind,duration_ms,version) VALUES(?,?,?,?)',
                  (datetime.now(timezone.utc).isoformat(),'replay',(time.perf_counter()-start)*1000,self.version))
                db.commit();return result
            units=request['units'];ids=[u['unit_id'] for u in units]
            a=np.array([u['sensors'] for u in units],dtype=float)
            usable=self.model.named_steps['sensors'].support_
            if np.isnan(a[:,usable]).mean(axis=1).max()>.5:
                raise ValueError('A unit has >50% missing usable sensors; retain existing inspection policy')
            scores=self.model.predict_proba(pd.DataFrame(a,columns=self.schema))[:,1]
            if not np.isfinite(scores).all():raise ValueError('Model returned invalid scores')
            k=math.ceil(len(a)*request['inspection_fraction'])
            order=ranked_indices(scores,ids);ranks=np.empty(len(a),int);ranks[order]=np.arange(1,len(a)+1)
            missing=np.isnan(a).mean(axis=0)
            ref=np.array(self.reference['missing_rate'])
            low=np.array(self.reference['p01']);high=np.array(self.reference['p99'])
            valid=np.isfinite(a)
            denom=int(valid.sum())
            outside=float(((a<low)|(a>high)).sum()/denom) if denom else 0.
            alerts=[]
            if len(a)<30: alerts.append('SMALL_BATCH_MONITORING_UNSTABLE')
            if np.sum(abs(missing-ref)>.2)>20: alerts.append('MISSINGNESS_SHIFT_REVIEW')
            if outside>.1:alerts.append('RANGE_SHIFT_REVIEW')
            now=datetime.now(timezone.utc).isoformat()
            result=dict(request_id=key,version=self.version,schema_version='secom-590-v1',
                mode='shadow_only',production_approved=False,replayed=False,created_at=now,
                inspected=k,actual_fraction=k/len(a),score_semantics='uncalibrated_ranking_score',
                alerts=alerts,monitoring=dict(out_of_reference_fraction=outside,
                   missing_shift_sensors=int(np.sum(abs(missing-ref)>.2)),batch_size=len(a)),
                units=[dict(unit_id=uid,risk_score=float(s),rank=int(rank),shadow_priority=bool(rank<=k))
                       for uid,s,rank in zip(ids,scores,ranks)],duration_ms=(time.perf_counter()-start)*1000)
            db.execute('INSERT INTO requests VALUES(?,?,?,?)',(key,digest,now,json.dumps(result,allow_nan=False)))
            db.execute('INSERT INTO events(created,kind,duration_ms,version) VALUES(?,?,?,?)',
                       (now,'scored',result['duration_ms'],self.version))
            # Bounded local retention: replay keys expire with retained requests.
            db.execute('DELETE FROM requests WHERE key NOT IN (SELECT key FROM requests ORDER BY created DESC LIMIT 1000)')
            db.execute('DELETE FROM events WHERE id NOT IN (SELECT id FROM events ORDER BY id DESC LIMIT 10000)')
            db.commit();return result
        except sqlite3.OperationalError as exc:
            db.rollback();raise BusyError('Local state unavailable; preserve existing inspection policy') from exc
        except Exception:
            db.rollback();raise
        finally:db.close()

    def health(self):
        with self.transaction() as db:db.execute('SELECT 1').fetchone()
        return dict(status='ready',version=self.version,mode='shadow_only',production_approved=False)

    def metrics(self):
        with self.transaction() as db:
            rows=db.execute('SELECT kind,count(*),avg(duration_ms),max(duration_ms) FROM events GROUP BY kind').fetchall()
            count=db.execute('SELECT count(*) FROM requests').fetchone()[0]
        return dict(version=self.version,retained_requests=count,events=[dict(kind=r[0],count=r[1],mean_ms=r[2],max_ms=r[3]) for r in rows])

    def event(self,kind):
        try:
            with self.transaction() as db:
                db.execute('INSERT INTO events(created,kind,version) VALUES(?,?,?)',
                    (datetime.now(timezone.utc).isoformat(),kind,self.version))
                db.execute('DELETE FROM events WHERE id NOT IN (SELECT id FROM events ORDER BY id DESC LIMIT 10000)')
        except sqlite3.Error:
            # State outage is returned by readiness/rank, never hide the original error.
            pass
