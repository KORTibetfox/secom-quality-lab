import copy
import json
import shutil
import socket
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pandas as pd
import uvicorn
from api import create_app
from operations import Runtime
from quality import ROOT


class OperationsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT/'tmp').mkdir(exist_ok=True)
        cls.temp=tempfile.TemporaryDirectory(dir=ROOT/'tmp')
        cls.runtime=Runtime(state=Path(cls.temp.name)/'state')
        cls.token='test-token-not-a-live-secret-000000000000'
        cls.app=create_app(cls.runtime,cls.token)
        cls.sock=socket.socket();cls.sock.bind(('127.0.0.1',0));cls.port=cls.sock.getsockname()[1]
        cls.server=uvicorn.Server(uvicorn.Config(cls.app,log_level='critical',access_log=False))
        cls.thread=threading.Thread(target=cls.server.run,kwargs={'sockets':[cls.sock]},daemon=True);cls.thread.start()
        deadline=time.monotonic()+15
        while not cls.server.started:
            if time.monotonic()>deadline:raise RuntimeError('Test API did not start')
            time.sleep(.05)
        values=json.loads(pd.read_csv(ROOT/'data/demo_sensors.csv').head(3).to_json(orient='values'))
        cls.payload={'request_id':'example','inspection_fraction':.1,
                     'units':[{'unit_id':f'unit-{i}','sensors':v} for i,v in enumerate(values)]}
        cls.statuses=[]

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit=True;cls.thread.join(10);cls.sock.close();cls.temp.cleanup()

    def call(self,path='/v1/rank',payload=None,token=True,raw=None):
        data=raw if raw is not None else (json.dumps(payload).encode() if payload else None)
        headers={'Content-Type':'application/json'}
        if token:headers['X-API-Key']=self.token
        req=urllib.request.Request(f'http://127.0.0.1:{self.port}'+path,data=data,headers=headers)
        try:
            with urllib.request.urlopen(req,timeout=10) as response:return response.status,json.load(response)
        except urllib.error.HTTPError as exc:return exc.code,json.load(exc)

    def request(self,key):
        p=copy.deepcopy(self.payload);p['request_id']=key;return p

    def test_healthy_auth_and_shadow_gate(self):
        self.assertEqual(self.call('/health/live',token=False)[0],200)
        self.assertEqual(self.call('/health/ready',token=False)[0],401)
        code,body=self.call('/health/ready');self.assertEqual(code,200)
        self.assertFalse(body['production_approved']);self.assertEqual(body['mode'],'shadow_only')
        self.assertEqual(self.call(payload=self.request('no-auth'),token=False)[0],401)

    def test_replay_and_conflict(self):
        p=self.request('replay')
        c,a=self.call(payload=p);d,b=self.call(payload=p)
        self.assertEqual((c,d),(200,200));self.assertFalse(a['replayed']);self.assertTrue(b['replayed'])
        self.assertEqual(a['units'],b['units']);self.assertEqual(a['inspected'],1)
        p['inspection_fraction']=.5;self.assertEqual(self.call(payload=p)[0],409)

    def test_concurrent_retry_exactly_one_result(self):
        p=self.request('concurrent')
        with ThreadPoolExecutor(2) as pool:responses=list(pool.map(lambda _:self.call(payload=p),range(2)))
        self.assertEqual([r[0] for r in responses],[200,200])
        self.assertEqual(sum(not r[1]['replayed'] for r in responses),1)

    def test_validation_failures(self):
        mutations=[lambda p:p['units'][0]['sensors'].pop(),
                   lambda p:p['units'][0]['sensors'].__setitem__(0,'123'),
                   lambda p:p['units'][0]['sensors'].__setitem__(0,True),
                   lambda p:p['units'][0]['sensors'].__setitem__(0,float('inf')),
                   lambda p:p['units'].append(p['units'][0]),
                   lambda p:p.__setitem__('inspection_fraction',1.1),
                   lambda p:p.__setitem__('extra','forbidden'),
                   lambda p:p.__setitem__('units',[]),
                   lambda p:p.__setitem__('units',[{**p['units'][0],'unit_id':str(i)} for i in range(101)]),
                   lambda p:p['units'][0].__setitem__('sensors',[None]*590)]
        for i,mutate in enumerate(mutations):
            with self.subTest(i=i):
                p=self.request(f'invalid-{i}');mutate(p)
                code,body=self.call(payload=p);self.assertEqual(code,422)
                self.assertNotIn('sensors": [',json.dumps(body))

    def test_oversized_body(self):
        self.assertEqual(self.call(raw=b' '*(2*1024*1024+1))[0],413)

    def test_model_corruption_rejected_before_load(self):
        folder=Path(self.temp.name)/'corrupt'
        shutil.copytree(ROOT/'releases',folder)
        version=json.loads((folder/'active.json').read_text())['version']
        with (folder/version/'model.joblib').open('ab') as f:f.write(b'corrupt')
        with self.assertRaisesRegex(ValueError,'integrity'):Runtime(releases=folder,state=Path(self.temp.name)/'bad-state')

    def test_rank_order_stable_for_unit_ids(self):
        a=self.request('order-a');a['units'][1]['sensors']=a['units'][0]['sensors']
        b=copy.deepcopy(a);b['request_id']='order-b';b['units']=b['units'][::-1]
        aa=self.call(payload=a)[1];bb=self.call(payload=b)[1]
        self.assertEqual({u['unit_id']:u['rank'] for u in aa['units']},{u['unit_id']:u['rank'] for u in bb['units']})

    def test_restart_preserves_replay(self):
        p=self.request('restart');self.call(payload=p)
        fresh=Runtime(state=self.runtime.state)
        self.assertTrue(fresh.score(p)['replayed'])

    def test_storage_lock_returns_503(self):
        db=self.runtime.connect()
        try:
            db.execute('BEGIN IMMEDIATE')
            self.assertEqual(self.call(payload=self.request('locked'))[0],503)
        finally:db.rollback();db.close()

    def test_capacity_rejection(self):
        self.runtime.gate.acquire();self.runtime.gate.acquire()
        try:self.assertEqual(self.call(payload=self.request('busy'))[0],503)
        finally:self.runtime.gate.release();self.runtime.gate.release()

    def test_feedback_keys_coverage_and_single_class(self):
        from evaluate_feedback import evaluate
        p=self.request('feedback');self.call(payload=p)
        path=Path(self.temp.name)/'feedback.csv'
        frame=pd.DataFrame({'request_id':['feedback']*3,'unit_id':[f'unit-{i}' for i in range(3)],'label':[0]*3})
        frame.to_csv(path,index=False)
        report=evaluate(path,self.runtime.db)
        self.assertIsNone(report['reports'][0]['average_precision'])
        self.assertLessEqual(report['reports'][0]['label_coverage'],1.)
        frame.loc[0,'unit_id']='unknown';frame.to_csv(path,index=False)
        with self.assertRaisesRegex(ValueError,'Unknown'):evaluate(path,self.runtime.db)


if __name__=='__main__':unittest.main()
