import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from quality import SensorFilter, pipeline, budget_metrics, choose_threshold, threshold_metrics, temporal_split, load_data


class QualityTests(unittest.TestCase):
    def test_train_only_preprocessing(self):
        train=np.array([[1.,np.nan,0.],[3.,np.nan,1.],[5.,np.nan,0.],[7.,np.nan,1.]])
        p=pipeline(LogisticRegression()).fit(train,[0,1,0,1])
        median=p.named_steps['imputer'].statistics_.copy()
        p.predict_proba([[100000.,2.,0.]])
        np.testing.assert_array_equal(p.named_steps['sensors'].support_,[True,False,True])
        np.testing.assert_array_equal(p.named_steps['imputer'].statistics_,median)
        self.assertEqual(median[0],4.)

    def test_budget_exact_and_ties(self):
        y=np.array([1,0,1,0,0])
        a=budget_metrics(y,np.ones(5),np.arange(5),.3)
        b=budget_metrics(y,np.ones(5),np.arange(5),.3)
        self.assertEqual(a,b)
        self.assertEqual(a['inspected'],2)
        self.assertEqual(a['hits']+a['missed'],2)
        self.assertAlmostEqual(a['random_expected_hits'],.8)
        self.assertEqual(budget_metrics(y,np.ones(5),range(5),1.)['hits'],2)

    def test_cost_and_threshold(self):
        y=np.array([0,1,0,1]); p=np.array([.2,.8,.6,.4])
        t,cost=choose_threshold(y,p,5)
        result=threshold_metrics(y,p,t,5)
        self.assertEqual(result['cost_units'],cost)
        self.assertEqual(result['cost_units'],result['fp']+5*result['fn'])
        self.assertEqual(result['tn']+result['fp']+result['fn']+result['tp'],4)
        self.assertEqual(cost,1)

    def test_time_boundary_ties(self):
        t=pd.Series(pd.to_datetime(['2008-01-01','2008-01-02','2008-01-02','2008-01-03']))
        a,b=temporal_split(t,.5)
        self.assertLess(t.iloc[a].max(),t.iloc[b].min())

    def test_original_data_split(self):
        x,y,t=load_data(); a,b=temporal_split(t)
        self.assertEqual(x.shape,(1567,590))
        self.assertEqual(int(y.sum()),104)
        self.assertFalse(set(a)&set(b))
        self.assertEqual(len(a)+len(b),len(y))
        self.assertLess(t.iloc[a].max(),t.iloc[b].min())
        self.assertEqual(int(y[b].sum()),17)


if __name__=='__main__': unittest.main()
