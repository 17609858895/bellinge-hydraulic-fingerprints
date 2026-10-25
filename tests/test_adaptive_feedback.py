"""Check that overlapping episodes cannot introduce unobserved feedback."""
from pathlib import Path
import importlib.util, sys, unittest
import numpy as np, pandas as pd

S=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(S))
spec=importlib.util.spec_from_file_location('adaptive_feedback',S/'17_adaptive_feedback.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class FeedbackTests(unittest.TestCase):
    def test_feedback_waits_for_arrival(self):
        L=pd.DataFrame({'eid':[0,1,2,3], 't':pd.to_datetime(['2018-01-01','2020-01-01','2020-01-01 00:05','2020-01-01 02:00'],format='mixed'),
            'T':[10.,60.,10.,10.],'delta':[1,1,1,1],'split':['val','test','test','test']})
        mu=np.log([10.,120.,10.,10.])
        bound,_,trace=m.adaptive(L,mu,.1)
        self.assertAlmostEqual(bound[2],10.)
        self.assertEqual(trace['deferred'][0]['pending_episodes'],[1])
        self.assertEqual([x['episode'] for x in trace['feedback']],[2,1])
        self.assertTrue(trace['all_feedback_available'])

if __name__=='__main__':unittest.main()
