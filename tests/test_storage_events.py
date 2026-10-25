"""Boundary-condition tests for the documented event definition."""
from pathlib import Path
import importlib.util,unittest
import numpy as np,pandas as pd
path=Path(__file__).resolve().parents[1]/'scripts/03_extract_storage_events.py'
spec=importlib.util.spec_from_file_location('storage_events',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class StorageDefinitionTests(unittest.TestCase):
    def series(self,n=240):
        return pd.Series(0.,index=pd.date_range('2020-01-01',periods=n,freq='5min'))

    def test_threshold_and_end_timestamp(self):
        s=self.series();s.iloc[2:5]=[1.,1.01,1.]
        self.assertEqual(module.episode_bounds(s),[(s.index[3],s.index[4])])

    def test_six_hour_gap_is_not_merged(self):
        s=self.series();s.iloc[1]=2.;s.iloc[74]=2.
        self.assertEqual(module.episode_bounds(s),[(s.index[1],s.index[2]),(s.index[74],s.index[75])])
        s.iloc[74]=0.;s.iloc[73]=2.
        self.assertEqual(module.episode_bounds(s),[(s.index[1],s.index[74])])

    def test_missing_and_terminal_open_excursion(self):
        s=self.series();s.iloc[1:4]=[2.,np.nan,3.];s.iloc[-1]=2.
        self.assertEqual(module.episode_bounds(s),[(s.index[1],s.index[4])])
        self.assertTrue(np.isnan(s.iloc[2]))

    def test_proxy_threshold_and_arrival(self):
        s=self.series();s.iloc[2:5]=[1.2,4.5,4.4]
        p=pd.DataFrame({'G71F68Y_LevelPS':s})
        for c in ['G71F05R_position','G71F68Y_Flow','G71F05R_LevelInlet','G71F06R_LevelInlet','G71F04R_Level2']:p[c]=0.
        e=module.extract(p).iloc[0]
        self.assertTrue(e.cso);self.assertEqual(e.t_cso,5.);self.assertEqual(e.t_above44,5)
        self.assertEqual(e.end,s.index[5])

if __name__=='__main__':unittest.main()
