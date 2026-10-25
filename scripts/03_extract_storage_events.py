"""Reconstruct storage episodes directly from the five-minute sensor panel.

This is a replacement implementation, not recovered historical source code.
No retained event CSV is read during extraction. The retained manifest is used
only by a separate validation step. End is the first non-exceeding timestamp.
Consecutive excursions separated by less than six hours form one episode.
The raw panel is used for boundaries; short-gap filling belongs to landmarks.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

STEP = pd.Timedelta(minutes=5)
GAP = pd.Timedelta(hours=6)

def episode_bounds(s: pd.Series):
    if not s.index.is_monotonic_increasing or s.index.has_duplicates:
        raise ValueError('Sensor timestamps must be unique and increasing.')
    if len(s)>1 and not (s.index.to_series().diff().iloc[1:]==STEP).all():
        raise ValueError('A complete five-minute index is required; represent gaps as NaN.')
    above=s.gt(1.0)  # NaN breaks an excursion; no invented level is inserted.
    starts=s.index[above & ~above.shift(1,fill_value=False)]
    stops=s.index[above & ~above.shift(-1,fill_value=False)]+STEP
    merged=[]
    for a,b in zip(starts,stops):
        # A trailing open excursion has no observed end and is not an episode.
        if b>s.index[-1]:continue
        if merged and a-merged[-1][1]<GAP:merged[-1]=(merged[-1][0],b)
        else:merged.append((a,b))
    return merged

def extract(panel: pd.DataFrame):
    s=panel['G71F68Y_LevelPS']; rows=[]; previous_end=None
    for a,b in episode_bounds(s):
        w=s.loc[a:b]
        hit=w[w>=4.5]
        pre=panel.loc[a-pd.Timedelta(hours=2):a]
        ante=s.loc[a-pd.Timedelta(hours=2):a-STEP]
        gate=panel.loc[a-pd.Timedelta(hours=12):a,'G71F05R_position']
        closed=gate[gate<.3]
        rows.append(dict(start=a,end=b,peak=float(w.max()),dur_h=(b-a).total_seconds()/3600,
            t_above44=int(w.gt(4.4).sum()*5),yr=a.year,cso=bool(len(hit)),
            ante=float(ante.mean()),gate_lead=(a-closed.index[0]).total_seconds()/60 if len(closed) else np.nan,
            t_cso=(hit.index[0]-a).total_seconds()/60 if len(hit) else np.nan,
            pump_pre=float(pre.G71F68Y_Flow.mean()),f05_max=float(pre.G71F05R_LevelInlet.max()),
            f06_max=float(pre.G71F06R_LevelInlet.max()),f04_max=float(pre.G71F04R_Level2.max()),
            gap_prev_h=(a-previous_end).total_seconds()/3600 if previous_end is not None else np.nan))
        previous_end=b
    result=pd.DataFrame(rows)
    if len(result):result['pump_pre']=result['pump_pre'].astype('float32')
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--panel',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    events=extract(pd.read_parquet(args.panel))
    events.iloc[:,:6].to_csv(args.output_dir/'E03_storage_events_raw.csv',index=False)
    events.to_csv(args.output_dir/'E03_storage_events.csv',index=False)
    print(f'Extracted {len(events)} episodes; {events.cso.sum()} reach the 4.5 m proxy threshold.',flush=True)

if __name__=='__main__':main()
