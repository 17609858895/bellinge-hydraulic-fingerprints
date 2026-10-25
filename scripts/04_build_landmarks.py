"""Dynamic landmark table for time-to-CSO after storage onset (G71F68Y sump > 1.0 m).

CSO proxy: sump level >= 4.5 m (full-basin plateau; ESSD reports ~5 overflows/yr to Odense River).
Landmarks every 10 min from onset until CSO or event end (max 48 h). Features use only data <= landmark.
Events that recede without CSO are 'cured' within the event: their time-to-CSO is treated as > event horizon
(subdistribution / competing-risk convention), not as non-informative censoring.
"""
import os, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = pd.read_parquet(os.path.join(ROOT, "processed", "panel_5min.parquet"))
E = pd.read_csv(os.path.join(ROOT, "results", "E03_storage_events.csv"), parse_dates=["start", "end"])
H_CSO, STEP, MAXH = 4.5, 2, 48 * 12  # 2 x 5 min = 10 min landmarks

S = P["G71F68Y_LevelPS"].ffill(limit=6)
f05 = P["G71F05R_LevelInlet"].ffill(limit=6); f06 = P["G71F06R_LevelInlet"].ffill(limit=6)
f04 = P["G71F04R_Level2"].ffill(limit=6); gate = P["G71F05R_position"].ffill(limit=6)
b05 = P["G71F05R_LevelBasin"].ffill(limit=6); q68 = P["G71F68Y_Flow"].ffill(limit=6)

rows = []
for eid, ev in E.iterrows():
    a = ev.start
    w0 = a - pd.Timedelta("6h")
    w1 = min(ev.end, a + pd.Timedelta("48h"))
    idx = S.loc[w0:w1].index
    s = S.loc[w0:w1].values
    tc = ev.start + pd.Timedelta(minutes=ev.t_cso) if ev.cso else None
    sl = {k: v.loc[w0:w1].values for k, v in dict(f05=f05, f06=f06, f04=f04, gate=gate, b05=b05, q68=q68).items()}
    i0 = np.searchsorted(idx, a)
    # gate closure time before onset (within 12 h)
    gw = gate.loc[a - pd.Timedelta("12h"):a]
    cl = gw[gw < 0.3]
    tgate = cl.index[0] if len(cl) else a
    for i in range(i0, len(idx), STEP):
        t = idx[i]
        if tc is not None and t >= tc:
            break
        def tr(x, k):
            return x[i] - x[i - k] if i - k >= 0 else np.nan
        h = s[i]
        dh30 = tr(s, 6)
        ttf = (H_CSO - h) / max(dh30 / 30.0, 1e-4) if np.isfinite(dh30) else 2880.0
        seg = slice(i0, i + 1)
        r = dict(eid=eid, t=t, year=t.year, elapsed=(t - a).total_seconds() / 60,
                 since_gate=(t - tgate).total_seconds() / 60,
                 h=h, dh10=tr(s, 2), dh30=dh30, dh60=tr(s, 12), hmax=np.nanmax(s[seg]),
                 ttf_phys=min(max(ttf, 1.0), 2880.0),
                 f05=sl["f05"][i], f05_d30=tr(sl["f05"], 6), f05_max60=np.nanmax(sl["f05"][max(0, i - 12):i + 1]),
                 f05_cum=np.nansum(np.clip(sl["f05"][max(0, i0 - 72):i + 1] - 0.30, 0, None)),
                 f06=sl["f06"][i], f06_d30=tr(sl["f06"], 6), f06_cum=np.nansum(np.clip(sl["f06"][max(0, i0 - 72):i + 1] - 0.36, 0, None)),
                 f04=sl["f04"][i], f04_d30=tr(sl["f04"], 6), f04_cum=np.nansum(np.clip(sl["f04"][max(0, i0 - 72):i + 1] - 0.30, 0, None)),
                 gate=sl["gate"][i], b05=sl["b05"][i], b05_d30=tr(sl["b05"], 6), q68=sl["q68"][i],
                 doy_s=np.sin(2 * np.pi * t.dayofyear / 365.25), doy_c=np.cos(2 * np.pi * t.dayofyear / 365.25),
                 hour_s=np.sin(2 * np.pi * t.hour / 24), hour_c=np.cos(2 * np.pi * t.hour / 24),
                 post2018=int(t >= pd.Timestamp("2018-10-05")),
                 event_cso=int(ev.cso))
        if tc is not None:
            r["T"] = (tc - t).total_seconds() / 60; r["delta"] = 1
        else:
            r["T"] = max((ev.end - t).total_seconds() / 60, 10.0); r["delta"] = 0
        rows.append(r)
L = pd.DataFrame(rows)
L.to_parquet(os.path.join(ROOT, "processed", "landmarks.parquet"))
print(L.shape, L.groupby("event_cso").eid.nunique().to_dict())
print(L.groupby("year").agg(n=("eid", "size"), ev=("eid", "nunique"), cso_ev=("event_cso", lambda x: 0)).T)
print(L.describe().T.round(3).to_string())
