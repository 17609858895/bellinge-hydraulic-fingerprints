"""Rain-adjusted consequence of the 2018 throttle-pipe swap.
Episodes are defined from gauge rainfall only (mean of 5425/5427, aligned to SCADA clock): >= 6 h dry gap,
total >= 2 mm. Confounders: rain depth/intensity/duration at both gauges, antecedent rain, season, pre-event sump.
Outcomes include a negative control (G71F05R inlet peak, upstream of the swap; should not change).
Estimators: (i) counterfactual GBM trained pre, applied post; (ii) DML partialling-out (5x5 cross-fit);
(iii) placebo-in-time breaks inside the pre-period (training from 2015-06, after the 2015 cast)."""
import os, sys, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import KFold
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")
OFF = pd.Timedelta(minutes=int(sys.argv[1]) if len(sys.argv) > 1 else 60)   # SCADA = UTC + OFF
P = pd.read_parquet(os.path.join(ROOT, "processed", "panel_5min.parquet")).loc["2011-01-01":]
Rn = pd.read_parquet(os.path.join(ROOT, "processed", "rain_5min_utc.parquet")); Rn.index = Rn.index + OFF
Rn = Rn.reindex(P.index)
r = Rn["mean"].fillna(0)
wet = r > 0
# rain events: merge rainy 5-min steps separated by < 6 h
t_wet = r.index[wet.values]
gaps = np.diff(t_wet.values).astype("timedelta64[m]").astype(int)
starts = [t_wet[0]] + list(t_wet[1:][gaps >= 360]); ends = list(t_wet[:-1][gaps >= 360]) + [t_wet[-1]]
rows = []
for a, b in zip(starts, ends):
    seg = Rn.loc[a:b]
    tot = seg["mean"].sum()
    if tot < 2.0:
        continue
    win = P.loc[a: b + pd.Timedelta("12h")]
    ante = Rn.loc[a - pd.Timedelta("7D"): a - pd.Timedelta("5min")]
    rr = seg["mean"].fillna(0)
    rows.append(dict(start=a, dur_h=(b - a).total_seconds() / 3600 + 1 / 12, depth=tot,
                     d5425=seg["5425"].sum(), d5427=seg["5427"].sum(),
                     i5=rr.max() * 12, i30=rr.rolling(6).sum().max() * 2, i60=rr.rolling(12).sum().max(),
                     ante24=ante["mean"].iloc[-288:].sum(), ante7d=ante["mean"].sum(),
                     gauge_ok=float(seg[["5425", "5427"]].notna().all(axis=1).mean()),
                     doy_s=np.sin(2 * np.pi * a.dayofyear / 365.25), doy_c=np.cos(2 * np.pi * a.dayofyear / 365.25),
                     sump_pre=P.G71F68Y_LevelPS.loc[a - pd.Timedelta("1h"):a].mean(),
                     f05_peak=win.G71F05R_LevelInlet.max(), f04_peak=win.G71F04R_Level2.max(), f06_peak=win.G71F06R_LevelInlet.max(),
                     f04_exc=np.clip(win.G71F04R_Level2 - 0.30, 0, None).sum() * 5 / 60,
                     sump_peak=win.G71F68Y_LevelPS.max(),
                     cso_min=(win.G71F68Y_LevelPS >= 4.5).sum() * 5.0,
                     v04=win.G71F04R_Level2.notna().mean(), v68=win.G71F68Y_LevelPS.notna().mean()))
W = pd.DataFrame(rows)
W = W[(W.gauge_ok > 0.9) & (W.v68 > 0.8)].reset_index(drop=True)
W["store"] = (W.sump_peak > 1.0).astype(int); W["cso"] = (W.sump_peak >= 4.5).astype(int)
W.to_csv(os.path.join(OUT, "E13_rain_events.csv"), index=False)
print("rain events (>=2 mm):", len(W), "| by year:", W.groupby(W.start.dt.year).size().to_dict())
print("CSO events:", int(W.cso.sum()), "| storage events:", int(W.store.sum()))
FX = ["dur_h", "depth", "d5425", "d5427", "i5", "i30", "i60", "ante24", "ante7d", "doy_s", "doy_c", "sump_pre"]
OUTC = {"f05_peak": "reg", "f04_peak": "reg", "f04_exc": "reg", "f06_peak": "reg", "sump_peak": "reg", "cso_min": "reg", "store": "clf", "cso": "clf"}

def mdl(kind, n=300):
    kw = dict(n_estimators=n, learning_rate=0.03, num_leaves=7, min_child_samples=15, subsample=0.8, subsample_freq=1, verbose=-1)
    return lgb.LGBMRegressor(**kw) if kind == "reg" else lgb.LGBMClassifier(**kw)

def cf(trm, evm, y, kind, rng=np.random.default_rng(0)):
    ok_tr = trm & W[y].notna(); ok_ev = evm & W[y].notna()
    m = mdl(kind).fit(W.loc[ok_tr, FX], W.loc[ok_tr, y])
    pe = m.predict(W.loc[ok_ev, FX]) if kind == "reg" else m.predict_proba(W.loc[ok_ev, FX])[:, 1]
    d = W.loc[ok_ev, y].values - pe
    bs = [rng.choice(d, len(d)).mean() for _ in range(500)]
    return dict(effect=d.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), obs=W.loc[ok_ev, y].mean(), cf=pe.mean(), n=int(ok_ev.sum()))

t = W.start
PRE0, SWAP, POST0, POST1 = pd.Timestamp("2015-06-01"), pd.Timestamp("2018-09-20"), pd.Timestamp("2018-10-05"), pd.Timestamp("2019-10-31")
res = []
for y, kind in OUTC.items():
    for lab, pe in [("post 2018-10..2019-10", (t >= POST0) & (t <= POST1)), ("post 2018-10..2021-08", t >= POST0)]:
        res.append(dict(outcome=y, analysis="real", window=lab, **cf((t >= PRE0) & (t < SWAP), pe, y, kind)))
    for fake in pd.date_range("2016-12-01", "2017-10-01", freq="2MS"):
        res.append(dict(outcome=y, analysis="placebo", window=str(fake.date()),
                        **cf((t >= PRE0) & (t < fake), (t >= fake) & (t < fake + pd.DateOffset(months=12)) & (t < SWAP), y, kind)))
R = pd.DataFrame(res); R.to_csv(os.path.join(OUT, "E13_rain_counterfactual.csv"), index=False)
pd.set_option("display.width", 220)
S = R[R.analysis == "real"][["outcome", "window", "effect", "lo", "hi", "obs", "cf", "n"]]
pl = R[R.analysis == "placebo"].groupby("outcome").effect.agg(pl_min="min", pl_max="max")
print(S.merge(pl, on="outcome").round(3).to_string(index=False))

# DML: pre (2015-06..2018-09) vs post windows
dml = []
for lab, post_end in (("2018-10..2019-10", POST1), ("2018-10..2021-08", pd.Timestamp("2021-12-31"))):
    sub = W[((t >= PRE0) & (t < SWAP)) | ((t >= POST0) & (t <= post_end))].reset_index(drop=True)
    D = (sub.start >= POST0).astype(float).values
    for y, kind in OUTC.items():
        ok = sub[y].notna().values
        Y = sub.loc[ok, y].astype(float).values; Xs = sub.loc[ok, FX]; Dd = D[ok]
        ry, rd = np.zeros(len(Y)), np.zeros(len(Y))
        for rep in range(5):
            for tri, tei in KFold(5, shuffle=True, random_state=rep).split(Xs):
                my = mdl("reg", 200).fit(Xs.iloc[tri], Y[tri]); md = mdl("clf", 200).fit(Xs.iloc[tri], Dd[tri])
                ry[tei] += (Y[tei] - my.predict(Xs.iloc[tei])) / 5
                rd[tei] += (Dd[tei] - md.predict_proba(Xs.iloc[tei])[:, 1]) / 5
        th = (rd @ ry) / (rd @ rd); eps = ry - th * rd
        se = np.sqrt(np.sum(rd ** 2 * eps ** 2)) / (rd @ rd)
        dml.append(dict(window=lab, outcome=y, theta=th, se=se, lo=th - 1.96 * se, hi=th + 1.96 * se, n=len(Y), n_post=int(Dd.sum()),
                        base_pre=Y[Dd == 0].mean()))
Dm = pd.DataFrame(dml); Dm.to_csv(os.path.join(OUT, "E13_rain_dml.csv"), index=False)
print(Dm.round(4).to_string(index=False))
