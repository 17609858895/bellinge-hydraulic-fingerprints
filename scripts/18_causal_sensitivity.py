"""E18: sensitivity of the rain-adjusted swap effects (E13/E15).
S7  rain-event definition: min depth {1,2,5} mm x dry gap {6,12} h
S10 CSO threshold {4.4, 4.5, 4.6} m
S11 pre-period truncated at 2017-09-30 (before the small G71F05R-inlet break found in E11)
G71F04R outcomes use the Nov-2019 zero-point correction (0.0541 m, E15)."""
import os, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import KFold
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")
P = pd.read_parquet(os.path.join(ROOT, "processed", "panel_5min.parquet")).loc["2011-01-01":]
Rn = pd.read_parquet(os.path.join(ROOT, "processed", "rain_5min_utc.parquet")).reindex(P.index)
OFF = 0.0541
f04c = P.G71F04R_Level2 - np.where(P.index >= "2019-11-01", OFF, 0)
r = Rn["mean"].fillna(0); t_wet = r.index[(r > 0).values]
gaps_min = np.diff(t_wet.values).astype("timedelta64[m]").astype(int)

def events(min_depth, gap_h):
    br = gaps_min >= gap_h * 60
    starts = [t_wet[0]] + list(t_wet[1:][br]); ends = list(t_wet[:-1][br]) + [t_wet[-1]]
    rows = []
    for a, b in zip(starts, ends):
        seg = Rn.loc[a:b]; tot = seg["mean"].sum()
        if tot < min_depth: continue
        win = P.loc[a: b + pd.Timedelta("12h")]; w04 = f04c.loc[a: b + pd.Timedelta("12h")]
        ante = Rn.loc[a - pd.Timedelta("7D"): a - pd.Timedelta("5min")]; rr = seg["mean"].fillna(0)
        rows.append(dict(start=a, dur_h=(b - a).total_seconds() / 3600 + 1 / 12, depth=tot, d5425=seg["5425"].sum(), d5427=seg["5427"].sum(),
                         i5=rr.max() * 12, i30=rr.rolling(6).sum().max() * 2, i60=rr.rolling(12).sum().max(),
                         ante24=ante["mean"].iloc[-288:].sum(), ante7d=ante["mean"].sum(),
                         gauge_ok=float(seg[["5425", "5427"]].notna().all(axis=1).mean()),
                         doy_s=np.sin(2 * np.pi * a.dayofyear / 365.25), doy_c=np.cos(2 * np.pi * a.dayofyear / 365.25),
                         sump_pre=P.G71F68Y_LevelPS.loc[a - pd.Timedelta("1h"):a].mean(),
                         f05_peak=win.G71F05R_LevelInlet.max(), f04_peak_c=w04.max(), f06_peak=win.G71F06R_LevelInlet.max(),
                         sump_peak=win.G71F68Y_LevelPS.max(), v68=win.G71F68Y_LevelPS.notna().mean()))
    Wd = pd.DataFrame(rows); Wd = Wd[(Wd.gauge_ok > 0.9) & (Wd.v68 > 0.8)].reset_index(drop=True)
    Wd["store"] = (Wd.sump_peak > 1.0).astype(int)
    for thr in (4.4, 4.5, 4.6): Wd[f"cso{thr}"] = (Wd.sump_peak >= thr).astype(int)
    return Wd

FX = ["dur_h", "depth", "d5425", "d5427", "i5", "i30", "i60", "ante24", "ante7d", "doy_s", "doy_c", "sump_pre"]
def m_(kind, n=300):
    kw = dict(n_estimators=n, learning_rate=0.03, num_leaves=7, min_child_samples=15, subsample=0.8, subsample_freq=1, verbose=-1)
    return lgb.LGBMRegressor(**kw) if kind == "reg" else lgb.LGBMClassifier(**kw)

def estimate(Wd, y, kind, pre_end):
    t = Wd.start; PRE0, POST0 = pd.Timestamp("2015-06-01"), pd.Timestamp("2018-10-05")
    trm = (t >= PRE0) & (t < pre_end) & Wd[y].notna(); pe = (t >= POST0) & Wd[y].notna()
    m = m_(kind).fit(Wd.loc[trm, FX], Wd.loc[trm, y])
    pr = m.predict(Wd.loc[pe, FX]) if kind == "reg" else m.predict_proba(Wd.loc[pe, FX])[:, 1]
    d = Wd.loc[pe, y].values - pr; rng = np.random.default_rng(0); bs = [rng.choice(d, len(d)).mean() for _ in range(300)]
    sub = Wd[trm | pe].reset_index(drop=True); D = (sub.start >= POST0).astype(float).values; Y = sub[y].astype(float).values; Xs = sub[FX]
    ry, rd = np.zeros(len(Y)), np.zeros(len(Y))
    for rep in range(3):
        for a, b in KFold(5, shuffle=True, random_state=rep).split(Xs):
            ry[b] += (Y[b] - m_("reg", 200).fit(Xs.iloc[a], Y[a]).predict(Xs.iloc[b])) / 3
            rd[b] += (D[b] - m_("clf", 200).fit(Xs.iloc[a], D[a]).predict_proba(Xs.iloc[b])[:, 1]) / 3
    th = (rd @ ry) / (rd @ rd); se = np.sqrt(np.sum(rd ** 2 * (ry - th * rd) ** 2)) / (rd @ rd)
    return dict(cf=d.mean(), cf_lo=np.percentile(bs, 2.5), cf_hi=np.percentile(bs, 97.5), dml=th, dml_lo=th - 1.96 * se, dml_hi=th + 1.96 * se,
                n_pre=int(trm.sum()), n_post=int(pe.sum()), n_cso_post=int(Wd.loc[pe, "cso4.5"].sum()) if "cso4.5" in Wd else np.nan)

OUTC = {"f05_peak": "reg", "f04_peak_c": "reg", "f06_peak": "reg", "sump_peak": "reg", "store": "clf", "cso4.4": "clf", "cso4.5": "clf", "cso4.6": "clf"}
rows = []
base_end = pd.Timestamp("2018-09-20")
for (md, gh) in [(2, 6), (1, 6), (5, 6), (2, 12), (5, 12)]:
    Wd = events(md, gh)
    ends = [("pre to 2018-09", base_end)] + ([("pre to 2017-09 (S11)", pd.Timestamp("2017-09-30"))] if (md, gh) == (2, 6) else [])
    for lab, pend in ends:
        for y, kind in OUTC.items():
            if y.startswith("cso") and y != "cso4.5" and not ((md, gh) == (2, 6) and pend == base_end):
                continue
            rows.append(dict(min_depth=md, gap_h=gh, pre=lab, n_events=len(Wd), outcome=y, **estimate(Wd, y, kind, pend)))
    print(md, gh, len(Wd), flush=True)
R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, "E18_causal_sensitivity.csv"), index=False)
pd.set_option("display.width", 250); print(R.round(3).to_string(index=False))
