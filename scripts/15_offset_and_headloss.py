"""(a) G71F04R effects with the detected Nov-2019 dry-weather offset removed (A shift from E11/E09b).
(b) Head loss across the throttle pipe, pre (2016-09..2018-08) vs post (2018-10..2019-06, before the zero-point change),
binned by upstream forcing (G71F05R inlet level), against two textbook expectations for the same discharge:
orifice/inlet control  dh_post/dh_pre = (A_pre/A_post)^2 = (0.0628/0.0603)^2 = 1.085
full-pipe friction (Manning, same length/roughness)  dh_post/dh_pre = (K_pre/K_post)^2 = 0.70."""
import os, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import KFold
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")
# (a)
W = pd.read_csv(os.path.join(OUT, "E13_rain_events.csv"), parse_dates=["start"])
M = pd.read_csv(os.path.join(OUT, "E11_monthly_AW_G71F04R_Level2.csv"), index_col=0, parse_dates=True)
off = M.loc["2019-11-01":, "A"].median() - M.loc["2018-10-01":"2019-06-30", "A"].median()
print("offset removed after 2019-11-01 (m):", round(off, 4))
W["f04_peak_c"] = W.f04_peak - np.where(W.start >= "2019-11-01", off, 0)
W["f04_exc_c"] = W.f04_exc  # recompute approx: shift lowers exceedance; done exactly below from 5-min data
P = pd.read_parquet(os.path.join(ROOT, "processed", "panel_5min.parquet"))
f04c = P.G71F04R_Level2 - np.where(P.index >= "2019-11-01", off, 0)
ex = []
for _, e in W.iterrows():
    seg = f04c.loc[e.start: e.start + pd.Timedelta(hours=e.dur_h) + pd.Timedelta("12h")]
    ex.append(np.clip(seg - 0.30, 0, None).sum() * 5 / 60)
W["f04_exc_c"] = ex
FX = ["dur_h", "depth", "d5425", "d5427", "i5", "i30", "i60", "ante24", "ante7d", "doy_s", "doy_c", "sump_pre"]
t = W.start; PRE0, SWAP, POST0 = pd.Timestamp("2015-06-01"), pd.Timestamp("2018-09-20"), pd.Timestamp("2018-10-05")
rows = []
for y in ("f04_peak_c", "f04_exc_c"):
    trm = (t >= PRE0) & (t < SWAP) & W[y].notna(); pe = (t >= POST0) & W[y].notna()
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=15, subsample=0.8, subsample_freq=1, verbose=-1).fit(W.loc[trm, FX], W.loc[trm, y])
    d = W.loc[pe, y].values - m.predict(W.loc[pe, FX]); rng = np.random.default_rng(0)
    bs = [rng.choice(d, len(d)).mean() for _ in range(500)]
    sub = W[(trm | pe)].reset_index(drop=True); D = (sub.start >= POST0).astype(float).values; Y = sub[y].values; Xs = sub[FX]
    ry, rd = np.zeros(len(Y)), np.zeros(len(Y))
    for rep in range(5):
        for a, b in KFold(5, shuffle=True, random_state=rep).split(Xs):
            ry[b] += (Y[b] - lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=15, verbose=-1).fit(Xs.iloc[a], Y[a]).predict(Xs.iloc[b])) / 5
            rd[b] += (D[b] - lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=15, verbose=-1).fit(Xs.iloc[a], D[a]).predict_proba(Xs.iloc[b])[:, 1]) / 5
    th = (rd @ ry) / (rd @ rd); se = np.sqrt(np.sum(rd ** 2 * (ry - th * rd) ** 2)) / (rd @ rd)
    rows.append(dict(outcome=y, cf_effect=d.mean(), cf_lo=np.percentile(bs, 2.5), cf_hi=np.percentile(bs, 97.5), dml=th, dml_lo=th - 1.96 * se, dml_hi=th + 1.96 * se, n_post=int(pe.sum())))
A = pd.DataFrame(rows); print(A.round(3).to_string(index=False)); A.to_csv(os.path.join(OUT, "E15_f04_offset_corrected.csv"), index=False)

# (b) head loss
lv04 = 17.05 + P.G71F04R_Level2; lv06 = 16.72 + P.G71F06R_LevelInlet
dh = (lv04 - lv06); f05 = P.G71F05R_LevelInlet
bins = [0.30, 0.33, 0.36, 0.40, 0.45, 0.50, 0.60, 1.0]
out = []
for lab, (a, b) in {"pre 2016-09..2018-08": ("2016-09-01", "2018-08-31"), "post 2018-10..2019-06": ("2018-10-05", "2019-06-30")}.items():
    x = pd.DataFrame({"dh": dh, "f05": f05, "h06": P.G71F06R_LevelInlet}).loc[a:b].dropna()
    g = x.groupby(pd.cut(x.f05, bins), observed=True).agg(dh_med=("dh", "median"), dh_q25=("dh", lambda v: v.quantile(.25)), dh_q75=("dh", lambda v: v.quantile(.75)),
                                                        h06=("h06", "median"), n=("dh", "size"))
    g["period"] = lab; out.append(g.reset_index())
H = pd.concat(out); H["f05_bin"] = H.f05.astype(str)
piv = H.pivot_table(index="f05_bin", columns="period", values=["dh_med", "n"])
piv["ratio_post_pre"] = piv[("dh_med", "post 2018-10..2019-06")] / piv[("dh_med", "pre 2016-09..2018-08")]
print(piv.round(3).to_string()); print("textbook ratios: orifice 1.085, Manning full-pipe 0.703")
H.drop(columns="f05").to_csv(os.path.join(OUT, "E15_headloss_bins.csv"), index=False)
