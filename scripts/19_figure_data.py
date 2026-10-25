"""E19: data for figure panels not stored by E02-E18 (results/figdata/)."""
import os, sys, importlib.util, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import KFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wlib as W
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FD = os.path.join(ROOT, "results", "figdata"); os.makedirs(FD, exist_ok=True)
P, RN = W.panel()
EV = pd.read_csv(os.path.join(W.OUT, "E13_rain_events.csv"), parse_dates=["start"])
OFF = 0.0541

# ---- Fig 5a: per-event observed vs counterfactual (pre: 5-fold cross-fitted; post: full pre model)
f04c = P.G71F04R_Level2 - np.where(P.index >= "2019-11-01", OFF, 0)
EV["f04_peak_c"] = [f04c.loc[s: s + pd.Timedelta(hours=d) + pd.Timedelta("12h")].max() for s, d in zip(EV.start, EV.dur_h)]
FX = ["dur_h", "depth", "d5425", "d5427", "i5", "i30", "i60", "ante24", "ante7d", "doy_s", "doy_c", "sump_pre"]
t = EV.start; pre = (t >= "2015-06-01") & (t < "2018-09-20"); post = t >= "2018-10-05"
rows = []
for y in ("f04_peak_c", "f06_peak", "f05_peak"):
    ok = EV[y].notna(); prm = dict(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=15, subsample=0.8, subsample_freq=1, verbose=-1)
    cf = pd.Series(np.nan, index=EV.index)
    idx = EV.index[pre & ok]
    for a, b in KFold(5, shuffle=True, random_state=0).split(idx):
        m = lgb.LGBMRegressor(**prm).fit(EV.loc[idx[a], FX], EV.loc[idx[a], y]); cf[idx[b]] = m.predict(EV.loc[idx[b], FX])
    m = lgb.LGBMRegressor(**prm).fit(EV.loc[idx, FX], EV.loc[idx, y]); ip = EV.index[post & ok]; cf[ip] = m.predict(EV.loc[ip, FX])
    for i in EV.index[(pre | post) & ok]:
        rows.append(dict(outcome=y, start=EV.start[i], period="pre" if pre[i] else "post", obs=EV.loc[i, y], cf=cf[i], depth=EV.depth[i]))
pd.DataFrame(rows).to_csv(os.path.join(FD, "fig5a_event_cf.csv"), index=False)
print("fig5a done", flush=True)

# ---- Fig 6a/f: design-B Phys-AFT (+rain) time courses and ACI bounds
L = W.design_B(4.5); feats = W.SC_B + W.F04_B + W.RF
p = np.mean([W.predict(L, feats, 120, "Phys-AFT", seed=s) for s in (0, 1, 2)], axis=0)
mu = np.mean([W.predict(L, feats, 120, "Phys-AFT", seed=s, return_mu=True) for s in (0, 1, 2)], axis=0)
E = pd.DataFrame({"eid": L.eid, "split": L.split, "cso": L.event_cso, "p": p}).groupby("eid").agg(split=("split", "first"), cso=("cso", "max"), pmax=("p", "max"))
tau = np.quantile(E[(E.split == "val") & (E.cso == 0)].pmax, 0.95)
cso = (L.delta == 1).values; logT = np.log(np.maximum(L["T"].values, 1.0)); cal = (L.split == "val").values & cso; te = (L.split == "test").values & cso
out = L[["eid", "t", "start", "split", "T", "delta", "event_cso", "h", "elapsed"]].copy(); out["p120"] = p; out["tau"] = tau
for name, m_ in (("phys", np.log(L.ttf_phys.values)), ("aft", mu)):
    a_t, s_hist = 0.1, list(m_[cal] - logT[cal]); Lb = np.full(len(L), np.nan)
    for e in L[te].groupby("eid").t.min().sort_values().index:
        mm = te & (L.eid.values == e); qq = np.quantile(s_hist, min(max(1 - a_t, 0), 1))
        Lb[mm] = np.exp(m_[mm] - qq); c = (L["T"].values[mm] >= Lb[mm]); a_t += 0.02 * (0.1 - (1 - c.mean())); s_hist += list(m_[mm] - logT[mm])
    out[f"L_{name}_aci10"] = Lb
out.to_parquet(os.path.join(FD, "fig6_designB_timecourse.parquet"))
print("fig6 done; tau =", round(tau, 4), flush=True)

# ---- Fig 3e: block-permutation null of the max split gain (G71F04R reference regime)
spec = importlib.util.spec_from_file_location("dw", os.path.join(ROOT, "scripts", "09b_drywet_plane.py")); dw = importlib.util.module_from_spec(spec); spec.loader.exec_module(dw)
D = pd.read_parquet(os.path.join(ROOT, "processed", "twin2_G71F04R_Level2_frozen.parquet"))
M0 = dw.month_stats(D.loc[dw.REF0:dw.REF1]); dw.set_floor(M0, ["A", "W"])
X0, S0 = M0[["A", "W"]].values, dw.floor_se(M0, ["A", "W"]); rng = np.random.default_rng(7); null = []
for _ in range(1000):
    blocks = [np.arange(i, min(i + 3, len(X0))) for i in range(0, len(X0), 3)]
    o = np.concatenate([blocks[j] for j in rng.permutation(len(blocks))]); null.append(dw.max_gain(X0[o], S0[o]))
M = pd.read_csv(os.path.join(W.OUT, "E11_monthly_AW_G71F04R_Level2.csv"), index_col=0, parse_dates=True)
dw.set_floor(M.loc[dw.REF0:dw.REF1], ["A", "W"])
real = dw.max_gain(M[["A", "W"]].values, dw.floor_se(M, ["A", "W"]))
pd.DataFrame({"null_max_gain": null}).assign(real_full_record=real).to_csv(os.path.join(FD, "fig3e_null.csv"), index=False)
print("fig3e done", flush=True)

# ---- Fig 2d: monthly P95 level vs monthly rain
mon = pd.DataFrame({"p95_f04": P.G71F04R_Level2.resample("MS").quantile(0.95), "rain_mm": RN["mean"].resample("MS").sum(min_count=1000),
                    "valid": P.G71F04R_Level2.notna().resample("MS").mean()})
mon.to_csv(os.path.join(FD, "fig2d_monthly.csv"))

# ---- Fig 2b/c: matched storms (similar depth and 30-min intensity), pre vs post swap, valid G71F04R
cand = EV[(EV.depth.between(8, 25)) & EV.f04_peak.notna()]
pre_c = cand[(cand.start >= "2016-09-01") & (cand.start < "2018-09-01")]; post_c = cand[(cand.start >= "2018-10-05") & (cand.start < "2019-07-01")]
best = None
for i, a in pre_c.iterrows():
    for j, b in post_c.iterrows():
        d = abs(np.log(a.depth / b.depth)) + abs(np.log((a.i30 + .1) / (b.i30 + .1))) + abs(a.f05_peak - b.f05_peak) * 5
        if best is None or d < best[0]: best = (d, i, j)
Tw = pd.read_parquet(os.path.join(ROOT, "processed", "twin2_G71F04R_Level2.parquet"))
ts = []
for lab, k in (("pre", best[1]), ("post", best[2])):
    s = EV.start[k]; w = slice(s - pd.Timedelta("2h"), s + pd.Timedelta(hours=EV.dur_h[k]) + pd.Timedelta("8h"))
    seg = pd.DataFrame({"rain_mm": RN["mean"].loc[w], "obs": P.G71F04R_Level2.loc[w], "twin": Tw.yhat.reindex(P.loc[w].index), "f05": P.G71F05R_LevelInlet.loc[w]})
    seg["period"] = lab; seg["hours"] = (seg.index - s).total_seconds() / 3600; seg["depth"] = EV.depth[k]; seg["i30"] = EV.i30[k]; seg["start"] = s
    ts.append(seg.reset_index())
pd.concat(ts).to_csv(os.path.join(FD, "fig2bc_storms.csv"), index=False)
print("fig2 done:", [str(EV.start[best[1]]), str(EV.start[best[2]])], flush=True)

# ---- Fig 1b: monthly availability of rain gauges
RA = RN[["5425", "5427"]].notna().resample("MS").mean(); RA.to_csv(os.path.join(FD, "fig1b_rain_availability.csv"))
