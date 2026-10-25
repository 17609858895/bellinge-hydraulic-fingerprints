"""Hydraulic fingerprint v2.
1) Cross-sensor twin per node (inputs: other nodes only + lags), reference regime 2015-06..2018-08,
   cross-fitted by quarter with a 7-day embargo inside the reference; full-reference model outside.
2) Monthly residual *profile*: median residual in fixed bins of the twin prediction (equal weight per bin).
3) Shape-basis projection of the profile: r(yhat) = a + g*yhat + c*max(yhat-kappa,0)
   a = offset (zero-point), g = gain (scaling), c = above-capacity conveyance; day-block bootstrap CI.
"""
import os, numpy as np, pandas as pd, lightgbm as lgb
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = pd.read_parquet(os.path.join(ROOT, "processed", "panel_5min.parquet")).loc["2010-08-01":]
OUT = os.path.join(ROOT, "results")
REF0, REF1 = pd.Timestamp("2015-06-01"), pd.Timestamp("2018-08-31")

def lagfeats(df, cols, lags=(1, 3, 6, 12), roll=(6, 12, 36)):
    F = {}
    for c in cols:
        s = df[c]; F[c] = s
        for l in lags: F[f"{c}_l{l}"] = s.shift(l)
        for r in roll:
            F[f"{c}_max{r}"] = s.rolling(r, min_periods=1).max()
            F[f"{c}_mean{r}"] = s.rolling(r, min_periods=1).mean()
    F["doy_s"] = np.sin(2 * np.pi * df.index.dayofyear / 365.25); F["doy_c"] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    F["hod_s"] = np.sin(2 * np.pi * df.index.hour / 24); F["hod_c"] = np.cos(2 * np.pi * df.index.hour / 24)
    return pd.DataFrame(F, index=df.index)

NODES = {
    "G71F04R_Level2": (["G71F05R_LevelInlet", "G71F06R_LevelInlet", "G71F68Y_LevelPS", "G71F05R_position", "G71F05R_LevelBasin"], 0.25,
                       [0, .06, .08, .10, .12, .15, .2, .25, .3, .4, .5, .7, 1.0, 1.6]),
    "G71F06R_LevelInlet": (["G71F05R_LevelInlet", "G71F04R_Level2", "G71F68Y_LevelPS", "G71F05R_position", "G71F05R_LevelBasin"], 0.30,
                           [0, .06, .09, .12, .15, .2, .25, .3, .35, .4, .45, .6]),
    "G71F05R_LevelInlet": (["G71F04R_Level2", "G71F06R_LevelInlet", "G71F68Y_LevelPS", "G71F05R_position", "G71F05R_LevelBasin"], 0.33,
                           [0, .22, .24, .26, .28, .30, .33, .36, .40, .45, .5, .6, 1.0]),
    "G71F68Y_LevelPS": (["G71F05R_LevelInlet", "G71F04R_Level2", "G71F06R_LevelInlet", "G71F05R_position", "G71F05R_LevelBasin", "G71F68Y_Flow"], 1.0,
                        [0, .2, .4, .6, .8, 1.0, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5.5]),
}

def twin(tgt, inp):
    X = lagfeats(P, inp); y = P[tgt]
    ok = y.notna() & X[inp].notna().all(axis=1)
    yhat = pd.Series(np.nan, index=P.index)
    ref = ok & (P.index >= REF0) & (P.index <= REF1)
    q = pd.Series(P.index.to_period("Q"), index=P.index)
    prm = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1,
               colsample_bytree=0.8, reg_lambda=5, verbose=-1)
    for qq in q[ref].unique():
        te = ref & (q == qq)
        lo, hi = P.index[te].min() - pd.Timedelta("7D"), P.index[te].max() + pd.Timedelta("7D")
        trm = ref & ~((P.index >= lo) & (P.index <= hi))
        m = lgb.LGBMRegressor(**prm).fit(X[trm], y[trm])
        yhat[te] = m.predict(X[te])
    m = lgb.LGBMRegressor(**prm).fit(X[ref], y[ref])
    out = ok & ~ref
    yhat[out] = m.predict(X[out])
    r = (y - yhat)[ref]; yy = y[ref]
    nse = 1 - (r ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
    imp = pd.Series(m.booster_.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
    return pd.DataFrame({"y": y, "yhat": yhat}).dropna(), nse, imp

def profile_fit(g, edges, kappa):
    b = pd.cut(g.yhat, edges)
    prof = g.assign(r=g.y - g.yhat, b=b).groupby("b", observed=True).agg(r=("r", "median"), x=("yhat", "median"), n=("r", "size"))
    prof = prof[prof.n >= 24]
    if len(prof) < 4:
        return None, prof
    Z = np.c_[np.ones(len(prof)), prof.x.values, np.maximum(prof.x.values - kappa, 0)]
    co, *_ = np.linalg.lstsq(Z, prof.r.values, rcond=None)
    return co, prof

rng = np.random.default_rng(0)
summary, profiles = [], []
for tgt, (inp, kappa, edges) in NODES.items():
    D, nse, imp = twin(tgt, inp)
    D.astype("float32").to_parquet(os.path.join(ROOT, "processed", f"twin2_{tgt}.parquet"))
    imp.head(15).to_csv(os.path.join(OUT, f"E08_importance_{tgt}.csv"))
    wet = D.loc[REF0:REF1]; wet = wet[wet.y > kappa]
    nse_w = 1 - ((wet.y - wet.yhat) ** 2).sum() / ((wet.y - wet.y.mean()) ** 2).sum()
    print(f"{tgt}: cross-fitted reference NSE={nse:.3f}, wet NSE={nse_w:.3f}", flush=True)
    for mon, g in D.groupby(pd.Grouper(freq="MS")):
        if len(g) < 3000:
            continue
        co, prof = profile_fit(g, edges, kappa)
        if co is None:
            continue
        for _, pr in prof.reset_index().iterrows():
            profiles.append(dict(node=tgt, month=mon, bin_x=pr.x, r=pr.r, n=pr.n))
        codes, ud = pd.factorize(g.index.normalize()); dpos = [np.where(codes == k)[0] for k in range(len(ud))]
        B = []
        for _ in range(100):
            idx = np.concatenate([dpos[k] for k in rng.integers(0, len(ud), len(ud))])
            cb, _ = profile_fit(g.iloc[idx], edges, kappa)
            if cb is not None: B.append(cb)
        B = np.array(B)
        res = g.y - g.yhat
        summary.append(dict(node=tgt, month=mon, n=len(g), n_wet=int((g.yhat > kappa).sum()),
                            a=co[0], g=co[1], c=co[2], a_se=B[:, 0].std(), g_se=B[:, 1].std(), c_se=B[:, 2].std(),
                            res_dry=float(res[g.yhat < np.quantile(D.yhat, 0.5)].median()),
                            res_wet=float(res[g.yhat > kappa].median()) if (g.yhat > kappa).sum() > 24 else np.nan))
    print(tgt, "profiles done", flush=True)
pd.DataFrame(summary).to_csv(os.path.join(OUT, "E08_monthly_shape_coefficients.csv"), index=False)
pd.DataFrame(profiles).to_csv(os.path.join(OUT, "E08_monthly_profiles.csv"), index=False)
