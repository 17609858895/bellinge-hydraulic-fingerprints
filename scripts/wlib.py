"""Shared helpers for the CSO warning experiments (E16, E17).
Rain features are log1p-transformed (heavy-tailed depths destabilised the logistic model in E14)."""
import os, numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")
_P = None; _RN = None

def panel():
    global _P, _RN
    if _P is None:
        _P = pd.read_parquet(os.path.join(ROOT, "processed", "panel_5min.parquet"))
        _RN = pd.read_parquet(os.path.join(ROOT, "processed", "rain_5min_utc.parquet")).reindex(_P.index)
    return _P, _RN

RF = ["rain_15", "rain_30", "rain_60", "rain_120", "rain_360", "rain_ante24", "rain_i10max2h", "rain_g_diff60"]

def rain_feats(times):
    P, RN = panel()
    r = RN["mean"].fillna(0); cum = r.cumsum(); t = pd.DatetimeIndex(times)
    f = {}
    for m in (15, 30, 60, 120, 360):
        f[f"rain_{m}"] = np.log1p(np.clip(cum.reindex(t).values - cum.reindex(t - pd.Timedelta(minutes=m)).values, 0, None))
    f["rain_ante24"] = np.log1p(np.clip(cum.reindex(t - pd.Timedelta("6h")).values - cum.reindex(t - pd.Timedelta("30h")).values, 0, None))
    f["rain_i10max2h"] = np.log1p(r.rolling(2).sum().rolling(24, min_periods=1).max().reindex(t).values)
    d = (RN["5425"].fillna(0).rolling(12).sum() - RN["5427"].fillna(0).rolling(12).sum()).reindex(t).values
    f["rain_g_diff60"] = np.sign(d) * np.log1p(np.abs(d))
    return pd.DataFrame(f, index=range(len(t)))

def split_of(start):
    start = pd.DatetimeIndex(start)
    s = np.where(start >= pd.Timestamp("2018-10-05"), "test", np.where(start >= pd.Timestamp("2017-01-01"), "val", "train"))
    return np.where((start >= pd.Timestamp("2018-09-20")) & (start < pd.Timestamp("2018-10-05")), "drop", s)

SC_A = ["elapsed", "since_gate", "h", "dh10", "dh30", "dh60", "hmax", "ttf_phys", "f05", "f05_d30", "f05_max60", "f05_cum",
        "f06", "f06_d30", "f06_cum", "gate", "b05", "b05_d30", "q68", "doy_s", "doy_c", "hour_s", "hour_c"]
F04_A = ["f04", "f04_d30", "f04_cum"]
SC_B = ["elapsed", "h", "dh30", "dh60", "ttf_phys", "f05", "f05_d30", "f06", "f06_d30", "gate", "b05", "q68", "doy_s", "doy_c"]
F04_B = ["f04", "f04_d30"]

def design_A():
    L = pd.read_parquet(os.path.join(ROOT, "processed", "landmarks.parquet")).reset_index(drop=True)
    L = pd.concat([L, rain_feats(L.t)], axis=1)
    L["split"] = split_of(L.groupby("eid").t.transform("min")); return L[L.split != "drop"].reset_index(drop=True)

def design_B(thr=4.5):
    """Rain-onset clock; CSO = sump >= thr. Rebuilt for thresholds other than 4.5."""
    f = os.path.join(ROOT, "processed", f"landmarks_rain_onset_{thr}.parquet")
    if thr == 4.5 and os.path.exists(os.path.join(ROOT, "processed", "landmarks_rain_onset.parquet")):
        L = pd.read_parquet(os.path.join(ROOT, "processed", "landmarks_rain_onset.parquet"))
        L[RF] = rain_feats(L.t).values
        return L.reset_index(drop=True)
    if os.path.exists(f):
        return pd.read_parquet(f)
    P, RN = panel()
    E13 = pd.read_csv(os.path.join(OUT, "E13_rain_events.csv"), parse_dates=["start"])
    S = P.G71F68Y_LevelPS.ffill(limit=6)
    cols = {k: v.ffill(limit=6) for k, v in dict(f05=P.G71F05R_LevelInlet, f06=P.G71F06R_LevelInlet, f04=P.G71F04R_Level2,
                                                 gate=P.G71F05R_position, b05=P.G71F05R_LevelBasin, q68=P.G71F68Y_Flow).items()}
    rows = []
    for eid, ev in E13.iterrows():
        a = ev.start; s = S.loc[a - pd.Timedelta("2h"): a + pd.Timedelta("24h")]
        hit = s.loc[a:][s.loc[a:] >= thr]; tc = hit.index[0] if len(hit) else None
        idx = s.index; i0 = idx.searchsorted(a); sv = s.values; cv = {k: v.reindex(idx).values for k, v in cols.items()}
        for i in range(i0, len(idx), 2):
            t = idx[i]
            if tc is not None and t >= tc: break
            tr_ = lambda x, k: x[i] - x[i - k] if i - k >= 0 else np.nan
            dh30 = tr_(sv, 6); h = sv[i]
            ttf = (thr - h) / max(dh30 / 30.0, 1e-4) if np.isfinite(dh30) and np.isfinite(h) else 2880.0
            rows.append(dict(eid=eid, t=t, elapsed=(t - a).total_seconds() / 60, h=h, dh30=dh30, dh60=tr_(sv, 12), ttf_phys=min(max(ttf, 1.0), 2880.0),
                             f05=cv["f05"][i], f05_d30=tr_(cv["f05"], 6), f06=cv["f06"][i], f06_d30=tr_(cv["f06"], 6), f04=cv["f04"][i], f04_d30=tr_(cv["f04"], 6),
                             gate=cv["gate"][i], b05=cv["b05"][i], q68=cv["q68"][i], doy_s=ev.doy_s, doy_c=ev.doy_c,
                             T=(tc - t).total_seconds() / 60 if tc is not None else 2880.0, delta=int(tc is not None), event_cso=int(tc is not None), start=a))
    L = pd.DataFrame(rows).reset_index(drop=True)
    L = pd.concat([L, rain_feats(L.t)], axis=1); L["split"] = split_of(L.start); L = L[L.split != "drop"].reset_index(drop=True)
    L.to_parquet(f); return L

def predict(L, feats, H, model, seed=0, return_mu=False):
    """Returns P(T<=H) for every landmark (models fit on train; AFT early-stopped on val)."""
    y = ((L.delta == 1) & (L["T"] <= H)).astype(int).values
    tr = (L.split == "train").values; va = (L.split == "val").values
    if model == "Logistic":
        m = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(C=0.5, max_iter=5000)).fit(L.loc[tr, feats], y[tr])
        return m.predict_proba(L[feats])[:, 1]
    if model == "LightGBM":
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=80, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, reg_lambda=5, random_state=seed, verbose=-1).fit(L.loc[tr, feats], y[tr])
        return m.predict_proba(L[feats])[:, 1]
    lo = np.where(L.delta == 1, L["T"], 2880.0); up = np.where(L.delta == 1, L["T"], np.inf)
    off = np.log(L.ttf_phys.values) if model == "Phys-AFT" else np.zeros(len(L))
    dtr = xgb.DMatrix(L.loc[tr, feats], label_lower_bound=lo[tr], label_upper_bound=up[tr]); dtr.set_base_margin(off[tr])
    dva = xgb.DMatrix(L.loc[va, feats], label_lower_bound=lo[va], label_upper_bound=up[va]); dva.set_base_margin(off[va])
    prm = dict(objective="survival:aft", eval_metric="aft-nloglik", aft_loss_distribution="normal", aft_loss_distribution_scale=1.2,
               learning_rate=0.03, max_depth=4, min_child_weight=20, subsample=0.8, colsample_bytree=0.8, reg_lambda=5, tree_method="hist", seed=seed)
    b = xgb.train(prm, dtr, 600, evals=[(dva, "v")], early_stopping_rounds=60, verbose_eval=False)
    da = xgb.DMatrix(L[feats]); da.set_base_margin(off)
    mu = b.predict(da, output_margin=True, iteration_range=(0, b.best_iteration + 1))
    if return_mu:
        return mu
    return norm.cdf((np.log(H) - mu) / 1.2)

def event_eval(L, p, far=0.05):
    ev = pd.DataFrame({"eid": L.eid.values, "split": L.split.values, "cso": L.event_cso.values, "p": p, "T": L["T"].values})
    E = ev.groupby("eid").agg(split=("split", "first"), cso=("cso", "max"), pmax=("p", "max"))
    tau = np.quantile(E[(E.split == "val") & (E.cso == 0)].pmax, 1 - far)
    out = {}
    for s in ("val", "test"):
        e = E[E.split == s]; leads = []
        for eid in e.index[e.cso == 1]:
            g = ev[(ev.eid == eid) & (ev.p > tau)]
            leads.append(g["T"].iloc[0] if len(g) else 0.0)
        out[s] = dict(hits=int(((e.cso == 1) & (e.pmax > tau)).sum()), n_cso=int(e.cso.sum()), FA=int(((e.cso == 0) & (e.pmax > tau)).sum()),
                      n0=int((e.cso == 0).sum()), med_lead=float(np.median(leads)) if leads else np.nan)
    return out
