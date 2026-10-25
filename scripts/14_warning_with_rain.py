"""CSO warning with observed (past-only) gauge rainfall. No rainfall forecast is used.
Design A: storage-onset clock (existing landmarks) +/- rain features +/- G71F04R features.
Design B: rain-onset clock: landmarks every 10 min from the start of each >=2 mm rain event (E13) until CSO or 24 h.
Split by event start: train <=2016, val 2017-01..2018-09, test >=2018-10-05. Event-level threshold at 5% val FAR."""
import os, numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")
P = pd.read_parquet(os.path.join(ROOT, "processed", "panel_5min.parquet"))
RN = pd.read_parquet(os.path.join(ROOT, "processed", "rain_5min_utc.parquet")).reindex(P.index)   # same clock (E12)
r = RN["mean"].fillna(0)
cum = r.cumsum()
def rain_feats(times):
    t = pd.DatetimeIndex(times)
    f = {}
    for m in (15, 30, 60, 120, 360):
        f[f"rain_{m}"] = (cum.reindex(t).values - cum.reindex(t - pd.Timedelta(minutes=m)).values)
    f["rain_ante24"] = cum.reindex(t - pd.Timedelta("6h")).values - cum.reindex(t - pd.Timedelta("30h")).values
    mx = r.rolling(2).sum().rolling(24, min_periods=1).max()   # max 10-min depth in last 2 h
    f["rain_i10max2h"] = mx.reindex(t).values
    f["rain_g_diff60"] = (RN["5425"].fillna(0).rolling(12).sum() - RN["5427"].fillna(0).rolling(12).sum()).reindex(t).values
    return pd.DataFrame(f, index=range(len(t)))

SC = ["elapsed", "since_gate", "h", "dh10", "dh30", "dh60", "hmax", "ttf_phys", "f05", "f05_d30", "f05_max60", "f05_cum",
      "f06", "f06_d30", "f06_cum", "gate", "b05", "b05_d30", "q68", "doy_s", "doy_c", "hour_s", "hour_c"]
F04 = ["f04", "f04_d30", "f04_cum"]
RF = ["rain_15", "rain_30", "rain_60", "rain_120", "rain_360", "rain_ante24", "rain_i10max2h", "rain_g_diff60"]

def split_of(start):
    s = np.where(start >= pd.Timestamp("2018-10-05"), "test", np.where(start >= pd.Timestamp("2017-01-01"), "val", "train"))
    s = np.where((start >= pd.Timestamp("2018-09-20")) & (start < pd.Timestamp("2018-10-05")), "drop", s)
    return s

def fit_eval(L, feats, H, model, design, fs):
    y = ((L.delta == 1) & (L["T"] <= H)).astype(int).values
    tr = (L.split == "train").values
    if model == "Logistic":
        m = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(C=0.5, max_iter=4000)).fit(L.loc[tr, feats], y[tr])
        p = m.predict_proba(L[feats])[:, 1]
    elif model == "LightGBM":
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=80, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, reg_lambda=5, verbose=-1).fit(L.loc[tr, feats], y[tr])
        p = m.predict_proba(L[feats])[:, 1]
    else:  # physics-anchored AFT (needs ttf_phys)
        lo = np.where(L.delta == 1, L["T"], 2880.0); up = np.where(L.delta == 1, L["T"], np.inf)
        off = np.log(L.ttf_phys.values)
        va = (L.split == "val").values
        dtr = xgb.DMatrix(L.loc[tr, feats], label_lower_bound=lo[tr], label_upper_bound=up[tr]); dtr.set_base_margin(off[tr])
        dva = xgb.DMatrix(L.loc[va, feats], label_lower_bound=lo[va], label_upper_bound=up[va]); dva.set_base_margin(off[va])
        prm = dict(objective="survival:aft", eval_metric="aft-nloglik", aft_loss_distribution="normal", aft_loss_distribution_scale=1.2,
                   learning_rate=0.03, max_depth=4, min_child_weight=20, subsample=0.8, colsample_bytree=0.8, reg_lambda=5, tree_method="hist")
        b = xgb.train(prm, dtr, 600, evals=[(dva, "v")], early_stopping_rounds=60, verbose_eval=False)
        da = xgb.DMatrix(L[feats]); da.set_base_margin(off)
        mu = b.predict(da, output_margin=True, iteration_range=(0, b.best_iteration + 1)); p = norm.cdf((np.log(H) - mu) / 1.2)
    out = []
    ev = pd.DataFrame({"eid": L.eid.values, "split": L.split.values, "cso": L.event_cso.values, "p": p, "T": L["T"].values})
    E = ev.groupby("eid").agg(split=("split", "first"), cso=("cso", "max"), pmax=("p", "max"))
    tau = np.quantile(E[(E.split == "val") & (E.cso == 0)].pmax, 0.95)
    for s in ("val", "test"):
        msk = (L.split == s).values
        e = E[E.split == s]
        leads = []
        for eid in e.index[e.cso == 1]:
            g = ev[(ev.eid == eid) & (ev.p > tau)]
            leads.append(g["T"].iloc[0] if len(g) else 0.0)
        out.append(dict(design=design, features=fs, model=model, H=H, split=s, AUC=roc_auc_score(y[msk], p[msk]), AP=average_precision_score(y[msk], p[msk]),
                        hits=int(((e.cso == 1) & (e.pmax > tau)).sum()), n_cso=int(e.cso.sum()), FA=int(((e.cso == 0) & (e.pmax > tau)).sum()),
                        n0=int((e.cso == 0).sum()), med_lead=float(np.median(leads)) if leads else np.nan))
    return out

rows = []
# ---------------- Design A: storage-onset clock
LA = pd.read_parquet(os.path.join(ROOT, "processed", "landmarks.parquet")).reset_index(drop=True)
LA = pd.concat([LA, rain_feats(LA.t)], axis=1)
st = LA.groupby("eid").t.transform("min"); LA["split"] = split_of(st); LA = LA[LA.split != "drop"].reset_index(drop=True)
for fs, feats in [("SCADA", SC + F04), ("SCADA+rain", SC + F04 + RF), ("SCADA-F04", SC), ("SCADA-F04+rain", SC + RF)]:
    for H in (60, 120, 240):
        for model in ("Logistic", "LightGBM", "Phys-AFT"):
            rows += fit_eval(LA, feats, H, model, "A storage-onset", fs)
    print("A", fs, flush=True)

# ---------------- Design B: rain-onset clock
E13 = pd.read_csv(os.path.join(OUT, "E13_rain_events.csv"), parse_dates=["start"])
S = P.G71F68Y_LevelPS.ffill(limit=6)
cols = dict(f05=P.G71F05R_LevelInlet, f06=P.G71F06R_LevelInlet, f04=P.G71F04R_Level2, gate=P.G71F05R_position, b05=P.G71F05R_LevelBasin, q68=P.G71F68Y_Flow)
cols = {k: v.ffill(limit=6) for k, v in cols.items()}
rowsB = []
for eid, ev in E13.iterrows():
    a = ev.start; w1 = a + pd.Timedelta("24h")
    s = S.loc[a - pd.Timedelta("2h"): w1]
    hit = s.loc[a:][s.loc[a:] >= 4.5]
    tc = hit.index[0] if len(hit) else None
    idx = s.index; i0 = idx.searchsorted(a)
    sv = s.values; cv = {k: v.reindex(idx).values for k, v in cols.items()}
    for i in range(i0, len(idx), 2):
        t = idx[i]
        if tc is not None and t >= tc: break
        if tc is None and (t - a) > pd.Timedelta("24h"): break
        tr_ = lambda x, k: x[i] - x[i - k] if i - k >= 0 else np.nan
        dh30 = tr_(sv, 6); h = sv[i]
        ttf = (4.5 - h) / max(dh30 / 30.0, 1e-4) if np.isfinite(dh30) and np.isfinite(h) else 2880.0
        rowsB.append(dict(eid=eid, t=t, elapsed=(t - a).total_seconds() / 60, h=h, dh30=dh30, dh60=tr_(sv, 12), ttf_phys=min(max(ttf, 1.0), 2880.0),
                          f05=cv["f05"][i], f05_d30=tr_(cv["f05"], 6), f06=cv["f06"][i], f06_d30=tr_(cv["f06"], 6), f04=cv["f04"][i], f04_d30=tr_(cv["f04"], 6),
                          gate=cv["gate"][i], b05=cv["b05"][i], q68=cv["q68"][i], doy_s=ev.doy_s, doy_c=ev.doy_c,
                          T=(tc - t).total_seconds() / 60 if tc is not None else 2880.0, delta=int(tc is not None), event_cso=int(tc is not None), start=a))
LB = pd.DataFrame(rowsB)
LB = pd.concat([LB.reset_index(drop=True), rain_feats(LB.t)], axis=1)
LB["split"] = split_of(LB.start); LB = LB[LB.split != "drop"].reset_index(drop=True)
LB.to_parquet(os.path.join(ROOT, "processed", "landmarks_rain_onset.parquet"))
print("Design B landmarks:", LB.shape, LB.groupby("split").eid.nunique().to_dict(), "CSO events:", LB.groupby("split").apply(lambda d: d.groupby("eid").event_cso.max().sum()).to_dict())
SCB = ["elapsed", "h", "dh30", "dh60", "ttf_phys", "f05", "f05_d30", "f06", "f06_d30", "gate", "b05", "q68", "doy_s", "doy_c"]
for fs, feats in [("rain only", RF + ["elapsed", "doy_s", "doy_c"]), ("SCADA", SCB + ["f04", "f04_d30"]), ("SCADA+rain", SCB + ["f04", "f04_d30"] + RF), ("SCADA-F04+rain", SCB + RF)]:
    for H in (60, 120, 240, 360):
        for model in ("Logistic", "LightGBM", "Phys-AFT"):
            rows += fit_eval(LB, feats, H, model, "B rain-onset", fs)
    print("B", fs, flush=True)
R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, "E14_warning_with_rain.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
T = R[R.split == "test"]
print(T.pivot_table(index=["design", "features"], columns=["model", "H"], values="AP").round(3).to_string())
print(T[T.H.isin([120, 240])][["design", "features", "model", "H", "hits", "n_cso", "FA", "n0", "med_lead"]].to_string(index=False))
