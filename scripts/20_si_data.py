"""E20: data for SI figures not produced earlier.
S6  stability of G71F04R breaks to band thresholds (kappa_d, kappa_w), minimum segment length and penalty quantile.
S9  test calibration (reliability) and AUC/AP/Brier of the four warning models, design B, 120 min, SCADA+rain."""
import os, sys, importlib.util, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FD = os.path.join(ROOT, "results", "figdata")
spec = importlib.util.spec_from_file_location("dw", os.path.join(ROOT, "scripts", "09b_drywet_plane.py")); dw = importlib.util.module_from_spec(spec); spec.loader.exec_module(dw)
rng = np.random.default_rng(20)

# ---- S6
D = pd.read_parquet(os.path.join(ROOT, "processed", "twin2_G71F04R_Level2.parquet"))
rows = []
for kd, kw in ((0.12, 0.22), (0.15, 0.25), (0.18, 0.30)):
    dw.KD, dw.KW = kd, kw
    M = dw.month_stats(D, B=40); M0 = M.loc[dw.REF0:dw.REF1]; dw.set_floor(M0, ["A", "W"])
    X0, S0 = M0[["A", "W"]].values, dw.floor_se(M0, ["A", "W"])
    X, S = M[["A", "W"]].values, dw.floor_se(M, ["A", "W"])
    for minseg in (2, 3, 4):
        null = []
        for _ in range(300):
            blocks = [np.arange(i, min(i + 3, len(X0))) for i in range(0, len(X0), 3)]
            o = np.concatenate([blocks[j] for j in rng.permutation(len(blocks))]); null.append(dw.max_gain(X0[o], S0[o], minseg))
        for q in (0.90, 0.95, 0.99):
            pen = float(np.quantile(null, q))
            for b in dw.binseg(X, S, pen, minseg):
                tp, info = dw.classify(M, b)
                rows.append(dict(kd=kd, kw=kw, minseg=minseg, q=q, penalty=pen, break_month=M.index[b].strftime("%Y-%m"), type=tp))
    print("S6", kd, kw, flush=True)
pd.DataFrame(rows).to_csv(os.path.join(FD, "figS6_break_stability.csv"), index=False)

# ---- S9
sys.path.insert(0, os.path.join(ROOT, "scripts")); import wlib as W
L = W.design_B(4.5); feats = W.SC_B + W.F04_B + W.RF; H = 120
y = ((L.delta == 1) & (L["T"] <= H)).astype(int).values; te = (L.split == "test").values
seeds = {"Logistic": [0], "LightGBM": [0, 1, 2], "Phys-AFT": [0, 1, 2], "XGB-AFT": [0, 1, 2]}
rel, met = [], []
for m, ss in seeds.items():
    p = np.mean([W.predict(L, feats, H, m, seed=s) for s in ss], axis=0)[te]; yy = y[te]
    met.append(dict(model=m, AUC=roc_auc_score(yy, p), AP=average_precision_score(yy, p), Brier=brier_score_loss(yy, np.clip(p, 0, 1)),
                    mean_pred=p.mean(), obs_rate=yy.mean()))
    bins = np.quantile(p, np.linspace(0, 1, 11)); bins[0], bins[-1] = -1e-9, 1 + 1e-9
    b = np.digitize(p, np.unique(bins)) - 1
    for k in np.unique(b):
        mk = b == k
        rel.append(dict(model=m, bin=int(k), pred=p[mk].mean(), obs=yy[mk].mean(), n=int(mk.sum())))
pd.DataFrame(rel).to_csv(os.path.join(FD, "figS9_reliability.csv"), index=False)
pd.DataFrame(met).to_csv(os.path.join(FD, "figS9_metrics.csv"), index=False)
print(pd.DataFrame(met).round(4).to_string(index=False))
