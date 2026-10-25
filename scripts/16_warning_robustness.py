"""E16: multi-seed, multi-model robustness of the CSO warning results (E10, E14) + S10 CSO-threshold sensitivity.
Seed-averaged predictions; paired event-cluster bootstrap (B=500) for dAP(+rain) and dAP(quarantine G71F04R)."""
import os, sys, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wlib as W

SEEDS = {"Logistic": [0], "LightGBM": [0, 1, 2, 3, 4], "Phys-AFT": [0, 1, 2, 3, 4], "XGB-AFT": [0, 1, 2]}
rng = np.random.default_rng(2026)
rows, boots = [], []

def ap_by_event_boot(L, y, preds, pairs, B=500):
    te = (L.split == "test").values
    eids = L.eid.values[te]; ue = np.unique(eids); pos = {e: np.where(eids == e)[0] for e in ue}
    yy = y[te]; out = []
    idxs = [np.concatenate([pos[e] for e in rng.choice(ue, len(ue))]) for _ in range(B)]
    for a, b in pairs:
        pa, pb = preds[a][te], preds[b][te]
        d = [average_precision_score(yy[i], pb[i]) - average_precision_score(yy[i], pa[i]) for i in idxs if yy[i].sum() > 0]
        out.append((a, b, float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(np.mean(np.array(d) > 0))))
    return out

for design, L, HS, SC, F04 in [("A storage-onset", W.design_A(), (60, 120, 240), W.SC_A, W.F04_A),
                               ("B rain-onset", W.design_B(4.5), (60, 120, 240, 360), W.SC_B, W.F04_B)]:
    FS = {"SCADA": SC + F04, "SCADA+rain": SC + F04 + W.RF, "SCADA-F04": SC, "SCADA-F04+rain": SC + W.RF}
    for H in HS:
        y = ((L.delta == 1) & (L["T"] <= H)).astype(int).values
        for model, seeds in SEEDS.items():
            mean_pred = {}
            for fs, feats in FS.items():
                ps = []
                for s in seeds:
                    p = W.predict(L, feats, H, model, seed=s); ps.append(p)
                    te = (L.split == "test").values; va = (L.split == "val").values
                    ev = W.event_eval(L, p)
                    rows.append(dict(design=design, H=H, model=model, features=fs, seed=s, AP_val=average_precision_score(y[va], p[va]),
                                     AP_test=average_precision_score(y[te], p[te]), **{f"test_{k}": v for k, v in ev["test"].items()}))
                mean_pred[fs] = np.mean(ps, axis=0)
            for a, b, d, lo, hi, pp in ap_by_event_boot(L, y, mean_pred, [("SCADA", "SCADA+rain"), ("SCADA", "SCADA-F04"), ("SCADA-F04", "SCADA-F04+rain"), ("SCADA", "SCADA-F04+rain")]):
                boots.append(dict(design=design, H=H, model=model, base=a, variant=b, dAP=d, lo=lo, hi=hi, p_gain=pp))
            print(design, H, model, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(W.OUT, "E16_robustness_runs.csv"), index=False)
    pd.DataFrame(boots).to_csv(os.path.join(W.OUT, "E16_robustness_bootstrap.csv"), index=False)

# S10: CSO threshold sensitivity for the warning (design B, SCADA+rain)
s10 = []
for thr in (4.4, 4.5, 4.6):
    L = W.design_B(thr)
    for H in (60, 120, 240):
        y = ((L.delta == 1) & (L["T"] <= H)).astype(int).values; te = (L.split == "test").values
        for model in ("Phys-AFT", "LightGBM", "Logistic"):
            ps = [W.predict(L, W.SC_B + W.F04_B + W.RF, H, model, seed=s) for s in (SEEDS[model][:3])]
            p = np.mean(ps, axis=0); ev = W.event_eval(L, p)["test"]
            s10.append(dict(thr=thr, H=H, model=model, n_cso_events_test=ev["n_cso"], AP_test=average_precision_score(y[te], p[te]), **ev))
    print("S10", thr, flush=True)
pd.DataFrame(s10).to_csv(os.path.join(W.OUT, "E16_S10_cso_threshold_warning.csv"), index=False)

R = pd.DataFrame(rows); Bt = pd.DataFrame(boots)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
S = R.groupby(["design", "H", "model", "features"]).agg(AP_test=("AP_test", "mean"), AP_sd=("AP_test", "std"), AP_val=("AP_val", "mean"),
                                                        hits=("test_hits", "mean"), FA=("test_FA", "mean"), lead=("test_med_lead", "mean")).reset_index()
S.to_csv(os.path.join(W.OUT, "E16_robustness_summary.csv"), index=False)
print(S.round(3).to_string(index=False)); print(Bt.round(3).to_string(index=False)); print(pd.DataFrame(s10).round(3).to_string(index=False))
