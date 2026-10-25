"""E21: post-hoc calibration of the warning probabilities.
Calibrators are fitted on the validation period (2017-01..2018-09) only and applied to the post-swap test period:
Platt  = logistic regression on logit(p);  isotonic = monotone step function (clipped outside the fitted range).
Both are monotone, so event-level alarms at a validation-FAR threshold are unchanged (isotonic can create ties).
Models: Phys-AFT, XGB-AFT (the under-predicting ones) plus Logistic and LightGBM for reference; seeds averaged.
Design B (rain-onset clock) at 60/120/240 min and design A (storage-onset clock) at 120 min; SCADA + rain features."""
import os, sys, numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wlib as W
FD = os.path.join(W.ROOT, "results", "figdata")
SEEDS = {"Phys-AFT": [0, 1, 2], "XGB-AFT": [0, 1, 2], "Logistic": [0], "LightGBM": [0, 1, 2]}

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))

def ece(y, p, k=10):
    q = np.unique(np.quantile(p, np.linspace(0, 1, k + 1))); b = np.clip(np.digitize(p, q[1:-1]), 0, len(q) - 2)
    return sum((b == i).mean() * abs(y[b == i].mean() - p[b == i].mean()) for i in np.unique(b))

def reliability(y, p, k=10):
    q = np.unique(np.quantile(p, np.linspace(0, 1, k + 1))); b = np.clip(np.digitize(p, q[1:-1]), 0, len(q) - 2)
    return [(p[b == i].mean(), y[b == i].mean(), int((b == i).sum())) for i in np.unique(b)]

rows, rel, tc = [], [], None
for design, L, feats, HS in [("B rain-onset", W.design_B(4.5), W.SC_B + W.F04_B + W.RF, (60, 120, 240)),
                             ("A storage-onset", W.design_A(), W.SC_A + W.F04_A + W.RF, (120,))]:
    va = (L.split == "val").values; te = (L.split == "test").values
    for H in HS:
        y = ((L.delta == 1) & (L["T"] <= H)).astype(int).values
        for m, ss in SEEDS.items():
            p = np.mean([W.predict(L, feats, H, m, seed=s) for s in ss], axis=0)
            platt = LogisticRegression(C=1e6, max_iter=5000).fit(logit(p[va]).reshape(-1, 1), y[va])
            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p[va], y[va])
            cal = {"raw": p, "Platt": platt.predict_proba(logit(p).reshape(-1, 1))[:, 1], "isotonic": iso.predict(p)}
            for c, pc in cal.items():
                ev = W.event_eval(L, pc)["test"]
                rows.append(dict(design=design, H=H, model=m, calibration=c, n_val_pos=int(y[va].sum()), n_test=int(te.sum()), n_test_pos=int(y[te].sum()),
                                 AUC=roc_auc_score(y[te], pc[te]), AP=average_precision_score(y[te], pc[te]), Brier=brier_score_loss(y[te], pc[te]),
                                 ECE=ece(y[te], pc[te]), ratio=pc[te].mean() / y[te].mean(), hits=ev["hits"], n_cso=ev["n_cso"], FA=ev["FA"], n0=ev["n0"],
                                 med_lead=ev["med_lead"]))
                if design.startswith("B") and H == 120:
                    for pm, om, n in reliability(y[te], pc[te]):
                        rel.append(dict(model=m, calibration=c, pred=pm, obs=om, n=n))
                if design.startswith("B") and H == 120 and m == "Phys-AFT":
                    if tc is None:
                        tc = pd.read_parquet(os.path.join(FD, "fig6_designB_timecourse.parquet"))
                    tc[f"p120_{c}"] = pc
            print(design, H, m, flush=True)
R = pd.DataFrame(rows); R.to_csv(os.path.join(W.OUT, "E21_calibration.csv"), index=False)
pd.DataFrame(rel).to_csv(os.path.join(FD, "figS9_reliability_calibrated.csv"), index=False)
# validation-FAR thresholds on the calibrated scale for Fig. 6a
E = tc.groupby("eid").agg(split=("split", "first"), cso=("event_cso", "max"), pmax=("p120_Platt", "max"))
tc["tau_Platt"] = np.quantile(E[(E.split == "val") & (E.cso == 0)].pmax, 0.95)
tc.to_parquet(os.path.join(FD, "fig6_designB_timecourse.parquet"))
pd.set_option("display.width", 250)
print(R.round(4).to_string(index=False))
