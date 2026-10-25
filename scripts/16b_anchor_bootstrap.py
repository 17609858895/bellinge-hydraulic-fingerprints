"""E16b: event-cluster bootstrap for the physics anchor (Phys-AFT vs XGB-AFT, same features incl. rain, 3 seeds averaged)."""
import os, sys, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wlib as W
rng = np.random.default_rng(7); rows = []
for design, L, HS, feats in [("A storage-onset", W.design_A(), (60, 120, 240), W.SC_A + W.F04_A + W.RF),
                             ("B rain-onset", W.design_B(4.5), (60, 120, 240, 360), W.SC_B + W.F04_B + W.RF)]:
    te = (L.split == "test").values; eids = L.eid.values[te]; ue = np.unique(eids); pos = {e: np.where(eids == e)[0] for e in ue}
    idxs = [np.concatenate([pos[e] for e in rng.choice(ue, len(ue))]) for _ in range(1000)]
    for H in HS:
        y = ((L.delta == 1) & (L["T"] <= H)).astype(int).values[te]
        pa = np.mean([W.predict(L, feats, H, "XGB-AFT", seed=s) for s in (0, 1, 2)], axis=0)[te]
        pb = np.mean([W.predict(L, feats, H, "Phys-AFT", seed=s) for s in (0, 1, 2)], axis=0)[te]
        d = np.array([average_precision_score(y[i], pb[i]) - average_precision_score(y[i], pa[i]) for i in idxs if y[i].sum() > 0])
        rows.append(dict(design=design, H=H, AP_plain=average_precision_score(y, pa), AP_anchor=average_precision_score(y, pb),
                         dAP=d.mean(), lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5), p_gain=(d > 0).mean()))
    print(design, flush=True)
R = pd.DataFrame(rows); R.to_csv(os.path.join(W.OUT, "E16b_physics_anchor_bootstrap.csv"), index=False); print(R.round(3).to_string(index=False))
