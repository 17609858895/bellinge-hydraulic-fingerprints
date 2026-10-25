"""Localization: dry-wet plane segmentation for every node twin; type probability by bootstrap of the break window."""
import os, sys, numpy as np, pandas as pd, importlib.util
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("dw", os.path.join(ROOT, "scripts", "09b_drywet_plane.py")); dw = importlib.util.module_from_spec(spec); spec.loader.exec_module(dw)
NODES = {"G71F04R_Level2": (0.15, 0.25), "G71F06R_LevelInlet": (0.15, 0.30), "G71F05R_LevelInlet": (0.28, 0.36), "G71F68Y_LevelPS": (0.6, 1.0)}
rng = np.random.default_rng(11)
rows = []
for node, (kd, kw) in NODES.items():
    dw.KD, dw.KW = kd, kw
    D = pd.read_parquet(os.path.join(ROOT, "processed", f"twin2_{node}.parquet"))
    M = dw.month_stats(D, B=40)
    M0 = M.loc[dw.REF0:dw.REF1]; dw.set_floor(M0, ["A", "W"])
    X0, S0 = M0[["A", "W"]].values, dw.floor_se(M0, ["A", "W"])
    null = []
    for _ in range(300):
        blocks = [np.arange(i, min(i + 3, len(X0))) for i in range(0, len(X0), 3)]
        o = np.concatenate([blocks[j] for j in rng.permutation(len(blocks))]); null.append(dw.max_gain(X0[o], S0[o]))
    pen = float(np.quantile(null, 0.95))
    bks = dw.binseg(M[["A", "W"]].values, dw.floor_se(M, ["A", "W"]), pen)
    M.to_csv(os.path.join(ROOT, "results", f"E11_monthly_AW_{node}.csv"))
    for b in bks:
        tp, info = dw.classify(M, b)
        # bootstrap type probability: resample months within +-6 windows
        votes = []
        for _ in range(500):
            pre = M.iloc[max(0, b - 6):b].sample(frac=1, replace=True, random_state=int(rng.integers(1e9)))
            post = M.iloc[b:b + 6].sample(frac=1, replace=True, random_state=int(rng.integers(1e9)))
            Mb = pd.concat([pre, post]); votes.append(dw.classify(Mb, len(pre))[0])
        v = pd.Series(votes).value_counts(normalize=True)
        rows.append(dict(node=node, prev_month=M.index[b - 1].strftime("%Y-%m"), break_month=M.index[b].strftime("%Y-%m"), type=tp,
                         **{k: round(x, 3) for k, x in info.items()}, **{f"p_{k}": round(v.get(k, 0), 2) for k in ("offset", "gain", "conveyance", "weak")}))
    print(node, "penalty", round(pen, 1), "breaks", len(bks), flush=True)
R = pd.DataFrame(rows); R.to_csv(os.path.join(ROOT, "results", "E11_node_breaks.csv"), index=False)
pd.set_option("display.width", 250); print(R.to_string(index=False))
