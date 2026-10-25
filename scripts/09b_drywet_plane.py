"""Dry-wet residual plane (v2 of change typing).

For each month: A = median twin residual where yhat < kappa_dry (dry-weather band), W = median residual where yhat > kappa_wet,
with day-block bootstrap SEs. Change vector (dA, dW) at a break is typed by geometry:
  offset (zero-point)  : dW ~ dA
  gain (scaling)       : dW ~ k dA, k = median(yhat | wet) / median(yhat | dry)
  conveyance (hydraulic): dA ~ 0, dW != 0   (wet-only)
Segmentation: SE-weighted binary segmentation on (A, W); penalty calibrated on the clean reference with a
moving-block permutation null (95th percentile of max split gain). Injection test as in script 09.
"""
import os, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")
TGT, KD, KW = "G71F04R_Level2", 0.15, 0.25
REF0, REF1 = "2015-06-01", "2018-08-31"
rng = np.random.default_rng(7)

def month_stats(D, B=60):
    rows = {}
    for mon, g in D.groupby(pd.Grouper(freq="MS")):
        if len(g) < 3000:
            continue
        r = (g.y - g.yhat).values; yh = g.yhat.values
        dry, wet = yh < KD, yh > KW
        codes, ud = pd.factorize(g.index.normalize())
        A = np.median(r[dry]) if dry.sum() > 100 else np.nan
        W = np.median(r[wet]) if wet.sum() > 24 else np.nan
        bA, bW = [], []
        for _ in range(B):
            pick = rng.integers(0, len(ud), len(ud))
            m = np.isin(codes, pick)  # day-resample approximated by subset (fast); weights by multiplicity ignored
            cnt = np.bincount(pick, minlength=len(ud))[codes]
            if dry.sum() > 100:
                rr = np.repeat(r[dry], cnt[dry]); bA.append(np.median(rr) if len(rr) else np.nan)
            if wet.sum() > 24:
                rr = np.repeat(r[wet], cnt[wet]); bW.append(np.median(rr) if len(rr) > 5 else np.nan)
        rows[mon] = dict(A=A, W=W, A_se=np.nanstd(bA) if bA else np.nan, W_se=np.nanstd(bW) if bW else np.nan,
                         yd=np.median(yh[dry]) if dry.sum() else np.nan, yw=np.median(yh[wet]) if wet.sum() else np.nan,
                         n_wet=int(wet.sum()), raw_med=np.median(g.y), raw_p95=np.quantile(g.y, .95), res_mean=r.mean())
    return pd.DataFrame(rows).T.astype(float)

def wcost(X, S):
    """SE-weighted squared deviation from weighted mean, NaN-aware. X,S: (n,d)."""
    w = 1 / np.where(np.isfinite(S) & (S > 0), S, np.inf) ** 2
    w = np.where(np.isfinite(X), w, 0); Xf = np.where(np.isfinite(X), X, 0)
    mu = (w * Xf).sum(0) / np.maximum(w.sum(0), 1e-12)
    return (w * (Xf - mu) ** 2).sum()

def binseg(X, S, pen, minseg=3):
    bks, todo = [], [(0, len(X))]
    while todo:
        s, e = todo.pop()
        if e - s < 2 * minseg: continue
        base = wcost(X[s:e], S[s:e]); best, bk = 0, None
        for k in range(s + minseg, e - minseg + 1):
            gn = base - wcost(X[s:k], S[s:k]) - wcost(X[k:e], S[k:e])
            if gn > best: best, bk = gn, k
        if bk is not None and best > pen:
            bks.append(bk); todo += [(s, bk), (bk, e)]
    return sorted(bks)

def max_gain(X, S, minseg=3):
    base = wcost(X, S)
    return max(base - wcost(X[:k], S[:k]) - wcost(X[k:], S[k:]) for k in range(minseg, len(X) - minseg + 1))

FLOOR = {}
def set_floor(M0, cols):
    """SE floor = month-to-month robust scale in the clean reference (between-month drift not in the day-bootstrap)."""
    for c in cols:
        d = M0[c].diff().dropna()
        FLOOR[c] = 1.4826 * np.median(np.abs(d - d.median())) / np.sqrt(2)

def floor_se(M, cols):
    S = M[[c + "_se" for c in cols]].values.copy()
    for j, c in enumerate(cols):
        S[:, j] = np.sqrt(np.nan_to_num(S[:, j], nan=np.inf) ** 2 + FLOOR[c] ** 2)
    return S

def classify(M, k, w=6):
    pre, post = M.iloc[max(0, k - w):k], M.iloc[k:k + w]
    dA = post.A.mean() - pre.A.mean(); dW = post.W.mean() - pre.W.mean()
    seA = np.sqrt(pre.A.var() / max(pre.A.count(), 1) + post.A.var() / max(post.A.count(), 1) + 1e-10)
    seW = np.sqrt(pre.W.var() / max(pre.W.count(), 1) + post.W.var() / max(post.W.count(), 1) + 1e-10)
    kk = (pd.concat([pre, post]).yw / pd.concat([pre, post]).yd).median()
    zA, zW = dA / seA, dW / seW
    if abs(zA) < 2 and abs(zW) < 2:
        typ = "weak"
    elif abs(zA) < 2 or dW * dA < 0 or abs(dW / dA) > 2 * kk:
        typ = "conveyance"          # wet-only signature
    else:
        r = abs(dW / dA)            # log-distance to the offset (1) and gain (k) signatures
        typ = "offset" if abs(np.log(r)) < abs(np.log(r / kk)) else "gain"
    return typ, dict(dA=dA, dW=dW, zA=zA, zW=zW, k=kk, ratio=dW / dA if dA else np.nan)

if __name__ == "__main__":
    D = pd.read_parquet(os.path.join(ROOT, "processed", f"twin2_{TGT}_frozen.parquet"))
    REF = D.loc[REF0:REF1]
    M0 = month_stats(REF)
    set_floor(M0, ["A", "W"])
    print("SE floors", {k: round(v, 4) for k, v in FLOOR.items()})
    X0, S0 = M0[["A", "W"]].values, floor_se(M0, ["A", "W"])
    # null: moving-block (3-month) permutation of the clean reference
    null = []
    for _ in range(300):
        blocks = [np.arange(i, min(i + 3, len(X0))) for i in range(0, len(X0), 3)]
        order = np.concatenate([blocks[j] for j in rng.permutation(len(blocks))])
        null.append(max_gain(X0[order], S0[order]))
    PEN = float(np.quantile(null, 0.95))
    print("penalty (95% block-permutation null):", round(PEN, 2))
    PEN_U = {}
    for c in ("raw_med", "raw_p95", "res_mean"):
        x = M0[[c]].values; s = np.full_like(x, 1.4826 * np.median(np.abs(np.diff(x[:, 0]) - np.median(np.diff(x[:, 0])))) / np.sqrt(2))
        nl = []
        for _ in range(300):
            blocks = [np.arange(i, min(i + 3, len(x))) for i in range(0, len(x), 3)]
            o = np.concatenate([blocks[j] for j in rng.permutation(len(blocks))]); nl.append(max_gain(x[o], s[o]))
        PEN_U[c] = (float(np.quantile(nl, 0.95)), s[0, 0])
    months = M0.index
    rows = []
    designs = [("none", 0)] + [("offset", d) for d in (0.01, 0.02, 0.04)] + [("gain", g) for g in (0.10, 0.20, 0.45)] + [("conveyance", c) for c in (0.05, 0.10, 0.20, 0.30)] + [("conveyance_hinge", c) for c in (0.5, 2.0)]
    for typ, size in designs:
        for rep in range(25):
            k0 = int(rng.integers(9, len(months) - 9)); t0 = months[k0]
            Xd = REF.copy(); post = Xd.index >= t0
            if typ == "offset": Xd.loc[post, "y"] += size
            elif typ == "gain": Xd.loc[post, "y"] *= (1 + size)
            elif typ == "conveyance":  # weir-capped wet-only step: ramps in over 0.10 m above kappa, then constant
                Xd.loc[post, "y"] += size * np.clip((Xd.loc[post, "y"] - KW) / 0.10, 0, 1)
            elif typ == "conveyance_hinge":  # unbounded variant (stress test)
                Xd.loc[post, "y"] += size * np.maximum(Xd.loc[post, "y"] - KW, 0)
            M = month_stats(Xd, B=30)
            bks = binseg(M[["A", "W"]].values, floor_se(M, ["A", "W"]), PEN)
            hit = [b for b in bks if abs(b - k0) <= 2]
            r = dict(type=typ, size=size, rep=rep, method="dry-wet plane", detected=int(bool(hit)), false_breaks=len(bks) - min(len(hit), 1))
            if hit:
                tp, info = classify(M, hit[0]); r.update(typed=tp, correct=int(tp == typ.split("_")[0]))
            rows.append(r)
            for c, (pen, sd) in PEN_U.items():
                x = M[[c]].values; s = np.full_like(x, sd)
                b2 = binseg(x, s, pen); h2 = [b for b in b2 if abs(b - k0) <= 2]
                rows.append(dict(type=typ, size=size, rep=rep, method=c, detected=int(bool(h2)), false_breaks=len(b2) - min(len(h2), 1)))
        print(typ, size, flush=True)
    R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, "E09b_injection_runs.csv"), index=False)
    S = R.groupby(["type", "size", "method"]).agg(detect=("detected", "mean"), false_br=("false_breaks", "mean"), type_acc=("correct", "mean")).reset_index()
    S.to_csv(os.path.join(OUT, "E09b_injection_summary.csv"), index=False)
    pd.set_option("display.width", 220)
    print(S.pivot_table(index=["type", "size"], columns="method", values=["detect", "false_br"]).round(2))
    print(R[R.method == "dry-wet plane"].dropna(subset=["typed"]).groupby(["type", "typed"]).size().unstack(fill_value=0))

    # ---- real timeline
    M = month_stats(D)
    M.to_csv(os.path.join(OUT, f"E09b_monthly_AW_{TGT}.csv"))
    bks = binseg(M[["A", "W"]].values, floor_se(M, ["A", "W"]), PEN)
    out = []
    for b in bks:
        tp, info = classify(M, b)
        out.append(dict(break_month=M.index[b].strftime("%Y-%m"), type=tp, **{k: round(v, 3) for k, v in info.items()}))
    print(pd.DataFrame(out).to_string(index=False))
    pd.DataFrame(out).to_csv(os.path.join(OUT, "E09b_real_breaks.csv"), index=False)
