"""Build per-signal 1-min parquet and a 5-min multi-sensor panel from the author-cleaned Bellinge CSVs.
Uses `depth_s` (author-cleaned, includes legitimate forward-fill for change-based iFix storage) and keeps
`ffill` / error flags as quality fractions. Raw files are never modified."""
import glob, os, re, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.environ["BELLINGE_SENSOR_DIR"]
OUT = os.path.join(ROOT, "processed")

FLAGS = ["man_remove", "ffill", "stamp", "outbound", "frozen", "outlier"]

def signal_name(fn):
    b = os.path.basename(fn).replace("_proc_v6.csv", "")
    b = re.sub(r"_(System2000|iFix|Danova)p\d+$", "", b)
    b = re.sub(r"_(Danova)p\d+$", "", b)
    b = b.replace("G71F68Yp1", "G71F68Y_Flow")
    if b.endswith("_Danova"):
        b = b[:-7]
    return b

files = sorted(glob.glob(os.path.join(SRC, "*.csv")))
if not files: raise FileNotFoundError("No sensor CSV files in BELLINGE_SENSOR_DIR")
groups = {}
for f in files:
    groups.setdefault(signal_name(f), []).append(f)
for k, v in groups.items():
    print(k, [os.path.basename(x) for x in v])

summ = []
five = {}
for sig, fl in groups.items():
    parts = []
    for f in fl:
        cols = pd.read_csv(f, nrows=0).columns.tolist()
        use = ["time", "depth_s", "value_no_errors"] + [c for c in FLAGS if c in cols]
        d = pd.read_csv(f, usecols=use, engine="pyarrow")
        d["time"] = pd.to_datetime(d["time"])
        for c in FLAGS:
            if c in d:
                d[c] = d[c].astype(str).str.lower().eq("true")
        d["src"] = "iFix" if "iFix" in f else ("S2000" if "System2000" in f else "Danova")
        parts.append(d)
    d = pd.concat(parts).sort_values("time")
    d = d.drop_duplicates("time", keep="last").set_index("time")
    d["depth_s"] = d["depth_s"].astype("float32")
    d["meas"] = d["value_no_errors"].notna()
    d[["depth_s", "meas", "ffill", "src"]].to_parquet(os.path.join(OUT, f"sig_{sig}.parquet"))
    r = d["depth_s"].resample("5min")
    five[sig] = r.mean()
    five[sig + "__max"] = r.max()
    five[sig + "__meas"] = d["meas"].resample("5min").mean().astype("float32")
    summ.append(dict(signal=sig, start=d.index.min(), end=d.index.max(), n_min=len(d),
                     frac_valid=float(d["depth_s"].notna().mean()),
                     frac_measured=float(d["meas"].mean()),
                     frac_ffill=float(d["ffill"].mean()),
                     q50=float(d["depth_s"].median()), q99=float(d["depth_s"].quantile(0.99)),
                     vmax=float(d["depth_s"].max())))
    print(summ[-1], flush=True)
    del d, parts

P = pd.DataFrame(five).astype("float32")
P.to_parquet(os.path.join(OUT, "panel_5min.parquet"))
pd.DataFrame(summ).to_csv(os.path.join(ROOT, "results", "T01_signal_summary.csv"), index=False)
print(P.shape)
