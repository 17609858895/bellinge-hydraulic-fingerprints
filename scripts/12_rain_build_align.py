"""Rain gauges 5425 (Brændekilde) and 5427 (Dalum), DTU Bellinge item 3a v2 (CC BY-NC 4.0, research use).
ts files: minute mean intensity [um/s] during gauge-defined events, UTC; absent minutes = 0 rain (outside
listed outages). Depth per minute = value * 0.06 mm.
SCADA clock: manual says local UTC+1 with incomplete DST flags -> estimate the offset empirically by
cross-correlating 5-min rain with the G71F05R inlet level rise, separately for summer/winter and year."""
import os, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAIN = os.environ["BELLINGE_RAIN_DIR"]
OUT = os.path.join(ROOT, "results")

def load(g):
    d = pd.read_csv(os.path.join(RAIN, f"{g}_ts.txt"), sep=";", skiprows=1)
    d.columns = ["time", "i"]
    d["time"] = pd.to_datetime(d["time"])
    s = d.set_index("time")["i"].astype(float) * 0.06          # mm per minute
    s = s[~s.index.duplicated()]
    return s

def outages(g):
    rows = []
    for fn in (f"{g}_missing_data.txt", f"{g}_planned_closing_and_not_accepted_data.txt"):
        for ln in open(os.path.join(RAIN, fn), encoding="latin-1"):
            p = ln.split()
            if len(p) >= 4 and p[0].isdigit() and len(p[0]) == 8 and p[2].isdigit():
                rows.append((pd.to_datetime(p[0] + p[1], format="%Y%m%d%H:%M"), pd.to_datetime(p[2] + p[3], format="%Y%m%d%H:%M")))
    return rows

R = {}
for g in ("5425", "5427"):
    s = load(g)
    idx = pd.date_range("2009-06-01", "2021-08-19 23:55", freq="5min")
    r5 = s.resample("5min").sum().reindex(idx, fill_value=0.0)
    for a, b in outages(g):
        r5[a:b] = np.nan
    R[g] = r5
    print(g, "annual totals (mm):", r5.resample("YS").sum().round(0).astype(int).to_dict())
rain_utc = pd.DataFrame(R)
rain_utc["mean"] = rain_utc[["5425", "5427"]].mean(axis=1)
print("gauge corr (hourly):", rain_utc[["5425", "5427"]].resample("h").sum().corr().iloc[0, 1].round(3))

# --- empirical clock offset: SCADA_time = UTC + offset
P = pd.read_parquet(os.path.join(ROOT, "processed", "panel_5min.parquet"))
resp = P["G71F05R_LevelInlet"].diff().clip(lower=0)
rows = []
for yr in range(2010, 2022):
    for season, months in (("summer", [5, 6, 7, 8, 9]), ("winter", [11, 12, 1, 2, 3])):
        best = []
        for off_min in range(-60, 241, 5):
            x = rain_utc["mean"].copy(); x.index = x.index + pd.Timedelta(minutes=off_min)
            m = (x.index.year == yr) & x.index.month.isin(months)
            xx = x[m]; yy = resp.reindex(xx.index)
            ok = xx.notna() & yy.notna()
            if ok.sum() < 5000 or xx[ok].sum() < 20:
                continue
            best.append((off_min, np.corrcoef(xx[ok], yy[ok])[0, 1]))
        if best:
            o, c = max(best, key=lambda t: t[1])
            rows.append(dict(year=yr, season=season, best_shift_min=o, r=c))
A = pd.DataFrame(rows); print(A.to_string(index=False))
A.to_csv(os.path.join(OUT, "E12_clock_offset.csv"), index=False)
rain_utc.astype("float32").to_parquet(os.path.join(ROOT, "processed", "rain_5min_utc.parquet"))
