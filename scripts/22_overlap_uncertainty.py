"""Overlap-cluster uncertainty sensitivity for three retained intervention outcomes.

Preserves the original point estimators. Reproduces the original estimates before
calculating cluster-sandwich and fixed-model cluster-bootstrap intervals.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import t as student_t
from sklearn.model_selection import KFold

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / 'results'
OUT = PROJECT / 'results_corrected'
OUT.mkdir(parents=True, exist_ok=True)
FX = ['dur_h','depth','d5425','d5427','i5','i30','i60','ante24','ante7d','doy_s','doy_c','sump_pre']
W = pd.read_csv(SRC / 'E13_rain_events.csv', parse_dates=['start'])
# E13 fitted directly to its assembled in-memory table, whose rainfall and
# hydraulic columns retained source float32 dtypes. Reassemble that table only;
# stop before any original output write or model fitting. E15 instead read CSV.
e13_source = PROJECT / 'scripts' / '13_retrofit_rain_causal.py'
prefix = e13_source.read_text(encoding='utf-8').split('W.to_csv',1)[0]
prefix = prefix.replace('pd.Timedelta(minutes=int(sys.argv[1]) if len(sys.argv) > 1 else 60)','pd.Timedelta(minutes=0)')
ns = {'__file__':str(e13_source),'__name__':'event_assembly'}
exec(compile(prefix,str(e13_source),'exec'),ns)
W13 = ns['W']
assert np.array_equal(W.start.values,W13.start.values)
original_dtypes = {c:str(W13[c].dtype) for c in FX}
M = pd.read_csv(SRC / 'E11_monthly_AW_G71F04R_Level2.csv', index_col=0, parse_dates=True)
offset = M.loc['2019-11-01':,'A'].median() - M.loc['2018-10-01':'2019-06-30','A'].median()
W['f04_peak_c'] = W.f04_peak - np.where(W.start >= '2019-11-01', offset, 0)
W['response_end'] = W.start + pd.to_timedelta(W.dur_h,unit='h') - pd.Timedelta('5min') + pd.Timedelta('12h')

# Connected components use all retained rainfall-response windows, prior to
# outcome-specific missingness exclusions.
cluster = np.zeros(len(W),dtype=int)
active_end = None
g = -1
for i in W.sort_values('start').index:
    if active_end is None or W.loc[i,'start'] > active_end:
        g += 1
        active_end = W.loc[i,'response_end']
    else:
        active_end = max(active_end,W.loc[i,'response_end'])
    cluster[i] = g
W['overlap_cluster'] = cluster
W[['start','response_end','overlap_cluster']].to_csv(OUT / 'E22_overlap_cluster_membership.csv',index=False)

def model(kind, trees, subsample=True):
    kw = dict(n_estimators=trees,learning_rate=.03,num_leaves=7,min_child_samples=15,verbose=-1)
    if subsample:
        kw.update(subsample=.8,subsample_freq=1)
    return lgb.LGBMRegressor(**kw) if kind=='reg' else lgb.LGBMClassifier(**kw)

original_dml = pd.read_csv(SRC / 'E13_rain_dml.csv')
original_dml = original_dml[original_dml.window=='2018-10..2021-08'].set_index('outcome')
original_cf = pd.read_csv(SRC / 'E13_rain_counterfactual.csv')
original_cf = original_cf[(original_cf.analysis=='real')&(original_cf.window=='post 2018-10..2021-08')].set_index('outcome')
original_f04 = pd.read_csv(SRC / 'E15_f04_offset_corrected.csv').set_index('outcome').loc['f04_peak_c']

rows = []
for y,kind in [('f04_peak_c','reg'),('f06_peak','reg'),('cso','clf')]:
    Wfit=W.copy()
    if y!='f04_peak_c':
        for col in FX+[y]:
            Wfit[col]=W13[col].values
    pre = (Wfit.start >= '2015-06-01')&(Wfit.start < '2018-09-20')&Wfit[y].notna()
    post = (Wfit.start >= '2018-10-05')&Wfit[y].notna()
    sub = Wfit[pre|post].reset_index(drop=True)
    D = (sub.start >= '2018-10-05').astype(float).values
    Y = sub[y].astype(float).values
    X = sub[FX]
    ry,rd = np.zeros(len(sub)),np.zeros(len(sub))
    for rep in range(5):
        for a,b in KFold(5,shuffle=True,random_state=rep).split(X):
            # The corrected F04 nuisance fits did not use row subsampling in E15.
            my = model('reg',200,subsample=(y!='f04_peak_c')).fit(X.iloc[a],Y[a])
            md = model('clf',200,subsample=(y!='f04_peak_c')).fit(X.iloc[a],D[a])
            ry[b] += (Y[b]-my.predict(X.iloc[b]))/5
            rd[b] += (D[b]-md.predict_proba(X.iloc[b])[:,1])/5
    theta = (rd@ry)/(rd@rd)
    eps = ry-theta*rd
    original_se = np.sqrt(np.sum(rd**2*eps**2))/(rd@rd)
    if y=='f04_peak_c':
        d0,lo0,hi0 = original_f04.dml,original_f04.dml_lo,original_f04.dml_hi
    else:
        z = original_dml.loc[y]
        d0,lo0,hi0 = z.theta,z.lo,z.hi
    if not np.isclose(theta,d0,atol=1e-5,rtol=0):
        raise AssertionError(f'{y}: DML reproduction failed: {theta} != {d0}')
    se0 = (hi0-lo0)/3.92
    if not np.isclose(original_se,se0,atol=1e-5,rtol=0):
        raise AssertionError(f'{y}: DML SE reproduction failed: {original_se} != {se0}')
    score = pd.Series(rd*eps).groupby(sub.overlap_cluster.values).sum()
    G = len(score)
    finite_correction = G/(G-1) # one coefficient, no fitted intercept in the orthogonal score regression
    clustered_se = np.sqrt(finite_correction*np.sum(score.values**2))/(rd@rd)
    critical = student_t.ppf(.975,G-1)

    # Same fixed pre-period counterfactual model as E13/E15.
    fitted = model(kind,300).fit(Wfit.loc[pre,FX],Wfit.loc[pre,y])
    predicted = fitted.predict(Wfit.loc[post,FX]) if kind=='reg' else fitted.predict_proba(Wfit.loc[post,FX])[:,1]
    residual = Wfit.loc[post,y].values-predicted
    cf = residual.mean()
    if y=='f04_peak_c':
        c0,clo0,chi0 = original_f04.cf_effect,original_f04.cf_lo,original_f04.cf_hi
    else:
        z = original_cf.loc[y]
        c0,clo0,chi0 = z.effect,z.lo,z.hi
    if not np.isclose(cf,c0,atol=1e-5,rtol=0):
        raise AssertionError(f'{y}: counterfactual reproduction failed: {cf} != {c0}')
    labels = W.loc[post,'overlap_cluster'].values
    unique = np.unique(labels)
    indices = {c:np.flatnonzero(labels==c) for c in unique}
    rng = np.random.default_rng(20261022)
    boot = np.array([residual[np.concatenate([indices[c] for c in rng.choice(unique,len(unique),replace=True)])].mean() for _ in range(1000)])
    clo,chi = np.percentile(boot,[2.5,97.5])
    rows.append(dict(outcome=y,n_pre=int(pre.sum()),n_post=int(post.sum()),n_dml_clusters=G,n_post_clusters=len(unique),
                     dml=d0,dml_reproduced=theta,dml_reproduction_difference=theta-d0,dml_lo_original=lo0,dml_hi_original=hi0,
                     dml_se_original=se0,dml_se_cluster=clustered_se,dml_lo_cluster=d0-critical*clustered_se,dml_hi_cluster=d0+critical*clustered_se,
                     cluster_critical_value=critical,finite_cluster_correction=finite_correction,
                     cf=c0,cf_reproduced=cf,cf_reproduction_difference=cf-c0,cf_lo_original=clo0,cf_hi_original=chi0,
                     cf_lo_cluster=clo,cf_hi_cluster=chi,bootstrap_replicates=1000))
    print(y,json.dumps(rows[-1]),flush=True)

R = pd.DataFrame(rows)
R.to_csv(OUT / 'E22_overlap_uncertainty.csv',index=False)
metadata = dict(offset_m=float(offset),total_events=len(W),total_connected_clusters=int(W.overlap_cluster.nunique()),
                original_e13_feature_dtypes=original_dtypes,
                reproduction_absolute_tolerance=1e-5,max_dml_point_difference=float(R.dml_reproduction_difference.abs().max()),
                max_counterfactual_point_difference=float(R.cf_reproduction_difference.abs().max()),
                cluster_rule='Connected closed response intervals [start, start + dur_h - 5 min + 12 h]',
                dml_variance='Cluster-summed orthogonal scores; G/(G-1) finite correction; t critical with G-1 df',
                cf_bootstrap='Fixed original pre-period model; resample post-period overlap clusters with replacement; all within-cluster events retained',
                original_outputs_modified=False)
(OUT / 'E22_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print(R.to_string(index=False))
