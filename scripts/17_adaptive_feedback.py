"""Recompute fixed-model scores and defer calibration feedback until observed.

No hyperparameter, cohort, label, or warning rule changes. Originals stay intact.
"""
from pathlib import Path
import sys,os,json
import numpy as np,pandas as pd
from scipy.stats import norm
ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=ROOT
sys.path.insert(0,str(ORIGINAL/'scripts'))
import wlib as W
OUT=ROOT/'results_corrected';OUT.mkdir(exist_ok=True)

def wquantile(x,w,q):
    order=np.argsort(x);x,w=x[order],w[order];cw=np.cumsum(w)/w.sum()
    return x[min(np.searchsorted(cw,q),len(x)-1)]

def adaptive(L,mu,alpha):
    cso=L.delta.eq(1).to_numpy();cal=L.split.eq('val').to_numpy()&cso;te=L.split.eq('test').to_numpy()&cso
    logT=np.log(np.maximum(L['T'].to_numpy(),1.0));hist=list(mu[cal]-logT[cal]);at=alpha
    order=L[te].groupby('eid').t.min().sort_values()
    bounds=np.full(len(L),np.nan);pending=[];deferred=[];feedback=[]
    for eid,now in order.items():
        ready=sorted([p for p in pending if p['completed']<=now],key=lambda p:(p['completed'],p['eid']))
        for old in ready:
            at+=.02*(alpha-old['error']);hist.extend(old['scores']);pending.remove(old)
            feedback.append({'episode':int(old['eid']),'completion':str(old['completed']),'used_from':str(now),'available':bool(old['completed']<=now)})
        if pending:deferred.append({'episode':int(eid),'time':str(now),'pending_episodes':[int(x['eid']) for x in pending]})
        idx=te&L.eid.eq(eid).to_numpy();q=np.quantile(hist,np.clip(1-at,0,1));bounds[idx]=np.exp(mu[idx]-q)
        covered=L['T'].to_numpy()[idx]>=bounds[idx]
        completion=(L.loc[idx,'t']+pd.to_timedelta(L.loc[idx,'T'],unit='min')).iloc[0]
        pending.append({'eid':eid,'completed':completion,'error':1-covered.mean(),'scores':list(mu[idx]-logT[idx])})
    for old in sorted(pending,key=lambda p:(p['completed'],p['eid'])):at+=.02*(alpha-old['error'])
    cov=L['T'].to_numpy()[te]>=bounds[te]
    row={'coverage':float(cov.mean()),'median_L':float(np.median(bounds[te])),'final_alpha':float(at)}
    return bounds,row,{'deferred':deferred,'feedback':feedback,'all_feedback_available':all(x['available'] for x in feedback)}

def main():
    rows=[];alarm_rows=[];audit={};cache={}
    for design,L,feats in [('B rain-onset',W.design_B(4.5),W.SC_B+W.F04_B+W.RF),('A storage-onset',W.design_A(),W.SC_A+W.F04_A+W.RF)]:
        print('Fixed-model score reconstruction:',design,flush=True)
        mus={'Physics TTF':np.log(L.ttf_phys.to_numpy())}
        p=None
        saved=OUT/('warning_scores_'+design[0]+'.parquet')
        previous=pd.read_parquet(saved) if saved.exists() else None
        for model in ['XGB-AFT','Phys-AFT']:
            if previous is not None:
                assert np.array_equal(previous.t.to_numpy(),L.t.to_numpy())
                mus[model]=previous['mu_'+model].to_numpy()
                if model=='Phys-AFT':p=previous.p120.to_numpy()
                continue
            scores=[]
            for seed in range(3):
                scores.append(W.predict(L,feats,120,model,seed=seed,return_mu=True))
                print(design,model,'seed',seed,'complete',flush=True)
            mus[model]=np.mean(scores,axis=0)
            if model=='Phys-AFT':p=np.mean([norm.cdf((np.log(120)-s)/1.2) for s in scores],axis=0)
        event=pd.DataFrame({'eid':L.eid,'split':L.split,'cso':L.event_cso,'p':p}).groupby('eid').agg(split=('split','first'),cso=('cso','max'),pmax=('p','max'))
        tau=np.quantile(event[(event.split=='val')&(event.cso==0)].pmax,.95)
        cso=L.delta.eq(1).to_numpy();cal=L.split.eq('val').to_numpy()&cso;te=L.split.eq('test').to_numpy()&cso
        logT=np.log(np.maximum(L['T'].to_numpy(),1));nev=L[cal].groupby('eid').eid.transform('size').to_numpy()
        cached_bounds={}
        for name,mu in mus.items():
            scorecal=mu[cal]-logT[cal]
            for alpha in [.1,.2]:
                for variant,q in [('split',np.quantile(scorecal,np.ceil((len(scorecal)+1)*(1-alpha))/len(scorecal),method='higher')),('event-balanced',wquantile(scorecal,1/nev,1-alpha))]:
                    bound=np.exp(mu[te]-q);cov=L['T'].to_numpy()[te]>=bound;evcov=pd.Series(cov).groupby(L.eid.to_numpy()[te]).mean()
                    rows.append(dict(design=design,model=name,alpha=alpha,variant=variant,q=q,coverage=cov.mean(),event_mean_cov=evcov.mean(),events_all_covered=(evcov==1).mean(),median_L=np.median(bound),median_T=np.median(L['T'].to_numpy()[te]),L_over_T=np.median(bound/L['T'].to_numpy()[te]),n_cal=int(cal.sum()),n_test=int(te.sum())))
                bounds,result,trace=adaptive(L,mu,alpha)
                rows.append(dict(design=design,model=name,alpha=alpha,variant='ACI',**result))
                audit[design+'|'+name+'|'+str(alpha)]=trace
                cached_bounds[(name,alpha)]=bounds
                if name in ['Physics TTF','Phys-AFT']:
                    first=pd.DataFrame({'eid':L.eid,'T':L['T'],'L':bounds,'flag':te&(p>tau)});first=first[first.flag].groupby('eid').first()
                    alarm_rows.append(dict(design=design,bound_model=name,alpha=alpha,n_alarmed_cso=len(first),coverage_at_alarm=(first['T']>=first['L']).mean(),median_bound_min=first['L'].median(),median_actual_lead_min=first['T'].median(),bound_ge_30min=(first['L']>=30).mean()))
        # Preserve the original event-balanced first-alarm diagnostic rows.
        for alpha in [.1,.2]:
            mu=mus['Phys-AFT'];q=wquantile(mu[cal]-logT[cal],1/nev,1-alpha)
            f=pd.DataFrame({'eid':L.eid,'T':L['T'],'L':np.exp(mu-q),'flag':te&(p>tau)});f=f[f.flag].groupby('eid').first()
            rows.append(dict(design=design,model='Phys-AFT at first alarm',alpha=alpha,variant='event-balanced',q=q,coverage=(f['T']>=f['L']).mean(),median_L=f['L'].median(),median_T=f['T'].median(),n_test=len(f)))
        if design.startswith('B'):
            tc=pd.read_parquet(ORIGINAL/'results/figdata/fig6_designB_timecourse.parquet')
            assert len(tc)==len(L) and np.array_equal(tc.eid.to_numpy(),L.eid.to_numpy()) and np.array_equal(tc.t.to_numpy(),L.t.to_numpy())
            maxerr=float(np.max(np.abs(tc.p120.to_numpy()-p)))
            assert maxerr<1e-5,maxerr
            tc['L_phys_aci10']=cached_bounds[('Physics TTF',.1)];tc['L_aft_aci10']=cached_bounds[('Phys-AFT',.1)]
            tc.to_parquet(OUT/'fig6_designB_timecourse.parquet',index=False)
            cache['probability_max_abs_difference']=maxerr
        score_cache=L[[c for c in ['eid','t','start','split','T','delta','event_cso','ttf_phys'] if c in L]].copy()
        for name,mu in mus.items():score_cache['mu_'+name.replace(' ','_')]=mu
        score_cache['p120']=p;score_cache['tau']=tau
        score_cache.to_parquet(OUT/('warning_scores_'+design[0]+'.parquet'),index=False)
    
    R=pd.DataFrame(rows);AR=pd.DataFrame(alarm_rows)
    R.to_csv(OUT/'E17_conformal_leadtime.csv',index=False);AR.to_csv(OUT/'E17b_alarm_physics_bound.csv',index=False)
    audit['summary']={'all_feedback_available':all(v['all_feedback_available'] for v in audit.values()),**cache}
    (OUT/'adaptive_feedback_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(R[R.variant.eq('ACI')].to_string(index=False))
    print(AR.to_string(index=False))

if __name__=="__main__": main()
