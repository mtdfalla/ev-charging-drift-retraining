"""03 - RQ2: Forecasting benchmark (LEAKAGE-FREE), full-dataset capable.

Adaptive models use only information available in real time: a forecast made at
origin t for target t+h is scored and used for learning only AFTER y_{t+h} is
observed, i.e. updates at step t use the example (x_{t-h}, y_t). This removes the
h-step lookahead present in naive prequential updating.

Model ladder:
  anchors (leakage-free): persistence, seasonal_naive_day, rolling_mean_24h
  adaptive (h-delayed):   ewa_ensemble, drift_triggered_mean, online_linear
  static global learned:  static_ridge, mlp_static, gbt_static
Headline metric MASE (vs in-sample naive/seasonal-naive). Horizons 1 and 24.
Modes: chunk START END NTOTAL | global | agg | reset
"""
import os, sys, time, warnings, shutil
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import utils as U
warnings.filterwarnings("ignore")

HORIZONS = (1, 24)
ORDER = ["persistence","seasonal_naive_day","rolling_mean_24h","ewa_ensemble",
         "drift_triggered_mean","online_linear","static_ridge","mlp_static","gbt_static"]
PARTIAL = os.path.join(U.RESULTS, "_benchmark_partial.csv")
MANU_FIG = os.path.join(U.ROOT, "manuscript", "figures")
if os.environ.get("MANU_COPY"): os.makedirs(MANU_FIG, exist_ok=True)


def make_features(s):
    T=len(s); idx=np.arange(T)
    lag1=np.full(T,np.nan); lag1[1:]=s[:-1]
    lag24=np.full(T,np.nan); lag24[24:]=s[:-24]
    lag168=np.full(T,np.nan); lag168[168:]=s[:-168]
    roll24=pd.Series(s).rolling(24,min_periods=1).mean().shift(1).values
    hod=idx%24; dow=(idx//24)%7
    return np.column_stack([lag1,lag24,lag168,roll24,
        np.sin(2*np.pi*hod/24),np.cos(2*np.pi*hod/24),
        np.sin(2*np.pi*dow/7),np.cos(2*np.pi*dow/7)])


def mase_denom(train,h):
    d=np.abs(train[24:]-train[:-24]) if h>=24 else np.abs(np.diff(train))
    return float(np.mean(d))+1e-9


def stratified_order(target,n_zones):
    fp=os.path.join(U.RESULTS,f"drift_summary_{target}.csv")
    df=U.load_target(target)
    if os.path.exists(fp):
        d=pd.read_csv(fp).dropna(subset=["psi_early_late"]).sort_values("psi_early_late")
        zl=[str(z) for z in d.zone.tolist()]
        return [zl[int(i)] for i in np.linspace(0,len(zl)-1,n_zones)]
    return list(df.columns)[:n_zones]


CANON=["model","horizon","MAE","RMSE","sMAPE","MASE","zone"]
def append(rows):
    out=pd.DataFrame(rows).reindex(columns=CANON)
    hdr=(not os.path.exists(PARTIAL)) or os.path.getsize(PARTIAL)==0
    out.to_csv(PARTIAL,mode="a",header=hdr,index=False)


def cheap_online(s,h,tr,va,te):
    """All adaptive updates are h-delayed (leakage-free)."""
    T=len(s); X=make_features(s)
    test_t=[t for t in range(te.start,T-h) if t>=168]
    truth=np.array([s[t+h] for t in test_t])
    # deterministic anchors indexed by origin (functions of past only)
    pers=lambda t: s[t]
    seas=lambda t: s[t-24+h] if (t-24+h)>=0 else s[t]
    roll=lambda t: np.mean(s[max(0,t-23):t+1])
    P={}
    P["persistence"]=np.array([pers(t) for t in test_t])
    P["seasonal_naive_day"]=np.array([seas(t) for t in test_t])
    P["rolling_mean_24h"]=np.array([roll(t) for t in test_t])

    # EWA over anchors, weights updated with h-delay (using observed target s[t])
    w=np.ones(3)/3; eta=0.5; ewa=np.zeros(len(test_t))
    for i,t in enumerate(test_t):
        o=t-h  # origin whose target s[t] is now observed
        if o>=168:
            bp=np.array([pers(o),seas(o),roll(o)]); err=(bp-s[t])**2
            w=w*np.exp(-eta*err); w=np.maximum(w,1e-6); w/=w.sum()
        ewa[i]=np.array([pers(t),seas(t),roll(t)])@w
    P["ewa_ensemble"]=ewa

    # drift-triggered rolling mean: uses only observed s[<=t] -> leakage-free
    dt=np.zeros(len(test_t)); win=[]; mean=0.0;n=0;mT=0.0;MT=0.0; lam=50*(np.std(s)+1e-9)
    for i,t in enumerate(test_t):
        x=s[t]; n+=1; mean+=(x-mean)/n; mT+=(x-mean-0.005); MT=min(MT,mT)
        if (mT-MT)>lam: win=[];mean=0.0;n=0;mT=0.0;MT=0.0
        win.append(x); win=win[-168:]; dt[i]=np.mean(win)
    P["drift_triggered_mean"]=dt

    # online linear: warm-start ridge on train; h-delayed online SGD during test
    tr_rows=[t for t in range(tr.start,tr.stop-h) if t>=168]
    Xtr=X[tr_rows]; ytr=np.array([s[t+h] for t in tr_rows])
    xs=StandardScaler().fit(Xtr); ymu,ysd=float(ytr.mean()),float(ytr.std()+1e-9)
    Xtr_s=np.nan_to_num(xs.transform(Xtr)); ytr_s=(ytr-ymu)/ysd
    d=Xtr_s.shape[1]; A=Xtr_s.T@Xtr_s+1e-2*np.eye(d); w=np.linalg.solve(A,Xtr_s.T@ytr_s); lr=0.02
    olp=np.empty(len(test_t))
    for i,t in enumerate(test_t):
        o=t-h
        if o>=168:  # learn from the just-observed (x_{t-h}, y_t)
            xu=np.nan_to_num(xs.transform(X[o:o+1]))[0]; g=(xu@w-(s[t]-ymu)/ysd); w=w-lr*g*xu
        xt=np.nan_to_num(xs.transform(X[t:t+1]))[0]; olp[i]=(xt@w)*ysd+ymu
    P["online_linear"]=np.clip(olp,0,None)

    denom=mase_denom(s[tr],h)
    return [{"model":m,"horizon":h,"MAE":U.mae(truth,yh),"RMSE":U.rmse(truth,yh),
             "sMAPE":U.smape(truth,yh),"MASE":U.mae(truth,yh)/denom} for m,yh in P.items()]


def run_chunk(start,end,ntotal,target=U.PRIMARY_TARGET):
    t0=time.time(); df=U.load_target(target).ffill().bfill().clip(lower=0)
    n=len(df); tr,va,te=U.temporal_split(n)
    zones=stratified_order(target,ntotal)[start:end]; rows=[]
    for z in zones:
        s=df[z].values.astype(float)
        for h in HORIZONS:
            for r in cheap_online(s,h,tr,va,te): r["zone"]=z; rows.append(r)
    append(rows); print(f"[03] chunk [{start}:{end}] {len(zones)} zones ({time.time()-t0:.0f}s)")


def run_global(ntotal,target=U.PRIMARY_TARGET):
    """Global STATIC learners (no test-time updates -> leakage-free): ridge, MLP, GBT."""
    t0=time.time(); df=U.load_target(target).ffill().bfill().clip(lower=0)
    n=len(df); tr,va,te=U.temporal_split(n); zones=stratified_order(target,ntotal)
    zmeta={}
    for h in HORIZONS:
        Xtr_all=[]; ytr_all=[]
        for z in zones:
            s=df[z].values.astype(float); mu,sd=s[tr].mean(),s[tr].std()+1e-9
            sn=(s-mu)/sd; X=make_features(sn)
            rows=[t for t in range(tr.start,tr.stop-h) if t>=168]
            Xtr_all.append(X[rows]); ytr_all.append(np.array([sn[t+h] for t in rows])); zmeta[z]=(mu,sd)
        Xtr=np.vstack(Xtr_all); ytr=np.concatenate(ytr_all)
        if len(Xtr)>60000:
            idx=np.random.default_rng(U.SEED).choice(len(Xtr),60000,replace=False); Xtr=Xtr[idx]; ytr=ytr[idx]
        xs=StandardScaler().fit(Xtr); Xtr_s=xs.transform(Xtr)
        ridge=Ridge(alpha=1.0).fit(Xtr_s,ytr)
        mlp=MLPRegressor(hidden_layer_sizes=(64,32),max_iter=120,early_stopping=True,random_state=U.SEED).fit(Xtr_s,ytr)
        gbt=HistGradientBoostingRegressor(max_depth=6,max_iter=250,learning_rate=0.05,random_state=U.SEED).fit(Xtr_s,ytr)
        rows_out=[]
        for z in zones:
            s=df[z].values.astype(float); mu,sd=zmeta[z]; sn=(s-mu)/sd; X=make_features(sn)
            test_t=[t for t in range(te.start,n-h) if t>=168]
            truth=np.array([s[t+h] for t in test_t]); denom=mase_denom(s[tr],h)
            for name,mdl in [("static_ridge",ridge),("mlp_static",mlp),("gbt_static",gbt)]:
                yh=np.clip(mdl.predict(xs.transform(X[test_t]))*sd+mu,0,None)
                rows_out.append({"model":name,"horizon":h,"zone":z,
                    "MAE":U.mae(truth,yh),"RMSE":U.rmse(truth,yh),
                    "sMAPE":U.smape(truth,yh),"MASE":U.mae(truth,yh)/denom})
        append(rows_out); print(f"[03] global static h={h}: {len(zones)} zones ({time.time()-t0:.0f}s)")


def aggregate(target=U.PRIMARY_TARGET):
    res=pd.read_csv(PARTIAL).drop_duplicates(["zone","horizon","model"])
    res.to_csv(os.path.join(U.RESULTS,f"benchmark_{target}.csv"),index=False)
    nz=res.zone.nunique()
    agg=res.groupby(["model","horizon"]).agg(MASE=("MASE","mean"),sMAPE=("sMAPE","mean"),
        MAE=("MAE","mean"),RMSE=("RMSE","mean")).reset_index()
    agg.to_csv(os.path.join(U.RESULTS,f"benchmark_{target}_agg.csv"),index=False)
    print(f"[03] aggregated over {nz} zones:"); print(agg.round(3).to_string(index=False))


if __name__=="__main__":
    m=sys.argv[1] if len(sys.argv)>1 else "agg"
    if m=="chunk": run_chunk(int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]))
    elif m=="global": run_global(int(sys.argv[2]) if len(sys.argv)>2 else 275)
    elif m=="agg": aggregate()
    elif m=="reset": open(PARTIAL,"w").close(); print("reset")
