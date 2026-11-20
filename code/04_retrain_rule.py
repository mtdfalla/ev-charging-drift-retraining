"""04 - RQ3: When to retrain? A cost-accuracy decision rule.

A windowed Ridge forecaster (lag+calendar features, trailing-window refit) is run
walk-forward over the test period under four retraining policies:
  never            : fit once at test start, never refit (degrades under drift)
  periodic_weekly  : refit every 168 h
  periodic_daily   : refit every 24 h
  drift_triggered  : refit only when Page-Hinkley flags drift in the residual stream
  always           : refit every step (upper bound on cost/accuracy)
Cost = number of refits. Accuracy = day-ahead MASE (h = 24). We also link each zone's
drift-triggered refit count to its PSI (RQ1 -> RQ3).

Outputs: experiments/results/retrain_<target>.csv (+ _agg.csv) and Fig 4.
Modes: chunk START END NTOTAL | agg | (default) quick test.
Run: python code/04_retrain_rule.py chunk 0 20 60
"""
import os, sys, time, warnings, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import utils as U
warnings.filterwarnings("ignore")

W = 672           # trailing window for refit (4 weeks)
H = 24            # day-ahead horizon (operational planning; where adaptivity pays)
POLICIES = ["never", "periodic_weekly", "periodic_daily", "drift_triggered", "always"]
PARTIAL = os.path.join(U.RESULTS, "_retrain_partial.csv")
MANU_FIG = os.path.join(U.ROOT, "manuscript", "figures")
if os.environ.get("MANU_COPY"): os.makedirs(MANU_FIG, exist_ok=True)


def make_features(s):
    T = len(s); idx = np.arange(T)
    lag1 = np.full(T, np.nan); lag1[1:] = s[:-1]
    lag24 = np.full(T, np.nan); lag24[24:] = s[:-24]
    lag168 = np.full(T, np.nan); lag168[168:] = s[:-168]
    roll24 = pd.Series(s).rolling(24, min_periods=1).mean().shift(1).values
    hod = idx % 24; dow = (idx // 24) % 7
    return np.column_stack([lag1, lag24, lag168, roll24,
        np.sin(2*np.pi*hod/24), np.cos(2*np.pi*hod/24),
        np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7)])


def mase_denom(train, h=H):
    d = np.abs(np.diff(train)) if h < 24 else np.abs(train[24:]-train[:-24])
    return float(np.mean(d)) + 1e-9


def fit_window(X, s, t_end):
    """Fit Ridge on trailing window ending at t_end; return fast-predict params."""
    lo = max(168, t_end - W)
    rows = [t for t in range(lo, t_end) if t < len(s)-H]
    Xtr = X[rows]; ytr = s[np.array(rows)+H]
    sc = StandardScaler().fit(Xtr)
    mdl = Ridge(alpha=1.0).fit(sc.transform(Xtr), ytr)
    return (mdl.coef_, float(mdl.intercept_), sc.mean_, sc.scale_)


def run_policy(s, X, test_t, policy):
    """Leakage-free walk forward. Predictions for s[t+H] use a model fit only on
    windows whose targets were observed by t (window ends at t-H+1). The detector
    consumes DELAYED errors: at origin t the target s[t] reveals the error of the
    forecast made at t-H. Returns (mean abs error, n_refits)."""
    def predict(par, t):
        coef,intercept,cmean,cscale = par
        return max(float(((X[t]-cmean)/cscale)@coef+intercept), 0.0)
    par = fit_window(X, s, test_t[0]-H+1)   # initial fit, observed targets only
    refits = 1
    mean=0.0; n=0; mT=0.0; MT=0.0
    res_scale = np.std(s[max(0,test_t[0]-W):test_t[0]]) + 1e-9
    lam = 3*res_scale
    preds={}; errs=np.empty(len(test_t))
    for i, t in enumerate(test_t):
        preds[t]=predict(par, t)            # forecast for s[t+H]
        errs[i]=abs(s[t+H]-preds[t])        # evaluation only (not used for decisions)
        drift=False
        if (t-H) in preds:                  # delayed error: target s[t] now observed
            de=abs(s[t]-preds[t-H]); n+=1; mean+=(de-mean)/n
            mT+=(de-mean-0.01*res_scale); MT=min(MT,mT); drift=(mT-MT)>lam
        do = (policy=="always") or (policy=="periodic_daily" and (i+1)%24==0)              or (policy=="periodic_weekly" and (i+1)%168==0) or (policy=="drift_triggered" and drift)
        if do:
            par = fit_window(X, s, t-H+1); refits+=1   # refit on observed targets only
            if drift: mean=0.0; n=0; mT=0.0; MT=0.0
    return float(np.mean(errs)), refits, None


def run_chunk(start, end, ntotal, target=U.PRIMARY_TARGET):
    t0=time.time()
    df = U.load_target(target).ffill().bfill().clip(lower=0)
    n=len(df); tr,va,te = U.temporal_split(n)
    # drift PSI per zone for RQ1->RQ3 link
    psi_map={}
    fp=os.path.join(U.RESULTS,f"drift_summary_{target}.csv")
    if os.path.exists(fp):
        d=pd.read_csv(fp); psi_map={str(z):p for z,p in zip(d.zone,d.psi_early_late)}
    # stratified zone order (reuse 03's logic)
    if os.path.exists(fp):
        d2=pd.read_csv(fp).dropna(subset=["psi_early_late"]).sort_values("psi_early_late")
        zl=[str(z) for z in d2.zone.tolist()]
        zones=[zl[int(i)] for i in np.linspace(0,len(zl)-1,ntotal)][start:end]
    else:
        zones=list(df.columns)[start:end]
    recs=[]
    for z in zones:
        s=df[z].values.astype(float); X=make_features(s)
        test_t=[t for t in range(te.start, n-H) if t>=168]
        denom=mase_denom(s[tr])
        for pol in POLICIES:
            mae,refits,_=run_policy(s,X,test_t,pol)
            recs.append({"zone":z,"policy":pol,"MAE":mae,"MASE":mae/denom,
                         "refits":refits,"psi":psi_map.get(z,np.nan)})
    out=pd.DataFrame(recs); hdr=(not os.path.exists(PARTIAL)) or os.path.getsize(PARTIAL)==0
    out.to_csv(PARTIAL, mode="a", header=hdr, index=False)
    print(f"[04] chunk [{start}:{end}] {len(zones)} zones done ({time.time()-t0:.0f}s)")


def aggregate(target=U.PRIMARY_TARGET):
    res=pd.read_csv(PARTIAL).drop_duplicates(["zone","policy"])
    res.to_csv(os.path.join(U.RESULTS,f"retrain_{target}.csv"), index=False)
    nz=res.zone.nunique()
    agg=res.groupby("policy").agg(MASE=("MASE","mean"), refits=("refits","mean")).reset_index()
    order={p:i for i,p in enumerate(POLICIES)}; agg=agg.sort_values("policy",key=lambda c:c.map(order))
    agg.to_csv(os.path.join(U.RESULTS,f"retrain_{target}_agg.csv"), index=False)
    # accuracy retained vs always, cost vs always
    never=agg.set_index("policy").loc["never","MASE"]; always=agg.set_index("policy").loc["always","MASE"]
    amax=agg.set_index("policy").loc["always","refits"]
    agg["acc_retained_%"]=100*(never-agg["MASE"])/(never-always+1e-9)
    agg["cost_vs_always_%"]=100*agg["refits"]/amax
    print(f"[04] aggregated over {nz} zones:"); print(agg.round(2).to_string(index=False))
    # RQ1->RQ3 link: correlation of drift-triggered refits with zone PSI
    dt=res[res.policy=="drift_triggered"].dropna(subset=["psi"])
    corr=np.corrcoef(dt["psi"], dt["refits"])[0,1] if len(dt)>2 else np.nan
    print(f"[04] corr(drift-triggered refits, zone PSI) = {corr:.3f}")
    with open(os.path.join(U.RESULTS,f"retrain_{target}_headline.txt"),"w") as f:
        f.write(agg.round(3).to_string(index=False)+f"\ncorr(refits,PSI)={corr:.3f}\n")

    # Fig 4: (left) cost-accuracy frontier; (right) accuracy retained vs cost
    a=agg.set_index("policy")
    colmap={"never":"#888","periodic_weekly":"#2ca02c","periodic_daily":"#9467bd",
            "drift_triggered":"#d62728","always":"#1f77b4"}
    fig, ax = plt.subplots(1,2, figsize=(9,3.4))
    for pol in POLICIES:
        ax[0].scatter(a.loc[pol,"refits"], a.loc[pol,"MASE"], s=80, color=colmap[pol], label=pol, zorder=3)
    ax[0].set_xscale("log")
    ax[0].axhline(a.loc["never","MASE"], color="#888", lw=0.7, ls=":")
    ax[0].axhline(a.loc["always","MASE"], color="#1f77b4", lw=0.7, ls=":")
    ax[0].set_xlabel("mean # refits over test (log; cost)"); ax[0].set_ylabel("mean MASE (day-ahead)")
    ax[0].set_title("Cost-accuracy frontier"); ax[0].legend(fontsize=7)
    for pol in POLICIES:
        ax[1].scatter(a.loc[pol,"cost_vs_always_%"], a.loc[pol,"acc_retained_%"],
                      s=80, color=colmap[pol], label=pol, zorder=3)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("retraining cost (% of always; log)"); ax[1].set_ylabel("accuracy gain retained (%)")
    ax[1].set_title("Drift-triggered: most of the gain, little cost")
    fig.tight_layout(); p=os.path.join(U.FIGURES,f"retrain_costacc_{target}.png")
    fig.savefig(p,dpi=600); plt.close(fig)
    (shutil.copy(p, os.path.join(MANU_FIG,"retrain_costacc.png")) if os.environ.get("MANU_COPY") else None)
    print(f"[04] wrote retrain_{target}.csv + agg + figure ({nz} zones)")


if __name__=="__main__":
    mode=sys.argv[1] if len(sys.argv)>1 else "test"
    if mode=="chunk": run_chunk(int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]))
    elif mode=="agg": aggregate()
    elif mode=="reset":
        if os.path.exists(PARTIAL): os.remove(PARTIAL)
        print("reset")
    else:
        run_chunk(0,4,60); print("(quick test done; use chunk/agg for full run)")
