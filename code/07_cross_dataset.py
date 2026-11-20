"""07 - Cross-dataset comparison (UrbanEV vs Palo Alto), leakage-free."""
import os, pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import utils as U
R=os.path.join(U.ROOT,"experiments")
ue=pd.read_csv(os.path.join(R,"results","benchmark_volume_agg.csv"))
pa=pd.read_csv(os.path.join(R,"results_pa","benchmark_volume_agg.csv"))
ORDER=["persistence","seasonal_naive_day","rolling_mean_24h","ewa_ensemble",
       "drift_triggered_mean","online_linear","static_ridge","mlp_static","gbt_static"]
LAB=["persist","seas-naive","roll24","EWA","drift-trig","online-lin","ridge*","MLP*","GBT*"]
def col(df,h): return df[df.horizon==h].set_index("model").reindex(ORDER)["MASE"].values
fig,ax=plt.subplots(1,2,figsize=(10,3.8))
# colour: naive anchors grey, adaptive blue, static learned red
colors=["#888"]*3+["#1f77b4"]*3+["#d62728"]*3
for a,(name,df) in zip(ax,[("UrbanEV (Shenzhen; higher drift)",ue),("Palo Alto (lower drift)",pa)]):
    v=col(df,24)
    a.bar(range(len(ORDER)),v,color=colors); a.axhline(np.nanmin(v),color="k",lw=.7,ls="--")
    a.set_xticks(range(len(ORDER))); a.set_xticklabels(LAB,rotation=40,ha="right",fontsize=8)
    a.set_title(name,fontsize=9); a.set_ylabel("MASE (day-ahead)")
fig.suptitle("Seasonal-naive is the strongest day-ahead baseline on both cities; learned and adaptive models do not beat it\n(grey: naive anchors,  blue: adaptive,  red: static learned;  dashed: best per panel)",fontsize=8.5)
fig.tight_layout(); p=os.path.join(R,"figures","cross_dataset_benchmark_v2.png")
fig.savefig(p,dpi=600); plt.close(fig)
t=pd.DataFrame({"model":ORDER,"UrbanEV_h24":col(ue,24).round(3),"PaloAlto_h24":col(pa,24).round(3)})
t.to_csv(os.path.join(R,"results","cross_dataset_h24_v2.csv"),index=False)
print(t.to_string(index=False)); print("wrote cross_dataset_benchmark_v2.png")
