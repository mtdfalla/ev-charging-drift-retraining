"""05 - Tables II and III and the paired tests, from the per-zone benchmark results.

Table II : mean MASE over zones at h = 24 and h = 1, with bootstrap 95% CIs over zones (h = 24).
Tests    : paired Wilcoxon over zones (online linear vs static ridge, GBT vs static ridge,
           persistence vs online linear) and the share of zones where persistence beats online.
Table III: day-ahead mean MASE by drift tercile (PSI cuts 0.13 and 0.28).
Run after 02 and 03 (agg): python code/05_tables.py
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import utils as U

target = U.PRIMARY_TARGET
res = pd.read_csv(os.path.join(U.RESULTS, f"benchmark_{target}.csv"))
res["zone"] = res["zone"].astype(str)
wide = {h: res[res.horizon == h].pivot(index="zone", columns="model", values="MASE") for h in (1, 24)}

ORDER = ["seasonal_naive_day", "persistence", "ewa_ensemble", "gbt_static", "online_linear",
         "mlp_static", "static_ridge", "rolling_mean_24h", "drift_triggered_mean"]
rng = np.random.default_rng(U.SEED)
w24 = wide[24]
idx = rng.integers(0, len(w24), size=(2000, len(w24)))
rows = []
for m in ORDER:
    v = w24[m].values
    boot = v[idx].mean(axis=1)
    rows.append({"model": m, "MASE_h24": v.mean(), "ci_low": np.quantile(boot, 0.025),
                 "ci_high": np.quantile(boot, 0.975), "MASE_h1": wide[1][m].mean()})
t2 = pd.DataFrame(rows)
t2.to_csv(os.path.join(U.RESULTS, f"table2_{target}.csv"), index=False)
print("Table II (mean MASE over zones)")
print(t2.round(3).to_string(index=False))

def paired(a, b):
    return wilcoxon(w24[a], w24[b]).pvalue

print("\nPaired Wilcoxon over zones, h = 24")
print("online linear vs ridge  p = %.3g" % paired("online_linear", "static_ridge"))
print("GBT vs ridge            p = %.3g" % paired("gbt_static", "static_ridge"))
print("persistence vs online   p = %.3g" % paired("persistence", "online_linear"))
print("persistence beats online in %.1f%% of zones" % (100 * (w24.persistence < w24.online_linear).mean()))
print("GBT beats persistence in %.1f%% of zones" % (100 * (w24.gbt_static < w24.persistence).mean()))

d = pd.read_csv(os.path.join(U.RESULTS, f"drift_summary_{target}.csv"))
d["zone"] = d["zone"].astype(str)
psi = d.set_index("zone")["psi_early_late"].reindex(w24.index)
q1, q2 = 0.13, 0.28  # PSI cuts of Table III (close to the terciles: 92, 92 and 91 zones)
band = pd.cut(psi, [-np.inf, q1, q2, np.inf], labels=["low", "medium", "high"])
t3 = w24[["seasonal_naive_day", "online_linear", "static_ridge", "gbt_static"]].groupby(band, observed=True).mean()
t3.to_csv(os.path.join(U.RESULTS, f"table3_{target}.csv"))
print("\nTable III (day-ahead mean MASE by PSI tercile, cuts %.2f and %.2f)" % (q1, q2))
print(t3.round(3).to_string())
