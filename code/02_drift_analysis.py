"""02 - RQ1: Characterize non-stationarity (drift) across the 275 UrbanEV zones.

Per zone: distributional shift (PSI, KS, Wasserstein, mean-shift), rolling-mean CoV,
STL trend/seasonal strength, Page-Hinkley change-point count, and a drift-type label.
Merges zone centroids/attributes; writes Table II source + paper figures.
Run: python code/02_drift_analysis.py
"""
import os, shutil, json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import utils as U

MANU_FIG = os.path.join(U.ROOT, "manuscript", "figures")
if os.environ.get("MANU_COPY"): os.makedirs(MANU_FIG, exist_ok=True)


def psi(expected, actual, bins=10):
    qs = np.quantile(expected, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    e = np.histogram(expected, qs)[0] / max(len(expected), 1) + 1e-6
    a = np.histogram(actual, qs)[0] / max(len(actual), 1) + 1e-6
    return float(np.sum((a - e) * np.log(a / e)))


def page_hinkley(x, delta=0.005, lam_factor=50.0):
    x = np.asarray(x, float)
    lam = lam_factor * (np.std(x) + 1e-9)
    mean = 0.0; n = 0; mT = 0.0; MT = 0.0; changes = 0
    for v in x:
        n += 1
        mean += (v - mean) / n
        mT += (v - mean - delta)
        MT = min(MT, mT)
        if (mT - MT) > lam:
            changes += 1
            mean = 0.0; n = 0; mT = 0.0; MT = 0.0
    return changes


def stl_strengths(x, period=U.SEASON_DAY):
    try:
        from statsmodels.tsa.seasonal import STL
        res = STL(pd.Series(x), period=period, robust=False).fit()
        r, t, s = res.resid.values, res.trend.values, res.seasonal.values
        var_r = np.var(r)
        Ft = max(0.0, 1 - var_r / (np.var(t + r) + 1e-9))
        Fs = max(0.0, 1 - var_r / (np.var(s + r) + 1e-9))
        return float(Ft), float(Fs)
    except Exception:
        return float("nan"), float("nan")


def classify(row):
    if row["psi_early_late"] < 0.1 and abs(row["mean_shift_ratio"]) < 0.1:
        return "near_stationary"
    if row.get("ph_changes", 0) >= 3:
        return "sudden"
    if abs(row["mean_shift_ratio"]) >= 0.3:
        return "level_shift"
    if row.get("seasonal_strength", 0) >= 0.6 and row["psi_early_late"] < 0.25:
        return "seasonal"
    return "gradual"


def zone_centroids():
    fp = os.path.join(U.RAW, "inf.csv")
    if not os.path.exists(fp):
        return None
    inf = pd.read_csv(fp)
    g = inf.groupby("TAZID").agg(
        lon=("longitude", "mean"), lat=("latitude", "mean"),
        charge=("charge_count", "sum"), area=("area", "first")).reset_index()
    g["TAZID"] = g["TAZID"].astype(int).astype(str)
    return g.set_index("TAZID")


def main(target=U.PRIMARY_TARGET, max_zones=None, do_stl=True):
    df = U.load_target(target).ffill().bfill().clip(lower=0)
    zones = list(df.columns)[:max_zones] if max_zones else list(df.columns)
    n = len(df); half = n // 2
    rows = []
    for z in zones:
        s = df[z].values
        early, late = s[:half], s[half:]
        roll = pd.Series(s).rolling(U.SEASON_WEEK, min_periods=U.SEASON_WEEK).mean().dropna()
        rec = {
            "zone": z, "mean": float(s.mean()),
            "psi_early_late": psi(early, late),
            "ks_stat": float(ks_2samp(early, late).statistic),
            "wasserstein": float(wasserstein_distance(early, late)),
            "mean_shift_ratio": float((late.mean() - early.mean()) / (early.mean() + 1e-9)),
            "rolling_cov": float(roll.std() / (roll.mean() + 1e-9)) if len(roll) else np.nan,
            "ph_changes": page_hinkley(s),
        }
        if do_stl:
            Ft, Fs = stl_strengths(s)
            rec["trend_strength"], rec["seasonal_strength"] = Ft, Fs
        rows.append(rec)
    summ = pd.DataFrame(rows)
    summ["drift_type"] = summ.apply(classify, axis=1)

    cz = zone_centroids()
    if cz is not None:
        summ = summ.merge(cz, left_on="zone", right_index=True, how="left")
    else:
        summ["lon"] = np.nan; summ["lat"] = np.nan

    out = os.path.join(U.RESULTS, "drift_summary_%s.csv" % target)
    summ.to_csv(out, index=False)
    print("[02] wrote %s (%d zones)" % (os.path.basename(out), len(summ)))
    print("[02] drift-type counts:\n", summ["drift_type"].value_counts().to_string())
    desc_cols = ["psi_early_late", "ks_stat", "mean_shift_ratio", "ph_changes"]
    if do_stl: desc_cols += ["trend_strength", "seasonal_strength"]
    print(summ[desc_cols].describe().round(3).to_string())

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.0))
    ax[0].hist(summ["psi_early_late"], bins=30); ax[0].set_title("PSI (early vs late)"); ax[0].set_xlabel("PSI")
    ax[1].hist(summ["mean_shift_ratio"], bins=30); ax[1].set_title("Mean shift ratio"); ax[1].set_xlabel("(late-early)/early")
    if do_stl:
        ax[2].scatter(summ["seasonal_strength"], summ["trend_strength"], s=10, alpha=0.6)
        ax[2].set_xlabel("seasonal strength"); ax[2].set_ylabel("trend strength"); ax[2].set_title("STL strengths")
    fig.tight_layout(); p = os.path.join(U.FIGURES, "drift_hist_%s.png" % target); fig.savefig(p, dpi=600); plt.close(fig)
    print("[02] wrote", os.path.basename(p))

    if summ["lon"].notna().any():
        fig, axm = plt.subplots(figsize=(5.2, 4.4))
        sc = axm.scatter(summ["lon"], summ["lat"], c=summ["psi_early_late"], cmap="viridis",
                         s=22, vmin=0, vmax=np.nanpercentile(summ["psi_early_late"], 95))
        axm.set_xlabel("longitude"); axm.set_ylabel("latitude")
        axm.set_title("Distributional drift (PSI) by Shenzhen zone")
        fig.colorbar(sc, label="PSI (early vs late)"); fig.tight_layout()
        p = os.path.join(U.FIGURES, "drift_map_%s.png" % target); fig.savefig(p, dpi=600); plt.close(fig)
        (shutil.copy(p, os.path.join(MANU_FIG, "drift_map.png")) if os.environ.get("MANU_COPY") else None)
        print("[02] wrote", os.path.basename(p), "(+ manuscript/figures/drift_map.png)")

    order = summ.sort_values("psi_early_late")
    picks = [order.iloc[-1]["zone"], order.iloc[len(order)//2]["zone"], order.iloc[0]["zone"]]
    labels = ["high drift", "median drift", "low drift"]
    fig, axs = plt.subplots(3, 1, figsize=(7.5, 5.2), sharex=True)
    for axx, z, lab in zip(axs, picks, labels):
        s = df[z].values
        axx.plot(s, lw=0.4, alpha=0.5, label="hourly")
        axx.plot(pd.Series(s).rolling(U.SEASON_WEEK, min_periods=24).mean(), lw=1.6, color="crimson", label="weekly mean")
        axx.set_ylabel("kWh"); axx.legend(loc="upper right", fontsize=7)
        psi_v = float(summ.loc[summ.zone == z, "psi_early_late"].iloc[0])
        axx.set_title("Zone %s (%s, PSI=%.2f)" % (z, lab, psi_v), fontsize=9)
    axs[-1].set_xlabel("hour index (Sep 2022 - Feb 2023)")
    fig.tight_layout(); p = os.path.join(U.FIGURES, "drift_examples_%s.png" % target); fig.savefig(p, dpi=600); plt.close(fig)
    (shutil.copy(p, os.path.join(MANU_FIG, "drift_examples.png")) if os.environ.get("MANU_COPY") else None)
    print("[02] wrote", os.path.basename(p), "(+ manuscript/figures/drift_examples.png)")

    stats = {
        "n_zones": int(len(summ)),
        "psi_median": float(summ["psi_early_late"].median()),
        "psi_p75": float(summ["psi_early_late"].quantile(0.75)),
        "pct_psi_gt_0.25": float((summ["psi_early_late"] > 0.25).mean() * 100),
        "mean_shift_min": float(summ["mean_shift_ratio"].min()),
        "mean_shift_max": float(summ["mean_shift_ratio"].max()),
        "drift_type_counts": summ["drift_type"].value_counts().to_dict(),
    }
    with open(os.path.join(U.RESULTS, "drift_headline_%s.json" % target), "w") as f:
        json.dump(stats, f, indent=2)
    print("[02] headline:", json.dumps(stats))


if __name__ == "__main__":
    mz = os.environ.get("MAXZONES")
    main(max_zones=int(mz) if mz else None)
