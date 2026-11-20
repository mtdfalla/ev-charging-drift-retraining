"""06 - Ingest City of Palo Alto ChargePoint sessions -> station x hourly volume (kWh).
Second dataset for generalization. Source: ev-load-open-data repo (Palo Alto, 47 stations).
Writes data_paloalto/raw/volume.csv in the same (time x series) format as UrbanEV.
Run: python code/06_paloalto_ingest.py /path/to/ChargePoint_Data.csv
"""
import os, sys
import numpy as np, pandas as pd
import utils as U

OUT = os.path.join(U.ROOT, "data_paloalto", "raw")
os.makedirs(OUT, exist_ok=True)


def main(src):
    df = pd.read_csv(src, low_memory=False)
    df["start"] = pd.to_datetime(df["Start Date"], errors="coerce")
    df["energy"] = pd.to_numeric(df["Energy (kWh)"], errors="coerce")
    df = df.dropna(subset=["start", "energy", "Station Name"])
    # dense modern window where most stations are active
    df = df[(df["start"] >= "2018-01-01") & (df["start"] < "2021-01-01")]
    df["hour"] = df["start"].dt.floor("h")
    df["station"] = df["Station Name"].str.replace("PALO ALTO CA / ", "", regex=False).str.strip()
    # energy delivered initiated per station-hour
    piv = df.groupby(["hour", "station"])["energy"].sum().unstack(fill_value=0.0)
    # full hourly index
    full = pd.date_range(piv.index.min(), piv.index.max(), freq="h")
    piv = piv.reindex(full, fill_value=0.0)
    # keep stations with enough activity (>= 1% of hours non-zero)
    active = piv.columns[(piv > 0).mean() >= 0.01]
    piv = piv[active]
    piv.index.name = "time"
    out = os.path.join(OUT, "volume.csv")
    piv.to_csv(out)
    print(f"[06] Palo Alto -> {piv.shape[0]} hours x {piv.shape[1]} stations")
    print(f"[06] span {piv.index.min()} -> {piv.index.max()}")
    print(f"[06] mean kWh/station-hour: {piv.values.mean():.2f}; wrote {out}")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ev-load-open-data/1. Input Data/4. City of Palo Alto/ChargePoint Data CY20Q4.csv"
    main(src)
