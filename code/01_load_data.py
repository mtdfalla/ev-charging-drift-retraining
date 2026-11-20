"""01 - Load & prepare UrbanEV targets into processed tensors + split metadata.
Output: data/processed/<target>.parquet and split_info.json
Run: python code/01_load_data.py
"""
import os, json
import numpy as np
import pandas as pd
import utils as U

TARGETS = ["volume", "occupancy", "duration"]


def main():
    info = {}
    for t in TARGETS:
        df = U.load_target(t)
        # basic cleaning: forward/backfill residual gaps, clip negatives to 0
        df = df.ffill().bfill().clip(lower=0)
        out = os.path.join(U.PROCESSED, f"{t}.parquet")
        try:
            df.to_parquet(out)
        except Exception:
            out = os.path.join(U.PROCESSED, f"{t}.csv")
            df.to_csv(out)
        n = len(df)
        tr, va, te = U.temporal_split(n)
        info[t] = {
            "rows": int(n), "zones": int(df.shape[1]),
            "start": str(df.index.min()), "end": str(df.index.max()),
            "train_end_idx": int(tr.stop), "val_end_idx": int(va.stop),
            "saved": os.path.basename(out),
            "mean": float(df.values.mean()), "max": float(df.values.max()),
        }
        print(f"[01] {t}: {df.shape} saved -> {os.path.basename(out)}")
    with open(os.path.join(U.PROCESSED, "split_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print("[01] wrote split_info.json")


if __name__ == "__main__":
    main()
