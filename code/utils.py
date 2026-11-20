"""Shared utilities: paths, seeds, data loading, metrics.
Supports multiple datasets via env vars:
  DATAROOT   -> raw data dir (default: <repo>/data/raw)
  RESULTS_DIR-> results dir  (default: <repo>/experiments/results)
  FIG_DIR    -> figures dir  (default: <repo>/experiments/figures)
  TARGET     -> primary target file stem (default: volume)
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE_DIR)
RAW = os.environ.get("DATAROOT", os.path.join(ROOT, "data", "raw"))
PROCESSED = os.path.join(ROOT, "data", "processed")
RESULTS = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "experiments", "results"))
FIGURES = os.environ.get("FIG_DIR", os.path.join(ROOT, "experiments", "figures"))
for _d in (PROCESSED, RESULTS, FIGURES):
    os.makedirs(_d, exist_ok=True)

PRIMARY_TARGET = os.environ.get("TARGET", "volume")
SEASON_DAY = 24
SEASON_WEEK = 24 * 7
HORIZONS = (1, 24)


def load_target(name: str = None) -> pd.DataFrame:
    name = name or PRIMARY_TARGET
    fp = os.path.join(RAW, f"{name}.csv")
    df = pd.read_csv(fp)
    tcol = df.columns[0]
    df[tcol] = pd.to_datetime(df[tcol])
    df = df.set_index(tcol)
    df.columns = [str(c) for c in df.columns]
    return df.astype(float)


def temporal_split(n: int, train=0.6, val=0.2):
    i_tr = int(n * train); i_va = int(n * (train + val))
    return slice(0, i_tr), slice(i_tr, i_va), slice(i_va, n)


def mae(y, yhat):  return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))
def rmse(y, yhat): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))

def smape(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    den = np.abs(y) + np.abs(yhat); m = den > 1e-8
    return float(np.mean(2*np.abs(y-yhat)[m]/den[m])*100) if m.any() else np.nan

def mape(y, yhat, eps=1.0):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = np.abs(y) > eps
    return float(np.mean(np.abs((y-yhat)[m]/y[m]))*100) if m.any() else np.nan


if __name__ == "__main__":
    df = load_target()
    print(f"[utils] RAW={RAW}\n[utils] loaded {PRIMARY_TARGET}: {df.shape}")
