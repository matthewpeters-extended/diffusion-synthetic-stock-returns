"""Raw prices -> scaled log returns -> fixed-length windows.

Windows are cut inside each split separately, so no window can straddle the
train/test boundary.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def fetch_prices(tickers=config.TICKERS) -> None:
    """Download daily OHLCV with yfinance and save one CSV per ticker to data/raw."""
    import yfinance as yf

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    end = config.DATA_END + pd.Timedelta(days=1)  # yfinance's end is exclusive
    for t in tickers:
        df = yf.download(t, start=config.DATA_START, end=end, auto_adjust=True, progress=False)
        if df.empty:
            raise RuntimeError(f"yfinance returned no data for {t}")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.to_csv(RAW_DIR / f"{t}.csv")


def load_close(ticker: str) -> pd.Series:
    df = pd.read_csv(RAW_DIR / f"{ticker}.csv", index_col=0, parse_dates=True)
    close = df["Close"].dropna()
    if (close <= 0).any():
        raise ValueError(f"{ticker}: non-positive close price")
    return close


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff().dropna()


def split(returns: pd.Series) -> tuple[pd.Series, pd.Series]:
    train = returns[(returns.index >= config.TRAIN_START) & (returns.index <= config.TRAIN_END)]
    test = returns[(returns.index >= config.TEST_START) & (returns.index <= config.TEST_END)]
    return train, test


def make_windows(x: np.ndarray, length: int, stride: int) -> np.ndarray:
    """Slice a 1D array into (n, length) windows; the incomplete tail is dropped."""
    n = (len(x) - length) // stride + 1
    if n <= 0:
        return np.empty((0, length), dtype=np.float32)
    idx = np.arange(length)[None, :] + stride * np.arange(n)[:, None]
    return x[idx].astype(np.float32)


def build_dataset(tickers=config.TICKERS) -> dict:
    """Per-ticker scaled returns and windows for each split.

    Each ticker is divided by its own train-period std (SCOPE.md, amended in MAT-43).
    """
    out = {"tickers": list(tickers), "scale": {}, "mu": {}, "sigma": {}}
    parts = {k: [] for k in ("train", "train_eval", "test_eval")}
    labels = {k: [] for k in parts}
    for i, t in enumerate(tickers):
        train, test = split(log_returns(load_close(t)))
        s = float(train.std())
        out["scale"][t] = s
        tr, te = train.to_numpy() / s, test.to_numpy() / s
        # GBM calibration inputs, in the same standardized units.
        out["mu"][t] = float(tr.mean())
        out["sigma"][t] = float(tr.std())
        for key, arr, stride in (
            ("train", tr, config.TRAIN_STRIDE),
            ("train_eval", tr, config.EVAL_STRIDE),
            ("test_eval", te, config.EVAL_STRIDE),
        ):
            w = make_windows(arr, config.WINDOW, stride)
            parts[key].append(w)
            labels[key].append(np.full(len(w), i))
    for k in parts:
        out[k] = np.concatenate(parts[k])
        out[k + "_ticker"] = np.concatenate(labels[k])
    return out


def save_dataset(ds: dict, path: Path = PROCESSED_DIR / "windows.npz") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in ds.items() if isinstance(v, np.ndarray)}
    meta = {k: ds[k] for k in ("tickers", "scale", "mu", "sigma")}
    np.savez(path, meta=np.array(json.dumps(meta)), **arrays)


def load_dataset(path: Path = PROCESSED_DIR / "windows.npz") -> dict:
    z = np.load(path)
    ds = {k: z[k] for k in z.files if k != "meta"}
    ds.update(json.loads(str(z["meta"])))
    return ds
