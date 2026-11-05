"""Stylized-fact metrics on (n_windows, length) arrays of standardized returns.

ACF lag products are pooled inside windows only: pairs never span two windows,
because neighbouring windows from different sources (or tickers) aren't contiguous.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from src import config


def pooled_acf(w: np.ndarray, max_lag: int = config.ACF_LAGS) -> np.ndarray:
    """ACF at lags 1..max_lag using one global mean/variance across all windows."""
    w = np.asarray(w, dtype=np.float64)
    if w.ndim != 2 or w.shape[1] <= max_lag:
        raise ValueError("expected (n_windows, length) with length > max_lag")
    d = w - w.mean()
    var = (d**2).mean()
    return np.array([(d[:, :-k] * d[:, k:]).mean() / var for k in range(1, max_lag + 1)])


def summarize(w: np.ndarray, reference: np.ndarray | None = None) -> dict:
    """All stylized-fact metrics for one source; Wasserstein-1 against `reference` if given."""
    r = np.asarray(w, dtype=np.float64).ravel()
    acf_r, acf_abs, acf_sq = pooled_acf(w), pooled_acf(np.abs(w)), pooled_acf(np.square(w))
    out = {
        "n_windows": int(len(w)),
        "n_returns": int(r.size),
        "mean": float(r.mean()),
        "std": float(r.std()),
        "excess_kurtosis": float(stats.kurtosis(r, fisher=True)),
        "skew": float(stats.skew(r)),
        "acf_r": acf_r.tolist(),
        "acf_abs": acf_abs.tolist(),
        "acf_sq": acf_sq.tolist(),
        "acf_r_mean_abs": float(np.abs(acf_r).mean()),
        "acf_abs_mean": float(acf_abs.mean()),
        "acf_sq_mean": float(acf_sq.mean()),
        "acf_band": float(2 / np.sqrt(r.size)),
    }
    if reference is not None:
        out["wasserstein_to_real_test"] = float(stats.wasserstein_distance(r, np.ravel(reference)))
    return out
