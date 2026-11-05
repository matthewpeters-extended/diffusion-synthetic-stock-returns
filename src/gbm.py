"""GBM Monte Carlo baseline.

Under GBM, log returns are i.i.d. N(mu - sigma^2/2, sigma^2) per step. We calibrate
directly on log returns, so the log-return mean already equals mu - sigma^2/2 and is
used as-is.
"""

from __future__ import annotations

import numpy as np


def calibrate(log_returns: np.ndarray) -> tuple[float, float]:
    """(mean, std) of the log returns; equivalent to GBM's (mu - sigma^2/2, sigma)."""
    x = np.asarray(log_returns, dtype=np.float64).ravel()
    return float(x.mean()), float(x.std(ddof=1))


def simulate(mean: float, std: float, n_windows: int, length: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(mean, std, size=(n_windows, length)).astype(np.float32)


def simulate_pool(params: dict, counts: dict, length: int, seed: int) -> np.ndarray:
    """Concatenate per-ticker GBM windows; params[t] = (mean, std), counts[t] = n windows."""
    rng = np.random.default_rng(seed)
    return np.concatenate([simulate(*params[t], counts[t], length, rng) for t in counts])
