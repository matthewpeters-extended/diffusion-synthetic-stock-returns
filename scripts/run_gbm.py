"""Simulate the GBM baseline into runs/gbm/samples.npy (same window count as diffusion)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src import config, data, gbm, metrics


def main() -> None:
    ds = data.load_dataset()
    tickers = ds["tickers"]
    # Equal windows per ticker, matching the near-equal ticker mix of the training pool.
    counts = {t: config.N_SAMPLES // len(tickers) for t in tickers}
    params = {t: (ds["mu"][t], ds["sigma"][t]) for t in tickers}
    samples = gbm.simulate_pool(params, counts, config.WINDOW, config.SEED)
    out = data.ROOT / "runs" / "gbm"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "samples.npy", samples)
    s = metrics.summarize(samples, reference=ds["test_eval"])
    print(f"GBM {samples.shape}: kurt {s['excess_kurtosis']:.3f} skew {s['skew']:.3f} "
          f"acf|r| {s['acf_abs_mean']:.3f} W1 {s['wasserstein_to_real_test']:.3f}")


if __name__ == "__main__":
    main()
