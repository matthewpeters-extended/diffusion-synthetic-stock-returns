"""Download raw prices (skip with --no-fetch) and build data/processed/windows.npz."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--no-fetch", action="store_true", help="reuse CSVs already in data/raw")
    args = p.parse_args()
    if not args.no_fetch:
        data.fetch_prices()
    ds = data.build_dataset()
    data.save_dataset(ds)
    for k in ("train", "train_eval", "test_eval"):
        print(f"{k:11s} windows: {ds[k].shape}")
    for t in ds["tickers"]:
        print(f"{t:5s} train std of log returns: {ds['scale'][t]:.5f}")


if __name__ == "__main__":
    main()
