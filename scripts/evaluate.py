"""Stylized facts for real_test, real_train, GBM and diffusion on equal footing.

    python scripts/evaluate.py --run default

Writes docs/phase08_results.json and the figures in reports/figures/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from src import config, data, metrics

FIG = data.ROOT / "reports" / "figures"
COLORS = {"real_test": "#222222", "real_train": "#888888", "gbm": "#d1495b", "diffusion": "#2e86ab"}
LABELS = {"real_test": "Real (test 2020–24)", "real_train": "Real (train 2005–19)", "gbm": "GBM", "diffusion": "Diffusion"}


def closer(a: float, b: float, ref: float) -> bool:
    return abs(a - ref) < abs(b - ref)


def criteria(s: dict) -> dict:
    """PLAN.md success criteria; True means diffusion passes."""
    ref, d, g = s["real_test"], s["diffusion"], s["gbm"]
    return {
        "vol_clustering_closer_than_gbm": closer(d["acf_abs_mean"], g["acf_abs_mean"], ref["acf_abs_mean"]),
        "kurtosis_closer_than_gbm": closer(d["excess_kurtosis"], g["excess_kurtosis"], ref["excess_kurtosis"]),
        "skew_closer_than_gbm": closer(d["skew"], g["skew"], ref["skew"]),
        "linear_acf_not_above_real": d["acf_r_mean_abs"] <= ref["acf_r_mean_abs"],
        "wasserstein_below_gbm": d["wasserstein_to_real_test"] < g["wasserstein_to_real_test"],
    }


def plot_acf(sources: dict, s: dict) -> None:
    lags = np.arange(1, config.ACF_LAGS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for ax, key, title in ((axes[0], "acf_r", "ACF of returns r"), (axes[1], "acf_abs", "ACF of |r| (volatility clustering)")):
        for name in sources:
            ax.plot(lags, s[name][key], marker="o", ms=3, lw=1.5, color=COLORS[name], label=LABELS[name])
        ax.axhline(0, color="black", lw=0.6)
        band = s["real_test"]["acf_band"]
        ax.axhspan(-band, band, color="#cccccc", alpha=0.4, lw=0)
        ax.set_title(title)
        ax.set_xlabel("lag (trading days)")
    axes[0].set_ylabel("autocorrelation")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "acf_comparison.png", dpi=150)
    plt.close(fig)


def plot_paths(sources: dict, seed: int = config.SEED) -> None:
    """Cumulative standardized return paths: 8 random windows per source."""
    rng = np.random.default_rng(seed)
    names = ["real_test", "gbm", "diffusion"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, name in zip(axes, names):
        w = sources[name]
        for i in rng.choice(len(w), size=8, replace=False):
            ax.plot(np.cumsum(w[i]), color=COLORS[name], alpha=0.75, lw=1)
        ax.set_title(LABELS[name])
        ax.set_xlabel("day in window")
    axes[0].set_ylabel("cumulative return (train-std units)")
    fig.tight_layout()
    fig.savefig(FIG / "sample_paths.png", dpi=150)
    plt.close(fig)


def plot_distribution(sources: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(-8, 8, 121)
    for name in ("real_test", "gbm", "diffusion"):
        r = sources[name].ravel()
        axes[0].hist(r, bins=bins, density=True, histtype="step", lw=1.5, color=COLORS[name], label=LABELS[name])
        q = np.linspace(0.001, 0.999, 400)
        axes[1].plot(stats.norm.ppf(q), np.quantile(r, q), color=COLORS[name], lw=1.5, label=LABELS[name])
    axes[0].set_yscale("log")
    axes[0].set_title("Return density (log scale)")
    axes[0].set_xlabel("standardized return")
    axes[1].plot([-3.2, 3.2], [-3.2, 3.2], color="black", lw=0.6, ls="--")
    axes[1].set_title("Q-Q vs normal")
    axes[1].set_xlabel("normal quantile")
    axes[1].set_ylabel("sample quantile")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "return_distribution.png", dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="default")
    args = p.parse_args()
    ds = data.load_dataset()
    runs = data.ROOT / "runs"
    sources = {
        "real_test": ds["test_eval"],
        "real_train": ds["train_eval"],
        "gbm": np.load(runs / "gbm" / "samples.npy"),
        "diffusion": np.load(runs / args.run / "samples.npy"),
    }
    s = {k: metrics.summarize(v, reference=ds["test_eval"]) for k, v in sources.items()}
    results = {
        "run": args.run,
        "window": config.WINDOW,
        "acf_lags": config.ACF_LAGS,
        "tickers": ds["tickers"],
        "train": [str(config.TRAIN_START.date()), str(config.TRAIN_END.date())],
        "test": [str(config.TEST_START.date()), str(config.TEST_END.date())],
        "sources": s,
        "criteria": criteria(s),
        # Reported for transparency only; the run above was chosen on train data (select_run.py).
        "all_runs_on_test": {
            n: {k: m[k] for k in ("std", "excess_kurtosis", "skew", "acf_r_mean_abs", "acf_abs_mean", "wasserstein_to_real_test")}
            for n in sorted(p.name for p in runs.glob("r[0-9]*") if (p / "samples.npy").exists())
            for m in [metrics.summarize(np.load(runs / n / "samples.npy"), reference=ds["test_eval"])]
        },
    }
    out = data.ROOT / "docs" / "phase08_results.json"
    out.write_text(json.dumps(results, indent=2))

    FIG.mkdir(parents=True, exist_ok=True)
    plot_acf(sources, s)
    plot_paths(sources)
    plot_distribution(sources)

    cols = ("excess_kurtosis", "skew", "acf_r_mean_abs", "acf_abs_mean", "acf_sq_mean", "wasserstein_to_real_test")
    print(f"{'':11s}" + "".join(f"{c[:14]:>16s}" for c in cols))
    for k, v in s.items():
        print(f"{k:11s}" + "".join(f"{v[c]:16.3f}" for c in cols))
    print(json.dumps(results["criteria"], indent=2))


if __name__ == "__main__":
    main()
