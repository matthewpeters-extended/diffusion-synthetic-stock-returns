"""Generate and execute the two walkthrough notebooks.

    python scripts/build_notebooks.py

Needs data/processed/windows.npz, runs/gbm and the run named in docs/phase08_results.json.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"

SETUP = """import sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from src import config, data, metrics
ds = data.load_dataset(ROOT / "data" / "processed" / "windows.npz")"""

EXPLORE = [
    ("md", """# 01 · What do real returns look like?

Before generating anything synthetic, establish the target. These are the stylized facts from SCOPE.md, measured on real daily log returns for SPY, AAPL, JPM, XOM and JNJ. Each ticker is scaled by its own 2005–2019 standard deviation."""),
    ("code", SETUP),
    ("code", """for name in ("train_eval", "test_eval"):
    print(name, ds[name].shape)
print({t: round(s, 5) for t, s in ds["scale"].items()})"""),
    ("md", "## A price path, and the returns that build it\n\nSPY over the full period. The return series shows calm and violent stretches: that is volatility clustering, visible by eye."),
    ("code", """close = data.load_close("SPY")
r = data.log_returns(close)
fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
ax[0].plot(close.index, close.values, lw=1); ax[0].set_ylabel("SPY adj. close")
ax[1].plot(r.index, r.values, lw=0.5); ax[1].set_ylabel("log return")
for a in ax: a.axvline(config.TEST_START, color="red", ls="--", lw=1)
ax[0].set_title("Red line = holdout cutoff (2020-01-02)")
plt.tight_layout()"""),
    ("md", "## Fact 1 and 4: fat tails and skew\n\nA normal distribution has excess kurtosis 0 and skew 0."),
    ("code", """for name in ("train_eval", "test_eval"):
    x = ds[name].ravel()
    print(f"{name:10s} excess kurtosis {stats.kurtosis(x):6.2f}   skew {stats.skew(x):6.2f}")
x = ds["test_eval"].ravel()
q = np.linspace(0.001, 0.999, 400)
plt.figure(figsize=(5, 5))
plt.plot(stats.norm.ppf(q), np.quantile(x, q), lw=1.5, label="real test")
plt.plot([-3.2, 3.2], [-3.2, 3.2], "k--", lw=0.6, label="normal")
plt.xlabel("normal quantile"); plt.ylabel("sample quantile"); plt.legend(); plt.title("Q-Q vs normal")"""),
    ("md", "## Fact 2 and 3: autocorrelation of r vs |r|\n\nRaw returns are close to uncorrelated. Absolute returns are strongly and persistently correlated. ACFs pool lag products inside 64-day windows only."),
    ("code", """lags = np.arange(1, config.ACF_LAGS + 1)
fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for name, c in (("train_eval", "gray"), ("test_eval", "black")):
    ax[0].plot(lags, metrics.pooled_acf(ds[name]), "o-", ms=3, color=c, label=name)
    ax[1].plot(lags, metrics.pooled_acf(np.abs(ds[name])), "o-", ms=3, color=c, label=name)
band = 2 / np.sqrt(ds["test_eval"].size)
for a, t in zip(ax, ("ACF of r", "ACF of |r|")):
    a.axhspan(-band, band, color="#ccc", alpha=0.5); a.axhline(0, color="k", lw=0.5); a.set_title(t); a.set_xlabel("lag")
ax[1].legend()"""),
    ("md", """## Takeaway

- **Fat tails:** strong in both periods.
- **Volatility clustering:** strong, decaying slowly over 20 lags. This is the fact GBM cannot produce.
- **Linear ACF:** small but not zero. Lag 1 is negative, outside the white-noise band.
- **Skew:** clearly negative only in the test period.

Notebook 02 asks how GBM and the diffusion model compare against this target."""),
]

COMPARE = [
    ("md", """# 02 · Diffusion vs GBM vs real

Both generators produce 2,000 windows of 64 days. Everything is compared against the real 2020–2024 holdout, which neither generator saw. The numbers here are re-read from `docs/phase08_results.json`, the same file the README is checked against."""),
    ("code", SETUP + """
runs = ROOT / "runs"
res = json.loads((ROOT / "docs" / "phase08_results.json").read_text())
src = {"real_test": ds["test_eval"], "gbm": np.load(runs / "gbm" / "samples.npy"),
       "diffusion": np.load(runs / res["run"] / "samples.npy")}
print("diffusion run:", res["run"])
colors = {"real_test": "#222222", "gbm": "#d1495b", "diffusion": "#2e86ab"}"""),
    ("md", "## Sample paths\n\nCumulative sums of 8 random windows from each source. GBM paths all look alike. Real and diffusion paths mix calm windows with wild ones."),
    ("code", """rng = np.random.default_rng(0)
fig, ax = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for a, (k, w) in zip(ax, src.items()):
    for i in rng.choice(len(w), 8, replace=False):
        a.plot(np.cumsum(w[i]), color=colors[k], alpha=0.75, lw=1)
    a.set_title(k)"""),
    ("md", "## The stylized-fact table"),
    ("code", """cols = ["excess_kurtosis", "skew", "acf_r_mean_abs", "acf_abs_mean", "acf_sq_mean", "wasserstein_to_real_test"]
print(f"{'':11s}" + "".join(f"{c[:14]:>16s}" for c in cols))
for k in ("real_test", "real_train", "gbm", "diffusion"):
    v = res["sources"][k]
    print(f"{k:11s}" + "".join(f"{v[c]:16.3f}" for c in cols))
res["criteria"]"""),
    ("md", "## Headline: volatility clustering\n\nThe ACF of |r| is where a learned generator must separate from GBM. GBM's i.i.d. Gaussian returns give a flat zero line by construction."),
    ("code", """lags = np.arange(1, config.ACF_LAGS + 1)
plt.figure(figsize=(7, 4))
for k in ("real_test", "gbm", "diffusion"):
    plt.plot(lags, res["sources"][k]["acf_abs"], "o-", ms=3, color=colors[k], label=k)
plt.axhline(0, color="k", lw=0.5); plt.xlabel("lag"); plt.ylabel("ACF of |r|"); plt.legend()"""),
    ("md", "## Tails\n\nThe Q-Q plot shows how far each source's tails stretch beyond a normal distribution."),
    ("code", """q = np.linspace(0.001, 0.999, 400)
plt.figure(figsize=(5, 5))
for k, w in src.items():
    plt.plot(stats.norm.ppf(q), np.quantile(w.ravel(), q), color=colors[k], label=k)
plt.plot([-3.2, 3.2], [-3.2, 3.2], "k--", lw=0.6); plt.legend(); plt.xlabel("normal quantile"); plt.ylabel("sample quantile")"""),
    ("md", "See `docs/RESULTS.md` for the written interpretation and caveats."),
]


def build(cells, name: str) -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(c) if kind == "md" else nbf.v4.new_code_cell(c) for kind, c in cells]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(NB)}})
    nbf.write(nb, NB / name)
    print("wrote", NB / name)


if __name__ == "__main__":
    build(EXPLORE, "01_explore_real_data.ipynb")
    build(COMPARE, "02_diffusion_vs_gbm.ipynb")
