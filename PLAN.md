# Plan

Written before any model code (MAT-43). Scope details live in [SCOPE.md](SCOPE.md). This file covers what gets built, in what order, and how success is judged.

## What this is

An original build inspired by the "Unconditional Generation" section of the [Diffusion Models in Finance](https://github.com/ZhuoHan1998/Diffusion-Models-In-Finance) paper list (see [docs/sources.md](docs/sources.md)). It is not a port of any one paper's repo. So unlike the InfoDiffusion project, there is no "defects we inherit and fix" section. Every defect here is our own.

## Scope (from SCOPE.md)

- **Universe:** SPY, AAPL, JPM, XOM, JNJ. Daily adjusted close, 2005-01-03 to 2024-12-31.
- **Unit of generation:** one window of 64 daily log returns. Each ticker is scaled by its own train-period standard deviation (one constant per ticker, never per window).
- **Split:** train 2005 to 2019, test 2020 to 2024, by time. No window crosses the boundary.
- **Stylized facts:** (1) fat tails, (2) no linear autocorrelation, (3) volatility clustering, (4) negative skew.
- **Non-goals:** cross-asset correlation, tick or order-book data, trading or backtesting, conditioning, forecasting.

## Baseline: GBM Monte Carlo

Per-ticker drift and volatility come from train-period log returns. Log returns are simulated i.i.d. normal in 64-day windows.

GBM is the right naive comparison because it is the textbook "diffusion" of quantitative finance: an SDE whose log returns are Gaussian and independent. By construction it has zero excess kurtosis, zero skew and no volatility clustering. Any learned generator worth its training cost has to beat it on exactly those three. The paper list's own scope note excludes this kind of "diffusion in the SDE sense". That exclusion is what makes GBM the baseline here.

## Model

A plain DDPM over 1D sequences:

- **Denoiser:** a residual 1D conv net with dilations 1 to 32, so the receptive field covers the whole 64-day window. A sinusoidal timestep embedding is added to every block.
- **Objective:** epsilon prediction (MSE) with a cosine noise schedule and T = 1000 steps.
- **Training:** unconditional, on all overlapping train windows from all five tickers pooled. CPU only.
- **Sampling:** ancestral DDPM sampling.

There is deliberately no latent variable, no conditioning, and nothing exotic. The interesting part of this project is the financial validation.

## Evaluation

Every metric is computed the same way on four sources:

| Source | What it is |
|--------|------------|
| `real_test` | Non-overlapping 64-day windows from 2020 to 2024. **The reference.** |
| `real_train` | Non-overlapping train windows. Shows how much of any gap is regime shift rather than model error. |
| `gbm` | GBM samples, with the same window count per ticker as the diffusion model's samples. |
| `diffusion` | Diffusion samples. |

The metrics:

- Excess kurtosis and skewness of the pooled returns.
- ACF of `r`, `|r|` and `r²` at lags 1 to 20. Lag products are pooled within windows, never across window boundaries.
- Wasserstein-1 distance from each source's return distribution to `real_test`.

All numbers go to `docs/phase08_results.json`. The README's headline numbers are checked against that file by `scripts/verify_readme.py`.

## Success criteria (measurable)

Every comparison is on the same windows and in the same standardized units:

1. **Headline, volatility clustering:** the mean ACF of `|r|` over lags 1 to 20 for diffusion is closer to `real_test` than GBM's is.
2. **Fat tails:** diffusion's excess kurtosis is closer to `real_test` than GBM's is.
3. **Skew:** diffusion's skewness is closer to `real_test` than GBM's is.
4. **No linear autocorrelation:** diffusion's mean |ACF(r)| over lags 1 to 20 stays below 2/√N, where N is the number of diffusion returns.
5. **Distribution:** diffusion's Wasserstein-1 distance to `real_test` is smaller than GBM's.

These are reported pass or fail as measured, including any failures. RESULTS.md states the numbers plainly, whichever way they land.

## Build order (Linear tickets)

| Ticket | Work |
|--------|------|
| 04 | `scripts/fetch_data.py` and `src/data.py`: raw CSVs, log returns, scaling, windows. `docs/DATA.md` |
| 05 | `src/gbm.py`: calibrate and simulate |
| 06 | `src/diffusion.py`: denoiser, schedule, loss, sampler |
| 07 | `scripts/run_experiment.py` with `configs/default.json`: train and sample into `runs/<name>/` (gitignored) |
| 08 | `src/metrics.py` and `scripts/evaluate.py`: produce `docs/phase08_results.json` and the figures |
| 09 | CPU-only tests for every module above |
| 10 | Two notebooks and WALKTHROUGH.md |
| 11 | `docs/RESULTS.md` and `reports/figures/` (tracked) |
| 12 | README.md and `scripts/verify_readme.py` |
| 13 | Push to GitHub. Remote creation is manual. |

## Known limits

- 64-day windows cap the ACF lags at 20. Long-memory decay is out of reach.
- The test period has only about 95 non-overlapping windows, so its statistics are noisy. Kurtosis in particular is driven by a handful of COVID days.
- Diffusion models start from Gaussian noise and can under-produce extreme tails. If kurtosis comes in short, that is reported, not tuned away against the test set.
