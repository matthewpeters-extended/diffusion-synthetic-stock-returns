# Synthetic Financial Time-Series via Diffusion

A small diffusion model that generates realistic 64-day paths of daily stock returns, tested against the classical baseline (geometric Brownian motion) on the statistical fingerprints real markets leave behind.

**Idea source:** the "Unconditional Generation" section of [Diffusion Models in Finance](https://github.com/ZhuoHan1998/Diffusion-Models-In-Finance). This is original code inspired by that list, not a port of any one paper. See [docs/sources.md](docs/sources.md).

## Why generate synthetic returns?

- **Backtest augmentation:** history gives only one realized path. Plausible alternatives help test whether a strategy was lucky.
- **Stress-testing:** generate many volatile windows without waiting for the next crisis.
- **Privacy-preserving sharing:** synthetic data can stand in where real data can't be shared.

All three only work if the synthetic data behaves like real returns. The question here is how well.

## The test: volatility clustering

Real returns have a signature that GBM cannot produce: calm and wild periods come in streaks, so the size of today's move predicts the size of tomorrow's. GBM draws every day independently, which makes this correlation exactly zero. The classical stochastic-calculus "diffusion" is the kind the paper list's own scope note excludes, and that is why it is the right naive baseline.

Mean autocorrelation of absolute returns over lags 1 to 20, on the real 2020–2024 holdout that neither model saw:

| Source | Mean ACF of \|r\| |
|--------|-------------------|
| Real (2020–24) | **0.250** |
| GBM baseline | **-0.002** |
| Diffusion model | **0.053** |

![ACF comparison](reports/figures/acf_comparison.png)

## Results, including where it falls short

Against the real holdout (full table in [docs/RESULTS.md](docs/RESULTS.md)):

| | Excess kurtosis | Skew | Distance to real (Wasserstein-1) | Return std |
|---|---|---|---|---|
| Real (2020–24) | 8.446 | -0.195 | 0 | 1.166 |
| GBM | 0.021 | -0.004 | 0.146 | 1.001 |
| Diffusion | 6.499 | -0.312 | 0.352 | 0.606 |

The diffusion model passes four of five success criteria set in [PLAN.md](PLAN.md): it is closer to real than GBM on volatility clustering, fat tails and skew, and it adds no spurious linear autocorrelation.

**It fails the fifth: distance to real data.** Its samples are too quiet (return std 0.606 against about 1.0 to 1.17 for real data), so GBM, which gets the overall scale right, wins that metric despite having no fat tails. The model also captures only about a fifth of the real volatility clustering. Three further training configurations did not fix the scale problem; all four are tabulated in the results.

![Sample paths](reports/figures/sample_paths.png)

## What was built

- **Data:** SPY, AAPL, JPM, XOM, JNJ daily returns, 2005–2024. Train 2005–2019, test 2020–2024, split by time with no window crossing the boundary. Each ticker is scaled by its own training-period standard deviation. See [docs/DATA.md](docs/DATA.md).
- **Baseline:** per-ticker GBM calibrated on training returns (`src/gbm.py`).
- **Model:** a plain DDPM with a dilated 1D residual conv denoiser (`src/diffusion.py`). No latent variables, no conditioning.
- **Evaluation:** kurtosis, skew, autocorrelation of r and |r| and r², and Wasserstein-1 distance, identical for every source (`src/metrics.py`).
- **Run selection:** among four training configurations, the reported one was chosen using **train-period data only**, by a rule fixed beforehand. The holdout never influenced the choice.

## Limits

- 95 holdout windows, so kurtosis and skew are noisy.
- 64-day windows only test autocorrelation up to lag 20.
- One seed per configuration.
- No cross-asset structure, no conditioning, no trading evaluation (see [SCOPE.md](SCOPE.md)).

## Reproduce

```bash
# setup: docs/SETUP.md
python scripts/fetch_data.py --no-fetch      # build windows from the committed CSVs
python scripts/run_gbm.py
python scripts/run_experiment.py --config configs/r1_lr2e-4_20k.json   # ~20 min on CPU
python scripts/evaluate.py --run r1_lr2e-4_20k
python scripts/verify_readme.py
pytest
```

Further reading: [WALKTHROUGH.md](WALKTHROUGH.md) (prose tour), [notebooks/](notebooks/).
