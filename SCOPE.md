# Scope: Synthetic Financial Time-Series via Diffusion

Decisions from MAT-41. This is the input for PLAN.md (MAT-43).

## Question

Can a small diffusion model generate daily return sequences that reproduce the stylized facts of real equity returns, where a calibrated GBM baseline cannot?

## Universe

| Ticker | Why |
|--------|-----|
| SPY | Broad index, the main target |
| AAPL | Mega-cap tech |
| JPM | Financials, strong 2008 crisis behavior |
| XOM | Energy, commodity-driven volatility |
| JNJ | Defensive, low volatility |

Each ticker is modeled on its own. There is no cross-asset correlation model. Pooling all five tickers into one training set tests whether the model generalizes across volatility levels.

- Source: daily adjusted close from Yahoo Finance (`yfinance`).
- Date range: 2005-01-03 to 2024-12-31. This covers the 2008 crisis, 2011, 2020 COVID, and 2022.

## Representation

- Daily log returns: `r_t = ln(P_t / P_{t-1})`.
- Window length **L = 64** trading days. The diffusion model generates one window, shape `(64,)`.
- Scaling: divide each ticker's returns by that ticker's own train-split standard deviation. This is one constant per ticker. Do **not** standardize each window. Per-window scaling would erase volatility regimes, and volatility regimes are what we measure.
  - *Amended in MAT-43:* this replaces "one global standard deviation". If all tickers share one scale, then pooling a low-vol ticker (JNJ) with a high-vol one (AAPL) makes |r| autocorrelated across the pool, even for i.i.d. GBM. That would contaminate the volatility-clustering metric.
- Training windows use stride 1 (overlapping). Evaluation windows use stride 64 (non-overlapping), so the statistics are not inflated by overlap.

## Train / test split (by time, never shuffled)

| Split | Dates | Purpose |
|-------|-------|---------|
| Train | 2005-01-03 to 2019-12-31 | Fit the diffusion model and calibrate GBM |
| Test (real holdout) | 2020-01-02 to 2024-12-31 | Real reference for evaluation |

- No window may cross the split boundary.
- The test period contains COVID, which is an out-of-sample regime. That is intended. Report the stylized facts on train-period real data as well, so that a regime shift is not mistaken for model failure.

## Stylized facts to validate

Each fact is computed on three sources: real returns, diffusion samples and GBM samples. First confirm the fact holds in **real** data (done in MAT-44). If it does not, drop it.

| # | Fact | Metric | Real data should show |
|---|------|--------|-----------------------|
| 1 | Fat tails | Excess kurtosis of pooled returns; Q-Q plot against a normal | Excess kurtosis well above 0 (typically 5 to 20 for daily equity) |
| 2 | No linear autocorrelation | ACF of `r` at lags 1 to 20 | All lags inside roughly ±2/√N |
| 3 | Volatility clustering | ACF of `\|r\|` and `r²` at lags 1 to 20 | Positive at all lags, slowly decaying |
| 4 | Negative skew | Skewness of pooled returns | Below 0 |

How to compare: for each metric, report the real value next to the diffusion value and the GBM value. Add a distribution distance as well: the Wasserstein-1 distance between the return marginals.

**Known limit:** a 64-day window caps the ACF lags we can measure. Lags 1 to 20 work well. Long-memory decay (lags up to 100+) cannot be tested at this window length. Say this in the README rather than claim it.

**Stretch (not required):** the leverage effect, i.e. a negative correlation between `r_t` and `|r_{t+k}|`.

## Baseline: GBM Monte Carlo

- Calibrate `μ` and `σ` for each ticker from train-split log returns.
- Simulate `r_t ~ N(μ − σ²/2, σ²)` i.i.d. in windows of 64 days. Draw the same number of windows as the diffusion model.
- By construction it fails facts 1, 3 and 4: Gaussian, i.i.d., and symmetric. It passes fact 2. That is the point. It is the naive "diffusion" in the stochastic-calculus sense, the kind that the awesome-diffusion list's scope note excludes. The README (MAT-52) must state this explicitly.

## Non-goals

- No multi-asset correlation structure or joint generation.
- No limit-order-book, intraday or tick data.
- No live trading, backtesting, or strategy evaluation.
- No conditioning on regimes, macro data or text. The model is unconditional.
- No forecasting. This project measures the quality of a generative model, not prediction accuracy.

## Done means

- The diffusion samples get closer to the real values than GBM on facts 1, 3 and 4, and stay near zero on fact 2.
- Every number comes from the non-overlapping test windows of the real holdout.
