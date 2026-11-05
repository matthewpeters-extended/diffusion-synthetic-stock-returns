# Data

## Source

Daily OHLCV from Yahoo Finance via `yfinance` (`auto_adjust=True`, so `Close` is adjusted for splits and dividends). Raw CSVs are committed in `data/raw/` so results reproduce even if Yahoo revises its history.

Rebuild everything:

```bash
python scripts/fetch_data.py            # download + build
python scripts/fetch_data.py --no-fetch # rebuild from the committed CSVs
```

## Universe and dates

| | |
|---|---|
| Tickers | SPY, AAPL, JPM, XOM, JNJ |
| Range | 2005-01-03 to 2024-12-31 |
| Train | 2005-01-03 to 2019-12-31 |
| **Holdout cutoff** | **2020-01-02**. Everything from this date on is test only. |
| Test | 2020-01-02 to 2024-12-31 |

All of these constants live in `src/config.py`.

## Processing

1. Compute log returns `r_t = ln(P_t / P_{t-1})` per ticker.
2. Split by date.
3. Divide each ticker by its **own train-period std**. That is one constant per ticker, applied to both train and test. Test windows are therefore in train units: the 2020 to 2024 test std is 1.17, not 1.0, because that period was more volatile.
4. Cut windows of 64 days **inside each split**, so no window crosses the cutoff:
   - `train`: stride 1, 18,555 windows. Used to fit the diffusion model.
   - `train_eval`: stride 64, 290 windows. The train-period reference.
   - `test_eval`: stride 64, 95 windows (6,080 returns). **The evaluation reference.**

Output: `data/processed/windows.npz` (gitignored and rebuilt by the script). It also stores each ticker's scale and its GBM calibration inputs.

Train-period std of daily log returns: SPY 0.01155, AAPL 0.02036, JPM 0.02350, XOM 0.01475, JNJ 0.01016.

## Do the stylized facts hold in real data? (checked before any modelling)

Pooled standardized returns, non-overlapping windows. ACFs are averaged over lags 1 to 20.

| Fact | Metric | Train 2005–19 | Test 2020–24 | Holds? |
|------|--------|---------------|--------------|--------|
| Fat tails | Excess kurtosis | 13.18 | 8.45 | Yes, strongly |
| No linear ACF | Mean \|ACF(r)\| | 0.025 (band 0.015) | 0.051 (band 0.026) | Roughly. Lag 1 is -0.07 / -0.09, outside ±2/√N |
| Vol clustering | Mean ACF(\|r\|) | 0.240 | 0.250 | Yes, strongly and slowly decaying (lag 1 ≈ 0.30, lag 20 ≈ 0.15) |
| Negative skew | Skewness | -0.03 | -0.20 | Weak in train, clear in test |

All four facts are kept, with two caveats. Skew is a weak signal in the training data, so a model trained on it cannot be expected to produce much skew. Fact 2 is judged against real data's own level, not against the white-noise band (see PLAN.md, success criterion 4).
