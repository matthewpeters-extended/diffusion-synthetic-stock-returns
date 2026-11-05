"""Universe, split and window constants from SCOPE.md (MAT-41).

Defined once here so no script or notebook picks its own boundary after seeing a result.
"""

from __future__ import annotations

import pandas as pd

TICKERS = ["SPY", "AAPL", "JPM", "XOM", "JNJ"]

DATA_START = pd.Timestamp("2005-01-03")
DATA_END = pd.Timestamp("2024-12-31")

# Split by time, never shuffled. COVID falls in test on purpose (see SCOPE.md).
TRAIN_START = DATA_START
TRAIN_END = pd.Timestamp("2019-12-31")
TEST_START = pd.Timestamp("2020-01-02")
TEST_END = DATA_END

WINDOW = 64  # trading days per generated sequence
TRAIN_STRIDE = 1  # overlapping windows for training
EVAL_STRIDE = WINDOW  # non-overlapping windows for evaluation

ACF_LAGS = 20  # 64-day windows can't support longer lags honestly

SEED = 0
