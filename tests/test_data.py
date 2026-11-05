"""Windowing and the temporal holdout. Synthetic prices only: never touches Yahoo."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, data


def test_make_windows_shapes_and_contents():
    x = np.arange(10, dtype=float)
    w = data.make_windows(x, length=4, stride=3)
    assert w.shape == (3, 4)
    np.testing.assert_array_equal(w[1], [3, 4, 5, 6])
    assert data.make_windows(x, length=11, stride=1).shape == (0, 11)


def test_log_returns():
    close = pd.Series([100.0, 110.0, 99.0], index=pd.date_range("2020-01-01", periods=3))
    np.testing.assert_allclose(data.log_returns(close), np.log([1.1, 0.9]))


@pytest.fixture
def fake_raw(tmp_path, monkeypatch):
    dates = pd.bdate_range(config.DATA_START, config.DATA_END)
    rng = np.random.default_rng(0)
    for t in ("AAA", "BBB"):
        # Encode the date in the price so leakage is detectable: train prices < 1e6 <= test prices.
        level = np.where(dates < config.TEST_START, 100.0, 1e7)
        close = level * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))) * 0.01)
        pd.DataFrame({"Close": close}, index=dates.rename("Date")).to_csv(tmp_path / f"{t}.csv")
    monkeypatch.setattr(data, "RAW_DIR", tmp_path)
    return ["AAA", "BBB"]


def test_split_respects_cutoff(fake_raw):
    train, test = data.split(data.log_returns(data.load_close("AAA")))
    assert train.index.max() <= config.TRAIN_END
    assert test.index.min() >= config.TEST_START
    assert len(set(train.index) & set(test.index)) == 0


def test_no_window_straddles_the_holdout(fake_raw):
    ds = data.build_dataset(fake_raw)
    # The price jump at the cutoff yields one huge return (log 1e5 / scale); it sits on
    # the first test day and must not appear inside any train window.
    for key in ("train", "train_eval"):
        assert np.abs(ds[key]).max() < 50
    assert ds["test_eval"].shape[1] == config.WINDOW
    assert ds["train"].shape[0] > ds["train_eval"].shape[0]


def test_scaling_is_per_ticker_train_std(fake_raw):
    ds = data.build_dataset(fake_raw)
    for t in fake_raw:
        assert ds["sigma"][t] == pytest.approx(1.0, rel=1e-6)


def test_dataset_round_trip(fake_raw, tmp_path):
    ds = data.build_dataset(fake_raw)
    path = tmp_path / "w.npz"
    data.save_dataset(ds, path)
    back = data.load_dataset(path)
    np.testing.assert_array_equal(back["train"], ds["train"])
    assert back["scale"] == ds["scale"]
