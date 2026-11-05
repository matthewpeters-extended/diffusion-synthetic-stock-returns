import numpy as np

from src import gbm


def test_simulated_moments_match_calibration():
    rng = np.random.default_rng(1)
    w = gbm.simulate(0.05, 1.3, n_windows=2000, length=64, rng=rng)
    assert w.shape == (2000, 64)
    assert abs(w.mean() - 0.05) < 0.01
    assert abs(w.std() - 1.3) < 0.01


def test_calibrate_recovers_inputs():
    x = np.random.default_rng(2).normal(-0.2, 0.7, 200_000)
    mean, std = gbm.calibrate(x)
    assert abs(mean + 0.2) < 0.01 and abs(std - 0.7) < 0.01


def test_pool_counts_and_reproducibility():
    params = {"A": (0.0, 1.0), "B": (0.0, 2.0)}
    a = gbm.simulate_pool(params, {"A": 3, "B": 5}, 64, seed=0)
    b = gbm.simulate_pool(params, {"A": 3, "B": 5}, 64, seed=0)
    assert a.shape == (8, 64)
    np.testing.assert_array_equal(a, b)
