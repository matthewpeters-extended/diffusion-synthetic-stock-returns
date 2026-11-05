"""Metric functions on inputs whose answers are known analytically."""

import numpy as np
import pytest

from src import metrics


def test_gaussian_has_no_tails_skew_or_acf():
    w = np.random.default_rng(0).normal(size=(2000, 64))
    s = metrics.summarize(w, reference=w)
    assert abs(s["excess_kurtosis"]) < 0.1
    assert abs(s["skew"]) < 0.05
    assert s["acf_r_mean_abs"] < s["acf_band"]
    assert abs(s["acf_abs_mean"]) < 0.01
    assert s["wasserstein_to_real_test"] == 0


def test_student_t_kurtosis():
    # Student-t with nu=10 has excess kurtosis 6/(nu-4) = 1.
    w = np.random.default_rng(1).standard_t(10, size=(20000, 64))
    assert metrics.summarize(w)["excess_kurtosis"] == pytest.approx(1.0, abs=0.15)


def test_ar1_acf_matches_phi_powers():
    rng = np.random.default_rng(2)
    phi, n, L = 0.6, 4000, 64
    x = np.zeros((n, L))
    x[:, 0] = rng.normal(size=n) / np.sqrt(1 - phi**2)
    for t in range(1, L):
        x[:, t] = phi * x[:, t - 1] + rng.normal(size=n)
    acf = metrics.pooled_acf(x, max_lag=3)
    np.testing.assert_allclose(acf, [phi, phi**2, phi**3], atol=0.02)


def test_volatility_clustering_detected():
    # A constant random vol per window gives |r| a positive ACF, raw r none.
    rng = np.random.default_rng(3)
    vol = np.exp(rng.normal(0, 0.7, size=(3000, 1)))
    w = vol * rng.normal(size=(3000, 64))
    s = metrics.summarize(w)
    assert s["acf_abs_mean"] > 0.1
    assert s["acf_r_mean_abs"] < 0.02


def test_acf_never_pairs_across_windows():
    # Each window is constant but alternates sign between windows: if lag products spanned
    # window boundaries the lag-1 ACF would be pulled below 1.
    w = np.repeat(np.array([[1.0], [-1.0]] * 50), 64, axis=1)
    assert metrics.pooled_acf(w, max_lag=1)[0] == pytest.approx(1.0)
