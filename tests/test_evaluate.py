"""The pass/fail logic that decides the headline claims."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("evaluate", Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py")
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)


def _s(kurt, skew, r, a, w):
    return {"excess_kurtosis": kurt, "skew": skew, "acf_r_mean_abs": r, "acf_abs_mean": a, "wasserstein_to_real_test": w}


def test_criteria_pass_and_fail():
    s = {
        "real_test": _s(8.0, -0.2, 0.05, 0.25, 0.0),
        "gbm": _s(0.0, 0.0, 0.0, 0.0, 0.15),
        "diffusion": _s(6.0, -0.3, 0.01, 0.05, 0.35),
    }
    c = evaluate.criteria(s)
    assert c["vol_clustering_closer_than_gbm"] and c["kurtosis_closer_than_gbm"] and c["skew_closer_than_gbm"]
    assert c["linear_acf_not_above_real"]
    assert not c["wasserstein_below_gbm"]
    s["diffusion"]["excess_kurtosis"] = -5.0
    assert not evaluate.criteria(s)["kurtosis_closer_than_gbm"]
