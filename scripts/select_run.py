"""Pick the diffusion run to evaluate, using TRAIN-period data only.

The test holdout is never consulted here, so choosing among runs can't leak test
information into the reported numbers. Rule (fixed before looking at any run):

    score = W1(samples, real_train) + |mean ACF(|r|) samples - mean ACF(|r|) real_train|

Lower is better. Usage: python scripts/select_run.py r1_... r2_... (writes runs/selection.json)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src import data, metrics


def main() -> None:
    names = sys.argv[1:]
    ref = data.load_dataset()["train_eval"]
    ref_acf = metrics.summarize(ref)["acf_abs_mean"]
    rows = {}
    for n in names:
        s = metrics.summarize(np.load(data.ROOT / "runs" / n / "samples.npy"), reference=ref)
        w1 = s["wasserstein_to_real_test"]  # here the reference is real_train, see above
        rows[n] = {
            "w1_to_real_train": w1,
            "acf_abs_mean": s["acf_abs_mean"],
            "std": s["std"],
            "excess_kurtosis": s["excess_kurtosis"],
            "score": w1 + abs(s["acf_abs_mean"] - ref_acf),
        }
    best = min(rows, key=lambda n: rows[n]["score"])
    out = {"rule": "W1 to real_train + |acf_abs_mean gap to real_train|", "real_train_acf_abs_mean": ref_acf,
           "runs": rows, "selected": best}
    (data.ROOT / "runs" / "selection.json").write_text(json.dumps(out, indent=2))
    for n, r in sorted(rows.items(), key=lambda kv: kv[1]["score"]):
        print(f"{n:22s} score {r['score']:.3f}  W1 {r['w1_to_real_train']:.3f}  acf|r| {r['acf_abs_mean']:.3f}  "
              f"std {r['std']:.3f}  kurt {r['excess_kurtosis']:.2f}")
    print("selected:", best)


if __name__ == "__main__":
    main()
