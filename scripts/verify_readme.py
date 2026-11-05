"""Assert every headline figure in README.md appears in docs/phase08_results.json.

    python scripts/verify_readme.py

Each figure is rendered from the JSON to 3 decimals and must appear verbatim in the README.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (source, metric) pairs the README quotes.
HEADLINE = [
    ("real_test", "acf_abs_mean"),
    ("gbm", "acf_abs_mean"),
    ("diffusion", "acf_abs_mean"),
    ("real_test", "excess_kurtosis"),
    ("gbm", "excess_kurtosis"),
    ("diffusion", "excess_kurtosis"),
    ("real_test", "skew"),
    ("diffusion", "skew"),
    ("gbm", "wasserstein_to_real_test"),
    ("diffusion", "wasserstein_to_real_test"),
    ("diffusion", "std"),
]


def main() -> int:
    results = json.loads((ROOT / "docs" / "phase08_results.json").read_text())
    readme = (ROOT / "README.md").read_text()
    missing = []
    for src, metric in HEADLINE:
        text = f"{results['sources'][src][metric]:.3f}"
        if text not in readme:
            missing.append(f"{src}.{metric} = {text}")
    failed = [n for n, ok in results["criteria"].items() if not ok]
    if failed and "fail" not in readme.lower():
        missing.append(f"criteria failed ({', '.join(failed)}) but README never says so")
    if missing:
        print("README is out of sync with docs/phase08_results.json:")
        print("\n".join(f"  - {m}" for m in missing))
        return 1
    print(f"OK: {len(HEADLINE)} headline figures found in README.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
