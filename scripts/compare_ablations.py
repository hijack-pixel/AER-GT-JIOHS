from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aergt.evaluation.ablation_stats import compare_ablation_runs


def main() -> None:
    rows = compare_ablation_runs(ROOT)
    print(f"Wrote {len(rows)} rows to outputs/ablation_statistics/ablation_statistics.csv")


if __name__ == "__main__":
    main()

