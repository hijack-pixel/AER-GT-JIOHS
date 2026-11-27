from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aergt.figures.simple_plots import plot_state_probability_trajectories  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw Fig. 5 (stage-wise aligned mean probability trajectories of the held-out recordings) "
                    "from the out-of-fold predictions of a run.")
    parser.add_argument("--run-dir", default=str(ROOT / "outputs" / "full_model"))
    parser.add_argument("--points", type=int, default=100, help="positions each recording is interpolated to")
    args = parser.parse_args()
    path = plot_state_probability_trajectories(args.run_dir, points=args.points)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
