from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aergt.figures.simple_plots import plot_control_evaluation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw Fig. 6 (b-d), the decoding of the pharmacological-control cohort, from the "
                    "control-inference outputs of a run.")
    parser.add_argument("--run-dir", default=str(ROOT / "outputs" / "paper_trained_inference"),
                        help="run directory whose control_inference/ folder is used")
    parser.add_argument("--recording", default="", help="recording of panel (b) as \"<MouseID>|<OriginalFile>\"")
    args = parser.parse_args()
    path = plot_control_evaluation(args.run_dir, recording=args.recording)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
