from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aergt.figures.simple_plots import plot_roi_attention_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the ROI-attention summary figure (Fig. 7 interpretability) from evaluation outputs.")
    parser.add_argument("--run-dir", default=str(ROOT / "outputs" / "full_model"))
    args = parser.parse_args()
    path = plot_roi_attention_summary(args.run_dir)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

