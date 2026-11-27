from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aergt.figures.simple_plots import plot_roi_attention_summary  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw Fig. 7 (readout ranking, ROI masking, readout contrasts, interaction map) "
                    "from the interpretability outputs of a run.")
    parser.add_argument("--run-dir", default=str(ROOT / "outputs" / "full_model"))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--roi-labels", default="", help="CSV with a ``label`` column (ROI names in index order)")
    args = parser.parse_args()
    path = plot_roi_attention_summary(args.run_dir, top_k=args.top_k,
                                      roi_labels=Path(args.roi_labels) if args.roi_labels else None)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
