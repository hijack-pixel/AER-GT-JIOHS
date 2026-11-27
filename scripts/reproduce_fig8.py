from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aergt.figures.simple_plots import plot_motion_quality  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw Fig. 8 (recording-quality and motion metrics) from the motion-quality outputs of a run.")
    parser.add_argument("--run-dir", default=str(ROOT / "outputs" / "full_model"))
    parser.add_argument("--recording-times", default="",
                        help="CSV with mouse_id, recording, t_post_inj_min (needed for panel c)")
    args = parser.parse_args()
    path = plot_motion_quality(args.run_dir, recording_times=Path(args.recording_times) if args.recording_times else None)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
