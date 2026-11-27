"""Frame-level motion and stability quantification (Sec. 2.10 / Sec. 3.5).

For every recording named in the out-of-fold predictions of a run (the window
with index k covers frames (k - 1) * 40 + 1 to (k - 1) * 40 + 200, i.e. 20 s at
10 Hz with the 4-s stride of Sec. 2.2) the raw frames are read and the two
consecutive-frame quantities are computed for every pair of neighbouring frames
of the recording:

- the frame-difference motion index: the mean absolute difference between the
  two frames (raw A/D levels);
- the consecutive-frame correlation: the Pearson correlation between the two
  frames.

Each window is then summarized by the mean of both quantities over its frame
pairs (200 pairs; 199 for the first window of a recording, whose first frame
has no predecessor).

Usage:
    python scripts/compute_motion_quality.py --config configs/full_model.yaml \
        --recordings-dir <root with one sub-folder of raw TIFF recordings per animal>

Animal sub-folders are matched to the dataset identifiers by comparing their
lower-case letters and digits (``Mouse 1`` matches ``mouse1``); an explicit table
can be given instead with ``--animal-folders`` (CSV with the columns ``mouse_id``
and ``folder``).

Requires the optional ``tifffile`` package.

Output:
    outputs/<run>/motion_quality/
        per_window_motion.csv             mouse_id, original_file, window_index, label, metrics
        windows_joined_predictions.csv    the same, joined with the out-of-fold predictions
        per_animal_median_motion.csv
        motion_quality_summary.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aergt.config import load_config  # noqa: E402


def read_tiff_stack(path: Path) -> np.ndarray:
    """Read a multi-frame TIFF recording as (frames, H, W) float32."""
    try:
        import tifffile
    except ImportError as exc:  # pragma: no cover
        raise ImportError("compute_motion_quality requires 'tifffile' (pip install tifffile).") from exc
    stack = np.asarray(tifffile.imread(str(path)), dtype=np.float32)
    if stack.ndim == 2:                       # single frame file
        stack = stack[None, ...]
    if stack.ndim == 4:                       # (frames, H, W, channels) -> gray
        stack = stack.mean(axis=-1)
    return stack


def consecutive_frame_metrics(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per pair of neighbouring frames: mean absolute difference and Pearson correlation.

    Entry t refers to the frames t and t + 1 (0-based).
    """
    flat = frames.reshape(frames.shape[0], -1).astype(np.float64)
    diffs = np.abs(np.diff(flat, axis=0)).mean(axis=1)
    a, b = flat[:-1], flat[1:]
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    denom = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        corrs = np.where(denom > 0, (a * b).sum(axis=1) / denom, np.nan)
    return diffs, corrs


def window_metrics(diffs: np.ndarray, corrs: np.ndarray, start: int, win_frames: int) -> dict[str, float]:
    """Means over the frame pairs of one window (start frame, 0-based), including
    the pair that leads into the window when the window does not start the recording."""
    lo = max(start - 1, 0)
    hi = start + win_frames - 1
    window_diffs, window_corrs = diffs[lo:hi], corrs[lo:hi]
    return {"motion_diff": float(window_diffs.mean()),
            "frame_corr": float(np.nanmean(window_corrs)),
            "frame_corr_min": float(np.nanmin(window_corrs))}


def folder_for(mouse_id: str, folders: list[Path]) -> Path:
    """Sub-folder whose name differs from the identifier only in case and non-alphanumerics."""
    key = re.sub(r"[^a-z0-9]", "", mouse_id.lower())
    for folder in folders:
        if re.sub(r"[^a-z0-9]", "", folder.name.lower()) == key:
            return folder
    raise SystemExit(f"no sub-folder of the recordings directory matches the identifier {mouse_id!r}; "
                     "pass --animal-folders")


def main() -> None:
    parser = argparse.ArgumentParser(description="Motion and recording-quality metrics (Sec. 2.10).")
    parser.add_argument("--config", default=str(ROOT / "configs" / "full_model.yaml"))
    parser.add_argument("--recordings-dir", required=True, help="root folder with one sub-folder of TIFF recordings per animal")
    parser.add_argument("--animal-folders", default="", help="CSV with the columns mouse_id and folder")
    parser.add_argument("--win-frames", type=int, default=200)
    parser.add_argument("--step-frames", type=int, default=40)
    parser.add_argument("--out-dir", default="", help="default: outputs/<run>/motion_quality")
    args = parser.parse_args()

    cfg = load_config(args.config)
    classes = [str(name).lower() for name in cfg["model"]["class_names"]]
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / cfg["project"]["output_dir"] / "motion_quality"
    out_dir.mkdir(parents=True, exist_ok=True)

    oof = pd.read_csv(ROOT / cfg["project"]["output_dir"] / "evaluation" / "oof_predictions.csv")
    oof["mouse_id"] = oof["mouse_id"].astype(str).str.strip()
    if "original_file" not in oof.columns or oof["original_file"].isna().any() or (oof["original_file"].astype(str) == "").any():
        raise SystemExit("the window samples carry no OriginalFile; the recordings cannot be located")
    oof["original_file"] = oof["original_file"].astype(str)

    recordings_dir = Path(args.recordings_dir)
    explicit = {}
    if args.animal_folders:
        table = pd.read_csv(args.animal_folders)
        explicit = {str(mouse).strip(): folder for mouse, folder in zip(table["mouse_id"], table["folder"])}
    folders = sorted(path for path in recordings_dir.iterdir() if path.is_dir())

    rows = []
    cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for row in oof.itertuples(index=False):
        key = (row.mouse_id, row.original_file)
        if key not in cache:
            folder = recordings_dir / explicit[row.mouse_id] if row.mouse_id in explicit else folder_for(row.mouse_id, folders)
            path = folder / row.original_file
            if not path.exists():
                raise SystemExit(f"recording not found: {path}")
            cache[key] = consecutive_frame_metrics(read_tiff_stack(path))
        diffs, corrs = cache[key]
        start = (int(row.window_index) - 1) * args.step_frames
        if start + args.win_frames > len(diffs) + 1:
            raise SystemExit(f"window {row.window_index} of {key} exceeds the {len(diffs) + 1} frames of the recording")
        rows.append({"mouse_id": row.mouse_id, "original_file": row.original_file, "window_index": int(row.window_index),
                     "label": int(row.label), **window_metrics(diffs, corrs, start, args.win_frames)})

    motion = pd.DataFrame(rows)
    motion.to_csv(out_dir / "per_window_motion.csv", index=False)
    joined = oof.merge(motion.drop(columns=["label"]), on=["mouse_id", "original_file", "window_index"], validate="one_to_one")
    joined["state"] = [classes[int(value)] for value in joined["label"]]
    joined["correct"] = (joined["pred"] == joined["label"]).astype(int)
    joined.to_csv(out_dir / "windows_joined_predictions.csv", index=False)

    summary: dict[str, object] = {
        "n_recordings": int(len(cache)),
        "n_windows": int(len(motion)),
        "min_frame_corr": float(motion["frame_corr"].min()),
        "min_frame_corr_min": float(motion["frame_corr_min"].min()),
        "min_frame_corr_by_state": {classes[int(k)]: float(v) for k, v in motion.groupby("label")["frame_corr"].min().items()},
    }
    correct = joined.loc[joined["correct"] == 1, "motion_diff"]
    incorrect = joined.loc[joined["correct"] == 0, "motion_diff"]
    summary["error_analysis"] = {
        "median_motion_correct": float(correct.median()), "n_correct": int(len(correct)),
        "median_motion_incorrect": float(incorrect.median()) if len(incorrect) else None, "n_incorrect": int(len(incorrect)),
    }
    medians = motion.groupby(["mouse_id", "label"])["motion_diff"].median().unstack()
    medians.to_csv(out_dir / "per_animal_median_motion.csv")
    if medians.shape[1] == len(classes) and medians.shape[0] >= 3 and not medians.isna().any().any():
        from scipy.stats import friedmanchisquare

        stat, p = friedmanchisquare(*[medians[column] for column in medians.columns])
        summary["friedman_motion_by_state"] = {"chi2": float(stat), "p": float(p), "n_animals": int(len(medians))}
    (out_dir / "motion_quality_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"outputs -> {out_dir}")


if __name__ == "__main__":
    main()
