"""Frame-level motion and stability quantification (Sec. 2.10 / Sec. 3.5).

Implements exactly the computation described in the manuscript:

* for **each 20-s window** (200 frames, 4-s stride - identical to the
  graph-sample windowing of Sec. 2.2) across all recordings of the five
  ictal-transition animals:
    - a **frame-difference motion index**: the mean absolute difference
      between consecutive raw frames;
    - a **consecutive-frame correlation**: the Pearson correlation between
      adjacent raw frames;
* per-window metrics are joined with the corresponding **out-of-fold
  decoding predictions** to test whether movement artifacts or recording
  instability could account for the reported state decoding (e.g. motion of
  incorrectly vs correctly decoded windows);
* the summary reports the minimum consecutive-frame correlation, per-animal
  per-state median motion, and the Friedman test of the median motion index
  across states at the animal level.

Usage:
    python scripts/compute_motion_quality.py \
        --recordings-dir <root of per-animal raw TIFF folders> \
        --oof-predictions outputs/full_model/evaluation/oof_predictions.csv \
        --out-dir outputs/motion_quality

The recordings directory is expected to contain one sub-folder per animal
(e.g. ``mouse1/``, ``mouse2/`` ...), each holding multi-frame TIFF stacks.
All numeric outputs are written under ``outputs/`` and are not versioned.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FPS, WIN, STEP = 10, 200, 40           # identical to Sec. 2.2 windowing


def read_tiff_stack(path: Path) -> np.ndarray:
    """Read a multi-frame TIFF recording as (frames, H, W) float32."""
    try:
        import tifffile
    except ImportError as exc:  # pragma: no cover
        raise ImportError("compute_motion_quality requires 'tifffile' (pip install tifffile).") from exc
    stack = tifffile.imread(str(path))
    stack = np.asarray(stack, dtype=np.float32)
    if stack.ndim == 2:                       # single frame file
        stack = stack[None, ...]
    if stack.ndim == 4:                       # (frames, H, W, channels) -> gray
        stack = stack.mean(axis=-1)
    return stack


def label_from_filename(name: str) -> int:
    """Reference-state labelling rule of Sec. 2.1 (same as preprocess/matlab
    aergt_label_from_filename.m): early post-injection recordings without
    seizure markers are baseline; seizure-marked or mid-session recordings
    are ictal; late recordings are recovery."""
    f = name.lower()
    if ("10min" in f) or ("20min" in f) or ("30min" in f):
        return 1 if "seizure" in f else 0
    if any(k in f for k in ("40min", "35.tif", "45.tif", "1h", "1.5h", "005")):
        return 1
    if any(k in f for k in ("2h", "2.5h", "3h", "3.5h", "4h", "recovery",
                            "60min", "90min", "150min", "180", "210", "240")):
        return 2
    return -1


def minutes_from_filename(name: str) -> float:
    """Post-injection time in minutes encoded in the recording filename."""
    m = re.match(r"(\d+(?:\.\d+)?)\s*min", name, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.match(r"(\d+(?:\.\d+)?)\s*h", name, flags=re.IGNORECASE)
    if m:
        return float(m.group(1)) * 60.0
    m = re.match(r"(\d+)", name)
    return float(m.group(1)) if m else float("nan")


def window_motion_metrics(frames: np.ndarray) -> tuple[float, float]:
    """(frame-difference motion index, median consecutive-frame correlation)
    for one window of raw frames."""
    flat = frames.reshape(frames.shape[0], -1)
    diffs = np.abs(np.diff(flat, axis=0)).mean(axis=1)      # |f_{t+1} - f_t|
    motion_index = float(diffs.mean())
    corrs = []
    for t in range(flat.shape[0] - 1):
        a, b = flat[t], flat[t + 1]
        if a.std() > 0 and b.std() > 0:
            corrs.append(float(np.corrcoef(a, b)[0, 1]))
    frame_corr = float(np.median(corrs)) if corrs else float("nan")
    return motion_index, frame_corr


def compute_all_windows(recordings_dir: Path, mouse_pattern: str = "mouse*"):
    """Yield one row per 20-s window across all recordings of all animals."""
    rows = []
    for mouse_dir in sorted(recordings_dir.glob(mouse_pattern)):
        mouse = mouse_dir.name
        for tif in sorted(mouse_dir.glob("*.tif")):
            stack = read_tiff_stack(tif)
            n_frames = stack.shape[0]
            state = label_from_filename(tif.name)
            minutes = minutes_from_filename(tif.name)
            win_idx = 0
            for start in range(0, n_frames - WIN + 1, STEP):
                motion, corr = window_motion_metrics(stack[start:start + WIN])
                rows.append({
                    "mouse_id": mouse,
                    "file": tif.name,
                    "win": win_idx,
                    "state": state,
                    "minutes": minutes,
                    "motion_diff": motion,
                    "frame_corr": corr,
                })
                win_idx += 1
    return rows


def join_with_predictions(window_rows, oof_csv: Path):
    """Join per-window motion metrics with the out-of-fold predictions of
    the trained model (columns: mouse_id / source_file / window_index /
    true_label_name / pred_label_name / correct)."""
    import pandas as pd

    df = pd.DataFrame(window_rows)
    oof = pd.read_csv(oof_csv)

    def norm_mouse(s):
        digits = re.search(r"(\d+)", str(s))
        return int(digits.group(1)) if digits else None

    def stem_of(path):
        base = Path(str(path)).name.replace(".mat", "")
        m = re.match(r".*?_(.+)_Win\d+$", base)
        if not m:
            return None
        return re.sub(r"[^a-z0-9]", "", m.group(1).lower())

    df["mouse_num"] = df["mouse_id"].map(norm_mouse)
    oof["mouse_num"] = oof["mouse_id"].map(norm_mouse)
    oof["stem"] = oof["source_file"].map(stem_of)
    df["stem"] = df["file"].map(lambda f: re.sub(r"[^a-z0-9]", "", Path(f).stem.lower()))

    merged = df.merge(oof, on=["mouse_num", "stem", "win"], how="inner")
    return merged, df


def summarize(merged, df) -> dict:
    """Summary statistics reported in Sec. 3.5."""
    from scipy.stats import friedmanchisquare

    out = {
        "n_recordings": int(df["file"].nunique()),
        "n_windows": int(len(df)),
        "min_frame_corr": float(df["frame_corr"].min()),
        "min_frame_corr_by_state": {
            str(k): float(v) for k, v in df.groupby("state")["frame_corr"].min().items()
        },
    }
    if not merged.empty and "correct" in merged:
        correct = merged.loc[merged["correct"].astype(bool), "motion_diff"]
        incorrect = merged.loc[~merged["correct"].astype(bool), "motion_diff"]
        out["error_analysis"] = {
            "median_motion_correct": float(correct.median()),
            "median_motion_incorrect": float(incorrect.median()),
            "n_correct": int(len(correct)),
            "n_incorrect": int(len(incorrect)),
        }
    # animal-level Friedman test on the per-animal per-state median motion
    medians = df.groupby(["mouse_num", "state"])["motion_diff"].median().unstack()
    medians = medians.dropna()
    if medians.shape == (5, 3):
        stat, p = friedmanchisquare(*[medians[c] for c in medians.columns])
        out["friedman_motion_by_state"] = {"chi2": float(stat), "p": float(p),
                                           "n_animals": int(len(medians))}
        out["per_animal_median_motion"] = {
            str(idx): {str(c): float(v) for c, v in row.items()}
            for idx, row in medians.iterrows()
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Motion / recording-quality metrics (Sec. 2.10).")
    parser.add_argument("--recordings-dir", required=True,
                        help="root folder with one sub-folder of raw TIFF stacks per animal")
    parser.add_argument("--oof-predictions", default=str(
        ROOT / "outputs" / "full_model" / "evaluation" / "oof_predictions.csv"))
    parser.add_argument("--out-dir", default=str(ROOT / "outputs" / "motion_quality"))
    args = parser.parse_args()

    rows = compute_all_windows(Path(args.recordings_dir))
    if not rows:
        raise SystemExit(f"no recordings found under {args.recordings_dir}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "per_window_motion.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    oof_csv = Path(args.oof_predictions)
    merged, df = (join_with_predictions(rows, oof_csv)
                  if oof_csv.exists() else (None, __import__("pandas").DataFrame(rows)))
    if merged is not None:
        merged.to_csv(out_dir / "windows_joined_predictions.csv", index=False)

    summary = summarize(merged if merged is not None else df, df)
    (out_dir / "motion_quality_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2)[:800])
    print(f"outputs -> {out_dir}")


if __name__ == "__main__":
    main()
