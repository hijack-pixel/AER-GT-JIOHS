"""Shared helpers for the supplementary-analysis scripts."""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

METRICS = ["accuracy", "macro_auc", "macro_f1", "baseline_f1", "seizure_f1", "recovery_f1"]
METRIC_LABELS = ["ACC", "Macro-AUC", "Macro-F1", "Baseline-F1", "Seizure-F1", "Recovery-F1"]
STATE_NAMES = ["Baseline", "Seizure", "Recovery"]


def load_run_config(path: str | Path) -> dict:
    """Load a run configuration given relative to the repository root."""
    from aergt.config import load_config

    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    return load_config(config_path)


def evaluation_dir(cfg: dict) -> Path:
    """Directory that ``run_inference.py`` fills for a run."""
    return ROOT / cfg["project"]["output_dir"] / "evaluation"


def display_map(identifiers) -> dict[str, str]:
    """Map dataset identifiers to manuscript-style labels (``Mouse 1``, ``Mouse 2``, ...)
    in ascending order of the identifiers. The manuscript tables number the animals
    by the same rule."""
    return {str(value): f"Mouse {index}"
            for index, value in enumerate(sorted({str(v) for v in identifiers}), start=1)}


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _read_csv(path: Path, required: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(
            f"Missing input file: {path}\nRun scripts/train_full_model.py and scripts/run_inference.py first.")
    frame = pd.read_csv(path)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SystemExit(f"{path} lacks the required column(s): {missing}")
    return frame


def read_fold_metrics(eval_dir: Path) -> pd.DataFrame:
    return _read_csv(Path(eval_dir) / "fold_metrics.csv", ["fold", "held_out_subjects", "n_windows", *METRICS])


def read_subject_metrics(eval_dir: Path) -> pd.DataFrame:
    return _read_csv(Path(eval_dir) / "subject_metrics_for_stats.csv", ["fold", "mouse_id", "n_windows", *METRICS])


def read_oof_predictions(eval_dir: Path) -> pd.DataFrame:
    return _read_csv(Path(eval_dir) / "oof_predictions.csv",
                     ["fold", "mouse_id", "original_file", "window_index", "label", "pred"])


def read_recording_times(path: str | Path) -> pd.DataFrame:
    """Per-recording metadata with columns ``mouse_id``, ``recording``, ``t_post_inj_min``."""
    times = _read_csv(resolve_path(path), ["mouse_id", "recording", "t_post_inj_min"])
    times["mouse_id"] = times["mouse_id"].astype(str).str.strip()
    times["recording"] = times["recording"].astype(str).str.strip()
    if times.duplicated(["mouse_id", "recording"]).any():
        raise SystemExit("The recording-times CSV must have one row per (mouse_id, recording).")
    return times


def class_names(cfg: dict) -> list[str]:
    return [str(name) for name in cfg["model"]["class_names"]]


def mean_sd(values) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan")
    return float(array.mean()), (float(array.std(ddof=1)) if array.size > 1 else 0.0)


def exact_sign_flip_test(differences, *, alternative: str = "two-sided") -> tuple[int, int]:
    """Exact paired sign-flip test on the mean of paired differences.

    All 2**n sign assignments are enumerated. Returns ``(exceedances, assignments)``;
    the p value is their ratio. ``alternative`` is ``"two-sided"`` (absolute mean at
    least as large as observed), ``"greater"``, or ``"less"``.
    """
    d = np.asarray(differences, dtype=float)
    n = int(d.size)
    if n == 0 or n > 20:
        raise ValueError("the exact test needs 1 to 20 paired differences")
    observed = float(d.mean())
    null = np.array([float((d * np.array(signs)).mean()) for signs in product((-1.0, 1.0), repeat=n)])
    if alternative == "two-sided":
        count = int(np.sum(np.abs(null) >= abs(observed) - 1e-12))
    elif alternative == "greater":
        count = int(np.sum(null >= observed - 1e-12))
    elif alternative == "less":
        count = int(np.sum(null <= observed + 1e-12))
    else:
        raise ValueError(alternative)
    return count, int(null.size)
