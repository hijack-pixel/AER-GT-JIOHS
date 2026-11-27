"""Statistics of the ablation analyses (Table 3 and Sec. 3.2).

For the full model and every ablation, the per-animal metrics are summarised as
mean +/- SEM over the held-out animals. Each ablation is compared with the full
model on the same held-out animals:

- the mean paired difference (ablation minus full model);
- its 95% percentile-bootstrap confidence interval (the paired differences are
  resampled over animals; ``BOOTSTRAP_RESAMPLES`` resamples, seed 42);
- a one-sided paired sign-flip permutation test (all 2**n sign assignments) for
  a decrease;
- Holm correction across the three Recovery-F1 contrasts.
"""
from __future__ import annotations

import csv
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from aergt.evaluation.metrics import METRIC_NAMES

BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 42
PRIMARY_METRIC = "recovery_f1"


def _load_subject_metrics(run_dir: Path, model_name: str) -> pd.DataFrame:
    path = run_dir / "evaluation" / "subject_metrics_for_stats.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing subject metrics for {model_name}: {path}")
    df = pd.read_csv(path)
    missing = [metric for metric in METRIC_NAMES if metric not in df.columns]
    if missing:
        raise ValueError(f"{path} lacks the column(s) {missing}")
    df["model"] = model_name
    return df


def _paired_permutation_pvalue(deltas: np.ndarray, *, alternative: str = "less") -> float:
    """Exact paired sign-flip test on the mean of the paired differences."""
    deltas = np.asarray(deltas, dtype=float)
    deltas = deltas[np.isfinite(deltas)]
    if deltas.size == 0 or deltas.size > 20:
        return float("nan")
    observed = float(deltas.mean())
    null = np.array([float((deltas * np.array(signs)).mean()) for signs in product((-1.0, 1.0), repeat=deltas.size)])
    if alternative == "less":
        count = int(np.sum(null <= observed + 1e-12))
    elif alternative == "greater":
        count = int(np.sum(null >= observed - 1e-12))
    else:
        count = int(np.sum(np.abs(null) >= abs(observed) - 1e-12))
    return float(count / null.size)


def bootstrap_ci_mean(values: Iterable[float], *, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """95% percentile-bootstrap interval of the mean."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, arr.size, size=(resamples, arr.size))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _mean_sem(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(arr.mean())
    sem = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return mean, sem


def _holm_bonferroni(p_values: list[float]) -> list[float]:
    ordered = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [float("nan")] * len(p_values)
    running_max = 0.0
    for rank, idx in enumerate(ordered):
        running_max = max(running_max, (len(p_values) - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def compare_ablation_runs(repo_root: str | Path) -> list[dict[str, object]]:
    root = Path(repo_root).resolve()
    run_dirs = {
        "full_model": root / "outputs" / "full_model",
        "no_pretraining": root / "outputs" / "ablation_no_pretraining",
        "no_aer_regularization": root / "outputs" / "ablation_no_aer_regularization",
        "no_topology": root / "outputs" / "ablation_no_topology",
    }

    tables = {name: _load_subject_metrics(path, name) for name, path in run_dirs.items()}
    full = tables["full_model"].set_index("mouse_id")
    rows: list[dict[str, object]] = []

    # ---------- descriptive summaries ----------
    for model_name, table in tables.items():
        for metric in METRIC_NAMES:
            mean, sem = _mean_sem(table[metric])
            rows.append({"comparison": "descriptive", "model": model_name, "metric": metric,
                         "n_subjects": int(len(table)), "mean": mean, "sem": sem})

    # ---------- paired contrasts against the full model ----------
    primary_rows: list[dict[str, object]] = []
    for model_name in ["no_pretraining", "no_aer_regularization", "no_topology"]:
        other = tables[model_name].set_index("mouse_id")
        if set(full.index) != set(other.index):
            raise ValueError(f"{model_name} and the full model were not evaluated on the same animals")
        for metric in METRIC_NAMES:
            delta = (other[metric] - full.loc[other.index, metric]).to_numpy(dtype=float)
            low, high = bootstrap_ci_mean(delta)
            row: dict[str, object] = {
                "comparison": "ablation_minus_full",
                "model": model_name,
                "metric": metric,
                "n_subjects": int(delta.size),
                "delta_mean": float(np.mean(delta)),
                "ci95_low": low,
                "ci95_high": high,
                "p_less": _paired_permutation_pvalue(delta, alternative="less"),
            }
            rows.append(row)
            if metric == PRIMARY_METRIC:
                primary_rows.append(row)

    # Holm correction within the family reported in the manuscript: the three
    # Recovery-F1 contrasts of the ablations against the full model. The other
    # p values are nominal.
    for row, p_adj in zip(primary_rows, _holm_bonferroni([float(row["p_less"]) for row in primary_rows])):
        row["p_holm"] = p_adj
        row["holm_family"] = f"{PRIMARY_METRIC}_ablations"
    for row in rows:
        row.setdefault("p_holm", "")
        row.setdefault("holm_family", "none_nominal")

    out_dir = root / "outputs" / "ablation_statistics"
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with (out_dir / "ablation_statistics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows
