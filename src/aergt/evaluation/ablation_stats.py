from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _load_subject_metrics(run_dir: Path, model_name: str) -> pd.DataFrame:
    path = run_dir / "evaluation" / "subject_metrics_for_stats.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing subject metrics for {model_name}: {path}")
    df = pd.read_csv(path)
    df["model"] = model_name
    return df


def _paired_permutation_pvalue(deltas: np.ndarray, *, alternative: str = "less") -> float:
    deltas = np.asarray(deltas, dtype=float)
    deltas = deltas[np.isfinite(deltas)]
    if deltas.size == 0:
        return float("nan")
    observed = deltas.mean()
    count = 0
    total = 2 ** deltas.size
    for mask in range(total):
        signs = np.ones(deltas.size)
        for i in range(deltas.size):
            if mask & (1 << i):
                signs[i] = -1.0
        stat = (deltas * signs).mean()
        if alternative == "less":
            count += stat <= observed + 1e-12
        elif alternative == "greater":
            count += stat >= observed - 1e-12
        else:
            count += abs(stat) >= abs(observed) - 1e-12
    return float(count / total)


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
    metrics = ["accuracy", "macro_auc", "macro_f1", "recovery_f1"]

    for model_name, table in tables.items():
        for metric in metrics:
            mean, sem = _mean_sem(table[metric])
            rows.append({"comparison": "descriptive", "model": model_name, "metric": metric, "mean": mean, "sem": sem})

    for model_name in ["no_pretraining", "no_aer_regularization", "no_topology"]:
        other = tables[model_name].set_index("mouse_id")
        shared = full.index.intersection(other.index)
        for metric in metrics:
            delta = other.loc[shared, metric].to_numpy(dtype=float) - full.loc[shared, metric].to_numpy(dtype=float)
            rows.append(
                {
                    "comparison": "ablation_minus_full",
                    "model": model_name,
                    "metric": metric,
                    "delta_mean": float(np.nanmean(delta)),
                    "p_less": _paired_permutation_pvalue(delta, alternative="less"),
                    "n_subjects": int(shared.size),
                }
            )

    # Holm correction within the primary family reported in the manuscript
    # (Recovery-F1, three ablation-vs-full comparisons), as specified in the
    # Methods. The initial release reported these as nominal one-sided p
    # values without applying the stated correction. All other p_less
    # entries remain nominal and are labeled as such.
    primary_metric = "recovery_f1"
    primary_rows = [
        row
        for row in rows
        if row.get("comparison") == "ablation_minus_full" and row["metric"] == primary_metric
    ]
    adjusted = _holm_bonferroni([row["p_less"] for row in primary_rows])
    for row, p_adj in zip(primary_rows, adjusted):
        row["p_holm"] = p_adj
        row["holm_family"] = f"{primary_metric}_ablations"
    for row in rows:
        if "p_holm" not in row:
            row["p_holm"] = ""
            row["holm_family"] = "none_nominal"

    out_dir = root / "outputs" / "ablation_statistics"
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with (out_dir / "ablation_statistics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return rows

