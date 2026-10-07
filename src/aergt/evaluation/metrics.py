from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize


def safe_macro_auc(labels: np.ndarray, probs: np.ndarray, num_classes: int) -> float:
    present = sorted(set(labels.tolist()))
    if len(present) < 2:
        return float("nan")
    y_bin = label_binarize(labels, classes=list(range(num_classes)))
    try:
        return float(roc_auc_score(y_bin, probs, average="macro"))
    except ValueError:
        return float("nan")


def classification_metrics(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray, num_classes: int) -> dict[str, object]:
    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_auc": safe_macro_auc(labels, probs, num_classes),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, preds, labels=list(range(num_classes))).tolist(),
    }
    for class_idx, name in enumerate(["baseline", "seizure", "recovery"][:num_classes]):
        metrics[f"{name}_f1"] = float(f1_score(labels, preds, labels=[class_idx], average="macro", zero_division=0))
    return metrics


def mean_std_ci(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0, "ci95_low": float("nan"), "ci95_high": float("nan")}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    half_width = 1.96 * std / np.sqrt(arr.size) if arr.size > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "n": int(arr.size),
        "ci95_low": float(mean - half_width),
        "ci95_high": float(mean + half_width),
    }

