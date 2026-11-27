from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aergt.data.samples import apply_scaler, load_samples, make_loader
from aergt.data.scalers import load_standardizer
from aergt.data.splits import subject_level_folds, subset_by_subjects
from aergt.evaluation.metrics import METRIC_NAMES, classification_metrics, mean_std_ci
from aergt.models import AERGT
from aergt.training.runner import _model_kwargs
from aergt.utils.runtime import configure_torch, resolve_device, set_seed, write_json


def _autocast(device: torch.device, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint format: {path}")


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


@torch.no_grad()
def predict(model: AERGT, loader, *, device: torch.device, amp: bool) -> dict[str, object]:
    model.eval()
    probs_all: list[np.ndarray] = []
    preds_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    attn_all: list[np.ndarray] = []
    metadata: dict[str, list[object]] = {"mouse_id": [], "source_file": [], "original_file": [], "window_index": []}

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        adj = batch["adj"].to(device, non_blocking=True)
        labels = batch["y"].numpy()
        with _autocast(device, amp and device.type == "cuda"):
            logits, alpha = model(x, adj, return_attention=True)
            probs = torch.softmax(logits, dim=1)
        probs_np = probs.detach().cpu().numpy()
        probs_all.append(probs_np)
        preds_all.append(probs_np.argmax(axis=1))
        labels_all.append(labels)
        attn_all.append(alpha.detach().cpu().numpy())
        for key in metadata:
            metadata[key].extend(batch[key])

    return {
        "labels": np.concatenate(labels_all),
        "preds": np.concatenate(preds_all),
        "probs": np.vstack(probs_all),
        "attention": np.vstack(attn_all),
        "metadata": metadata,
    }


def evaluate_from_config(cfg: dict[str, Any], *, repo_root: str | Path) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    output_dir = (repo_root / cfg["project"]["output_dir"]).resolve()
    artifact_dir = (repo_root / cfg.get("artifacts", {}).get("artifact_dir", cfg["project"]["output_dir"])).resolve()
    checkpoint_dir = artifact_dir / "checkpoints"
    scaler_dir = artifact_dir / "scalers"
    eval_dir = output_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(cfg["project"]["seed"]), deterministic=bool(cfg["hardware"]["deterministic"]))
    configure_torch(tf32=bool(cfg["hardware"]["tf32"]))
    device = resolve_device(str(cfg["hardware"]["device"]))

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    hardware_cfg = cfg["hardware"]
    labeled_samples = load_samples(
        repo_root / data_cfg["labeled_dir"],
        labeled=True,
        num_nodes=int(model_cfg["num_nodes"]),
        in_channels=int(model_cfg["in_channels"]),
        use_abs_adj=bool(data_cfg["use_abs_adjacency"]),
    )
    folds = subject_level_folds(labeled_samples, int(cfg["training"]["num_folds"]))

    fold_rows: list[dict[str, object]] = []
    subject_rows: list[dict[str, object]] = []
    oof_rows: list[dict[str, object]] = []
    aggregate_cm = np.zeros((int(model_cfg["num_classes"]), int(model_cfg["num_classes"])), dtype=np.int64)
    fold_summaries: list[dict[str, object]] = []

    for fold_idx, _, test_subjects in folds:
        scaler = load_standardizer(scaler_dir / f"fold_{fold_idx}_scaler.pkl")
        test_raw = subset_by_subjects(labeled_samples, test_subjects)
        test_samples = apply_scaler(test_raw, scaler)
        loader = make_loader(
            test_samples,
            batch_size=int(cfg["training"]["finetune_batch_size"]),
            shuffle=False,
            num_workers=int(hardware_cfg["num_workers"]),
            pin_memory=bool(hardware_cfg["pin_memory"]) and device.type == "cuda",
        )

        model = AERGT(**_model_kwargs(cfg)).to(device)
        model.load_state_dict(_load_state_dict(checkpoint_dir / f"fold_{fold_idx}_final_model.pth"))
        pred = predict(model, loader, device=device, amp=bool(hardware_cfg["amp"]))

        labels = pred["labels"]
        preds = pred["preds"]
        probs = pred["probs"]
        metrics = classification_metrics(labels, preds, probs, int(model_cfg["num_classes"]))
        aggregate_cm += np.asarray(metrics["confusion_matrix"], dtype=np.int64)

        fold_row = {
            "fold": fold_idx,
            "held_out_subjects": ";".join(test_subjects),
            "n_windows": int(labels.size),
            **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
        }
        fold_rows.append(fold_row)
        fold_summaries.append({"fold": fold_idx, "metrics": metrics})

        metadata = pred["metadata"]
        for i in range(labels.size):
            row = {
                "fold": fold_idx,
                "mouse_id": metadata["mouse_id"][i],
                "source_file": metadata["source_file"][i],
                "original_file": metadata["original_file"][i],
                "window_index": metadata["window_index"][i],
                "label": int(labels[i]),
                "pred": int(preds[i]),
            }
            for class_idx, class_name in enumerate(model_cfg["class_names"]):
                row[f"prob_{class_name.lower()}"] = float(probs[i, class_idx])
            oof_rows.append(row)

        for subject in test_subjects:
            idx = np.asarray([m == subject for m in metadata["mouse_id"]], dtype=bool)
            sub_metrics = classification_metrics(labels[idx], preds[idx], probs[idx], int(model_cfg["num_classes"]))
            subject_rows.append(
                {
                    "fold": fold_idx,
                    "mouse_id": subject,
                    "n_windows": int(idx.sum()),
                    **{k: v for k, v in sub_metrics.items() if k != "confusion_matrix"},
                }
            )

        np.savez_compressed(
            eval_dir / f"fold_{fold_idx}_interpretability.npz",
            labels=labels,
            preds=preds,
            probs=probs,
            attention=pred["attention"],
            mouse_id=np.asarray(metadata["mouse_id"], dtype=object),
            source_file=np.asarray(metadata["source_file"], dtype=object),
            original_file=np.asarray(metadata["original_file"], dtype=object),
            window_index=np.asarray(metadata["window_index"], dtype=object),
        )

    metric_names = METRIC_NAMES
    summary_metrics = {
        metric: mean_std_ci([float(row[metric]) for row in fold_rows if metric in row])
        for metric in metric_names
    }

    _write_csv(eval_dir / "fold_metrics.csv", fold_rows, ["fold", "held_out_subjects", "n_windows", *metric_names])
    _write_csv(eval_dir / "subject_metrics_for_stats.csv", subject_rows, ["fold", "mouse_id", "n_windows", *metric_names])
    _write_csv(
        eval_dir / "oof_predictions.csv",
        oof_rows,
        ["fold", "mouse_id", "source_file", "original_file", "window_index", "label", "pred"]
        + [f"prob_{name.lower()}" for name in model_cfg["class_names"]],
    )

    cm_rows = []
    for true_idx, row in enumerate(aggregate_cm):
        cm_rows.append({"true_label": int(true_idx), **{f"pred_{i}": int(v) for i, v in enumerate(row)}})
    _write_csv(eval_dir / "aggregate_confusion_matrix.csv", cm_rows, ["true_label", *[f"pred_{i}" for i in range(int(model_cfg["num_classes"]))]])

    summary = {
        "mode": "evaluate",
        "output_dir": str(output_dir),
        "artifact_dir": str(artifact_dir),
        "summary_metrics": summary_metrics,
        "fold_results": fold_summaries,
        "aggregate_confusion_matrix": aggregate_cm.tolist(),
    }
    write_json(summary, eval_dir / "cv_summary.json")
    return summary
