import os
import csv
import json
from typing import Dict, List, Optional, Sequence

import joblib
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

# Seed-rerun copy of aergft/aergft_infer_stats_only.py (original untouched).
# Imports the seed-123 training module in this same folder, so OUTPUT_DIR
# resolves to aer_gft_cv_outputs_split_seed123 and no original results are touched.
from aergft_train_seed123 import (
    AERGFT,
    CLASS_NAMES,
    DEVICE,
    FINETUNE_BATCH_SIZE,
    LABELED_PATH,
    NUM_CLASSES,
    NUM_FOLDS,
    NUM_NODES,
    OUTPUT_DIR,
    GraphSample,
    apply_scaler_to_samples,
    build_fold_splits,
    get_subjects,
    load_graph_samples,
    make_loader,
    subset_by_subjects,
)

CONTROL_PATH = None
METHOD_NAME = "AER-GT"

ROI_NAMES = [
    "M2L", "M2R",
    "MBL", "MBR",
    "ACL", "ACR",
    "MFL", "MFR",
    "FLL", "FLR",
    "HLL", "HLR",
    "PTL", "PTR",
    "BCL", "BCR",
    "RSL", "RSR",
    "V2L", "V2R",
    "V1L", "V1R",
]


# =========================================================
# 1. Generic helpers
# =========================================================
def write_csv(path: str, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def load_train_manifest(output_dir: str) -> Optional[Dict[str, object]]:
    path = os.path.join(output_dir, "train_manifest.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_manifest_path(output_dir: str, path: Optional[str]) -> Optional[str]:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(os.path.dirname(output_dir), path))


def find_fold_from_manifest(manifest: Optional[Dict[str, object]], fold_idx: int):
    if manifest is None:
        return None
    for item in manifest.get("folds", []):
        if int(item.get("fold", -1)) == int(fold_idx):
            return item
    return None


def safe_macro_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    try:
        labels_bin = label_binarize(labels, classes=[0, 1, 2])
        return float(roc_auc_score(labels_bin, probs, average="macro"))
    except Exception:
        return float("nan")


def mean_std_ci(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0, "ci95_low": float("nan"), "ci95_high": float("nan")}
    if len(arr) == 1:
        v = float(arr[0])
        return {"mean": v, "std": 0.0, "n": 1, "ci95_low": v, "ci95_high": v}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)),
        "n": int(len(arr)),
        "ci95_low": float(np.percentile(arr, 2.5)),
        "ci95_high": float(np.percentile(arr, 97.5)),
    }


# =========================================================
# 2. Prediction and metrics
# =========================================================
@torch.no_grad()
def predict_model(model: torch.nn.Module, samples: Sequence[GraphSample], device: torch.device) -> Dict[str, np.ndarray]:
    loader = make_loader(samples, FINETUNE_BATCH_SIZE, shuffle=False)
    model.eval()

    all_logits = []
    all_labels = []
    all_alphas = []

    for batch in loader:
        batch = batch.to(device)
        logits, alpha = model(batch, mode="classify", return_attn=True)
        labels = batch.y.view(-1)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
        all_alphas.append(alpha.cpu())

    logits = torch.cat(all_logits, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    preds = probs.argmax(axis=1)
    alphas = torch.cat(all_alphas, dim=0).numpy()

    return {
        "logits": logits,
        "labels": labels,
        "probs": probs,
        "preds": preds,
        "alphas": alphas,
    }


def compute_metrics(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> Dict[str, object]:
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_auc": safe_macro_auc(labels, probs),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "baseline_f1": float(f1_score(labels, preds, labels=[0], average="macro", zero_division=0)),
        "seizure_f1": float(f1_score(labels, preds, labels=[1], average="macro", zero_division=0)),
        "recovery_f1": float(f1_score(labels, preds, labels=[2], average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1, 2]).tolist(),
        "classification_report": classification_report(
            labels,
            preds,
            labels=[0, 1, 2],
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
            output_dict=True,
        ),
    }


# =========================================================
# 3. Output files
# =========================================================
def save_oof_predictions(output_dir: str, rows: Sequence[Dict[str, object]]) -> str:
    fieldnames = [
        "fold",
        "mouse_id",
        "source_file",
        "true_label_id",
        "true_label_name",
        "pred_label_id",
        "pred_label_name",
        "correct",
        "logit_baseline",
        "logit_seizure",
        "logit_recovery",
        "p_baseline",
        "p_seizure",
        "p_recovery",
    ]
    for roi_name in ROI_NAMES:
        fieldnames.append(f"alpha_{roi_name}")
    path = os.path.join(output_dir, "oof_predictions.csv")
    write_csv(path, rows, fieldnames)
    return path


def save_fold_metrics(output_dir: str, rows: Sequence[Dict[str, object]]) -> str:
    fieldnames = [
        "fold",
        "test_subject",
        "n_windows",
        "accuracy",
        "macro_auc",
        "macro_f1",
        "baseline_f1",
        "seizure_f1",
        "recovery_f1",
        "cm_00",
        "cm_01",
        "cm_02",
        "cm_10",
        "cm_11",
        "cm_12",
        "cm_20",
        "cm_21",
        "cm_22",
    ]
    path = os.path.join(output_dir, "fold_metrics.csv")
    write_csv(path, rows, fieldnames)
    return path


def save_subject_metrics_for_stats(output_dir: str, rows: Sequence[Dict[str, object]]) -> str:
    fieldnames = [
        "method",
        "fold",
        "subject",
        "n_windows",
        "accuracy",
        "macro_auc",
        "macro_f1",
        "baseline_f1",
        "seizure_f1",
        "recovery_f1",
    ]
    path = os.path.join(output_dir, "subject_metrics_for_stats.csv")
    write_csv(path, rows, fieldnames)
    return path


def save_summary_metrics_csv(output_dir: str, summary_metrics: Dict[str, Dict[str, float]]) -> str:
    rows = []
    for metric_name, item in summary_metrics.items():
        rows.append({
            "metric": metric_name,
            "mean": item.get("mean"),
            "std": item.get("std"),
            "n": item.get("n"),
            "ci95_low": item.get("ci95_low"),
            "ci95_high": item.get("ci95_high"),
        })
    path = os.path.join(output_dir, "summary_metrics.csv")
    write_csv(path, rows, ["metric", "mean", "std", "n", "ci95_low", "ci95_high"])
    return path


def save_aggregate_cm(output_dir: str, cm: np.ndarray) -> str:
    rows = []
    for i, true_name in enumerate(CLASS_NAMES):
        rows.append({
            "true_label": true_name,
            "pred_Baseline": int(cm[i, 0]),
            "pred_Seizure": int(cm[i, 1]),
            "pred_Recovery": int(cm[i, 2]),
        })
    path = os.path.join(output_dir, "aggregate_confusion_matrix.csv")
    write_csv(path, rows, ["true_label", "pred_Baseline", "pred_Seizure", "pred_Recovery"])
    return path


def save_json_summary(summary: Dict[str, object], output_dir: str) -> None:
    json_path = os.path.join(output_dir, "cv_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    txt_path = os.path.join(output_dir, "cv_summary.txt")
    sm = summary["summary_metrics"]
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("AER-GT inference-only summary\n")
        f.write("=" * 60 + "\n")
        for metric in ["accuracy", "macro_auc", "macro_f1", "baseline_f1", "seizure_f1", "recovery_f1"]:
            item = sm[metric]
            f.write(
                f"{metric:14s}: {item['mean']:.4f} +/- {item['std']:.4f} "
                f"[95% CI {item['ci95_low']:.4f}, {item['ci95_high']:.4f}], n={item['n']}\n"
            )
        f.write("\nOutput files\n")
        for k, v in summary.get("output_paths", {}).items():
            f.write(f"{k}: {v}\n")

    print(f">>> Summary JSON saved to: {json_path}")
    print(f">>> Summary TXT  saved to: {txt_path}")


# =========================================================
# 4. Main inference-only pipeline
# =========================================================
def run_inference_only(
    labeled_path: str = LABELED_PATH,
    output_dir: str = OUTPUT_DIR,
    control_path: Optional[str] = CONTROL_PATH,
) -> Dict[str, object]:
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory not found: {output_dir}. Run train-only first.")

    print("\n>>> Loading labeled dataset")
    labeled_raw = load_graph_samples(labeled_path, is_labeled=True)
    if control_path:
        print("Control inference is not used in this script unless you extend the output table.")

    folds = build_fold_splits(labeled_raw, n_splits=NUM_FOLDS)
    manifest = load_train_manifest(output_dir)

    print(f">>> Found {len(get_subjects(labeled_raw))} labeled subjects for {NUM_FOLDS}-fold subject-level CV")

    fold_results: List[Dict[str, object]] = []
    fold_metric_rows: List[Dict[str, object]] = []
    subject_metric_rows: List[Dict[str, object]] = []
    oof_rows: List[Dict[str, object]] = []
    aggregate_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for fold_idx, train_subjects, test_subjects in folds:
        print("\n" + "=" * 80)
        print(f"Fold {fold_idx}/{NUM_FOLDS}")
        print(f"  Train subjects: {train_subjects}")
        print(f"  Test subjects : {test_subjects}")

        manifest_item = find_fold_from_manifest(manifest, fold_idx)
        scaler_path = os.path.join(output_dir, f"fold_{fold_idx}_scaler.pkl")
        model_path = os.path.join(output_dir, f"fold_{fold_idx}_final_model.pth")
        if manifest_item:
            scaler_path = resolve_manifest_path(output_dir, manifest_item.get("scaler_path", scaler_path))
            model_path = resolve_manifest_path(output_dir, manifest_item.get("final_model_path", model_path))

        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Final model not found: {model_path}")

        scaler = joblib.load(scaler_path)
        test_raw = subset_by_subjects(labeled_raw, test_subjects)
        test_samples = apply_scaler_to_samples(test_raw, scaler)

        print(f"  Test samples: {len(test_samples)}")
        print(f"  Loading scaler: {scaler_path}")
        print(f"  Loading model : {model_path}")

        model = AERGFT().to(DEVICE)
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)

        pred = predict_model(model, test_samples, DEVICE)
        labels = pred["labels"]
        preds = pred["preds"]
        probs = pred["probs"]
        logits = pred["logits"]
        alphas = pred["alphas"]

        fold_metrics = compute_metrics(labels, preds, probs)
        cm = np.array(fold_metrics["confusion_matrix"], dtype=np.int64)
        aggregate_cm += cm

        fold_metric_rows.append({
            "fold": fold_idx,
            "test_subject": ",".join(test_subjects),
            "n_windows": len(test_samples),
            "accuracy": fold_metrics["accuracy"],
            "macro_auc": fold_metrics["macro_auc"],
            "macro_f1": fold_metrics["macro_f1"],
            "baseline_f1": fold_metrics["baseline_f1"],
            "seizure_f1": fold_metrics["seizure_f1"],
            "recovery_f1": fold_metrics["recovery_f1"],
            "cm_00": int(cm[0, 0]),
            "cm_01": int(cm[0, 1]),
            "cm_02": int(cm[0, 2]),
            "cm_10": int(cm[1, 0]),
            "cm_11": int(cm[1, 1]),
            "cm_12": int(cm[1, 2]),
            "cm_20": int(cm[2, 0]),
            "cm_21": int(cm[2, 1]),
            "cm_22": int(cm[2, 2]),
        })

        # Subject-level table. With 5-fold subject CV this is one row per held-out mouse.
        for subject in sorted({s.mouse_id for s in test_samples}):
            idx = np.array([s.mouse_id == subject for s in test_samples], dtype=bool)
            subject_metrics = compute_metrics(labels[idx], preds[idx], probs[idx])
            subject_metric_rows.append({
                "method": METHOD_NAME,
                "fold": fold_idx,
                "subject": subject,
                "n_windows": int(idx.sum()),
                "accuracy": subject_metrics["accuracy"],
                "macro_auc": subject_metrics["macro_auc"],
                "macro_f1": subject_metrics["macro_f1"],
                "baseline_f1": subject_metrics["baseline_f1"],
                "seizure_f1": subject_metrics["seizure_f1"],
                "recovery_f1": subject_metrics["recovery_f1"],
            })

        for i, sample in enumerate(test_samples):
            true_id = int(labels[i])
            pred_id = int(preds[i])
            row = {
                "fold": fold_idx,
                "mouse_id": sample.mouse_id,
                "source_file": sample.source_file,
                "true_label_id": true_id,
                "true_label_name": CLASS_NAMES[true_id],
                "pred_label_id": pred_id,
                "pred_label_name": CLASS_NAMES[pred_id],
                "correct": int(true_id == pred_id),
                "logit_baseline": float(logits[i, 0]),
                "logit_seizure": float(logits[i, 1]),
                "logit_recovery": float(logits[i, 2]),
                "p_baseline": float(probs[i, 0]),
                "p_seizure": float(probs[i, 1]),
                "p_recovery": float(probs[i, 2]),
            }
            for roi_idx, roi_name in enumerate(ROI_NAMES):
                row[f"alpha_{roi_name}"] = float(alphas[i, roi_idx])
            oof_rows.append(row)

        fold_record = {
            "fold": fold_idx,
            "train_subjects": train_subjects,
            "test_subjects": test_subjects,
            "model_path": model_path,
            "scaler_path": scaler_path,
            "metrics": fold_metrics,
            "test_predictions": {
                "labels": labels.astype(int).tolist(),
                "preds": preds.astype(int).tolist(),
                "probs": probs.astype(float).tolist(),
                "logits": logits.astype(float).tolist(),
                "alphas": alphas.astype(float).tolist(),
            },
            "test_metadata": {
                "mouse_ids": [s.mouse_id for s in test_samples],
                "source_files": [s.source_file for s in test_samples],
            },
        }
        fold_results.append(fold_record)

        print(
            f"  Fold metrics | Acc={fold_metrics['accuracy']:.4f}, "
            f"Macro-AUC={fold_metrics['macro_auc']:.4f}, "
            f"Macro-F1={fold_metrics['macro_f1']:.4f}, "
            f"Recovery-F1={fold_metrics['recovery_f1']:.4f}"
        )

    metric_names = ["accuracy", "macro_auc", "macro_f1", "baseline_f1", "seizure_f1", "recovery_f1"]
    summary_metrics = {
        metric: mean_std_ci([float(fr["metrics"][metric]) for fr in fold_results])
        for metric in metric_names
    }

    output_paths = {
        "oof_predictions_csv": save_oof_predictions(output_dir, oof_rows),
        "fold_metrics_csv": save_fold_metrics(output_dir, fold_metric_rows),
        "subject_metrics_for_stats_csv": save_subject_metrics_for_stats(output_dir, subject_metric_rows),
        "aggregate_confusion_matrix_csv": save_aggregate_cm(output_dir, aggregate_cm),
    }
    output_paths["summary_metrics_csv"] = save_summary_metrics_csv(output_dir, summary_metrics)

    summary = {
        "n_folds": len(fold_results),
        "fold_results": fold_results,
        "summary_metrics": summary_metrics,
        "aggregate_confusion_matrix": aggregate_cm.tolist(),
        "output_paths": output_paths,
    }
    save_json_summary(summary, output_dir)
    return summary


def main() -> None:
    summary = run_inference_only(
        labeled_path=LABELED_PATH,
        output_dir=OUTPUT_DIR,
        control_path=CONTROL_PATH,
    )

    sm = summary["summary_metrics"]
    print("\nFinal 5-fold summary")
    print("-" * 60)
    for metric in ["accuracy", "macro_auc", "macro_f1", "baseline_f1", "seizure_f1", "recovery_f1"]:
        item = sm[metric]
        print(
            f"{metric:14s}: {item['mean']:.4f} +/- {item['std']:.4f} "
            f"[95% CI {item['ci95_low']:.4f}, {item['ci95_high']:.4f}], n={item['n']}"
        )


if __name__ == "__main__":
    main()
