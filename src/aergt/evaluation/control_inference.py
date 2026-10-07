"""Pharmacological-control cohort decoding (Sec. 2.9).

Implements exactly the evaluation described in the manuscript:

* control recordings are preprocessed with the identical pipeline of
  Sec. 2.2 (registration, within-recording normalization, 20-s windows with
  a 4-s stride, six-dimensional node features, correlation-derived
  adjacency) by ``preprocess/matlab/export_control_pretrain_samples.m``;
* each control window is standardized with the corresponding **per-fold
  scaler** and decoded by **all five trained fold models**;
* the ensemble class probability is the **mean of the per-fold softmax
  probabilities**;
* because these recordings were never labelled or used in training, the
  outputs are class-assignment proportions per animal and overall, plus the
  mean baseline-class probability.

Outputs are written under ``outputs/control_inference/`` and are not
versioned in the repository.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aergt.data.scalers import load_standardizer
from aergt.data.samples import GraphSample, read_aert_sample
from aergt.models import AERGT
from aergt.utils.runtime import resolve_device, write_json


def load_control_samples(control_dir: str | Path) -> list[GraphSample]:
    """Load preprocessed control ``.mat`` samples (labels are placeholders)."""
    control_dir = Path(control_dir)
    samples: list[GraphSample] = []
    for mat_file in sorted(control_dir.rglob("*.mat")):
        samples.append(read_aert_sample(mat_file))
    if not samples:
        raise FileNotFoundError(
            f"No control .mat samples found under {control_dir}. "
            "Run preprocess/matlab/export_control_pretrain_samples.m first, "
            "or point --control-dir at the prepared folder."
        )
    return samples


def load_fold_models(artifact_dir: str | Path, *, model_kwargs: dict[str, Any]):
    """Load the five trained fold models and their per-fold scalers."""
    artifact_dir = Path(artifact_dir)
    entries = []
    for fold in range(1, 6):
        ckpt = artifact_dir / "checkpoints" / f"fold_{fold}_final_model.pth"
        scaler = artifact_dir / "scalers" / f"fold_{fold}_scaler.pkl"
        if not ckpt.exists() or not scaler.exists():
            raise FileNotFoundError(
                f"Missing fold-{fold} checkpoint or scaler under {artifact_dir}."
            )
        model = AERGT(**model_kwargs)
        state = torch.load(ckpt, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        model.eval()
        entries.append((fold, model, load_standardizer(scaler)))
    return entries


@torch.no_grad()
def decode_control_cohort(
    control_samples: list[GraphSample],
    fold_entries,
    *,
    device: torch.device | str = "cpu",
    batch_size: int = 256,
) -> dict[str, Any]:
    """Out-of-fold ensemble decoding of all control windows.

    Each window is standardized with every per-fold scaler, decoded by the
    matching fold model, and the ensemble probability is the mean softmax
    across the five fold models (Sec. 2.9).
    """
    device = torch.device(device) if isinstance(device, str) else device
    n = len(control_samples)
    ensemble_probs = np.zeros((n, 3), dtype=np.float64)

    for fold, model, scaler in fold_entries:
        model = model.to(device)
        # standardize this fold's copy of all windows with the fold scaler
        xs = np.stack([np.asarray(s.x, dtype=np.float64) for s in control_samples])
        shape = xs.shape
        flat = xs.reshape(shape[0], -1)
        flat = scaler.transform(flat)
        xs = flat.reshape(shape)

        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            x = torch.tensor(xs[start:stop], dtype=torch.float32, device=device)
            adj = torch.tensor(
                np.stack([np.asarray(s.adj) for s in control_samples[start:stop]]),
                dtype=torch.float32, device=device)
            logits = model(x, adj)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            ensemble_probs[start:stop] += probs / len(fold_entries)

    preds = ensemble_probs.argmax(axis=1)
    class_names = ["baseline", "seizure", "recovery"]
    per_animal: dict[str, dict[str, float]] = {}
    animals = sorted({s.mouse_id for s in control_samples})
    for animal in animals:
        idx = [i for i, s in enumerate(control_samples) if s.mouse_id == animal]
        counts = np.bincount(preds[idx], minlength=3)
        per_animal[animal] = {
            "n_windows": int(len(idx)),
            **{f"{name}_proportion": float(counts[k] / len(idx))
               for k, name in enumerate(class_names)},
            "mean_baseline_probability": float(ensemble_probs[idx, 0].mean()),
        }
    overall_counts = np.bincount(preds, minlength=3)
    summary = {
        "n_windows": int(n),
        "overall": {
            f"{name}_proportion": float(overall_counts[k] / n)
            for k, name in enumerate(class_names)
        },
        "mean_baseline_probability": float(ensemble_probs[:, 0].mean()),
        "per_animal": per_animal,
    }
    return {"summary": summary, "ensemble_probs": ensemble_probs, "preds": preds}


def write_control_outputs(result: dict[str, Any], samples: list[GraphSample],
                          out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(result["summary"], out_dir / "control_decoding_summary.json")
    with (out_dir / "control_window_predictions.csv").open("w", encoding="utf-8",
                                                           newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mouse_id", "source_file", "window_index",
                         "p_baseline", "p_seizure", "p_recovery", "pred_class"])
        probs, preds = result["ensemble_probs"], result["preds"]
        for i, s in enumerate(samples):
            writer.writerow([s.mouse_id, s.source_file, s.window_index,
                             f"{probs[i, 0]:.6f}", f"{probs[i, 1]:.6f}",
                             f"{probs[i, 2]:.6f}", int(preds[i])])
