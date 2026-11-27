"""Pharmacological-control cohort decoding (Sec. 2.9).

Implements the evaluation described in the manuscript:

* control recordings are preprocessed with the pipeline of Sec. 2.2
  (registration, within-recording normalization, 20-s windows with a 4-s stride,
  six-dimensional node features, correlation-derived adjacency) by
  ``preprocess/matlab/export_control_samples.m``;
* each control window is standardized with the per-fold scaler and decoded by
  the trained fold model of the same fold;
* the ensemble class probability is the mean of the per-fold softmax
  probabilities;
* because these recordings were never labelled or used in training, the
  outputs are class-assignment proportions per animal and overall, the mean
  class probabilities, and the per-window ensemble probabilities.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from aergt.data.scalers import load_standardizer
from aergt.data.samples import GraphSample, read_aert_sample
from aergt.models import AERGT
from aergt.utils.runtime import write_json


def load_control_samples(control_dir: str | Path, *, num_nodes: int, in_channels: int, use_abs_adj: bool) -> list[GraphSample]:
    """Load preprocessed control ``.mat`` samples (exported with ``Label = -1``)."""
    control_dir = Path(control_dir)
    samples: list[GraphSample] = []
    skipped = 0
    for mat_file in sorted(control_dir.rglob("*.mat")):
        sample = read_aert_sample(mat_file, labeled=False, num_nodes=num_nodes, in_channels=in_channels, use_abs_adj=use_abs_adj)
        if sample is None:
            skipped += 1
            continue
        samples.append(sample)
    if not samples:
        raise FileNotFoundError(
            f"No control .mat samples found under {control_dir}. "
            "Run preprocess/matlab/export_control_samples.m first, or point --control-dir at the prepared folder."
        )
    print(f"Loaded {len(samples)} control samples from {control_dir}; skipped {skipped}.")
    return samples


def load_fold_models(artifact_dir: str | Path, *, model_kwargs: dict[str, Any], num_folds: int) -> list[tuple[int, AERGT, Any]]:
    """Load the trained fold models and their per-fold scalers."""
    artifact_dir = Path(artifact_dir)
    entries = []
    for fold in range(1, num_folds + 1):
        ckpt = artifact_dir / "checkpoints" / f"fold_{fold}_final_model.pth"
        scaler = artifact_dir / "scalers" / f"fold_{fold}_scaler.pkl"
        if not ckpt.exists() or not scaler.exists():
            raise FileNotFoundError(f"Missing fold-{fold} checkpoint or scaler under {artifact_dir}.")
        model = AERGT(**model_kwargs)
        state = torch.load(ckpt, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        model.eval()
        entries.append((fold, model, load_standardizer(scaler)))
    return entries


def standardize(samples: Sequence[GraphSample], scaler) -> np.ndarray:
    """Apply a feature scaler (fitted on the six node features) to a set of samples."""
    xs = np.stack([np.asarray(s.x, dtype=np.float64) for s in samples])
    shape = xs.shape
    scaled = scaler.transform(xs.reshape(-1, shape[-1])).reshape(shape)
    return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


@torch.no_grad()
def decode_control_cohort(
    control_samples: Sequence[GraphSample],
    fold_entries,
    *,
    class_names: Sequence[str],
    device: torch.device | str = "cpu",
    batch_size: int = 256,
) -> dict[str, Any]:
    """Ensemble decoding of all control windows.

    Each window is standardized with every per-fold scaler, decoded by the
    matching fold model, and the ensemble probability is the mean softmax over
    the fold models (Sec. 2.9).
    """
    device = torch.device(device) if isinstance(device, str) else device
    n = len(control_samples)
    num_classes = len(class_names)
    ensemble_probs = np.zeros((n, num_classes), dtype=np.float64)
    adjacency = np.stack([np.asarray(s.adj) for s in control_samples])

    for _fold, model, scaler in fold_entries:
        model = model.to(device)
        xs = standardize(control_samples, scaler)
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            x = torch.tensor(xs[start:stop], dtype=torch.float32, device=device)
            adj = torch.tensor(adjacency[start:stop], dtype=torch.float32, device=device)
            probs = torch.softmax(model(x, adj), dim=1).cpu().numpy()
            ensemble_probs[start:stop] += probs / len(fold_entries)

    preds = ensemble_probs.argmax(axis=1)
    counts = np.bincount(preds, minlength=num_classes)

    def proportions(index: list[int]) -> dict[str, float]:
        local = np.bincount(preds[index], minlength=num_classes)
        return {f"{name}_proportion": float(local[k] / len(index)) for k, name in enumerate(class_names)}

    animals = sorted({s.mouse_id for s in control_samples})
    per_animal = {}
    for animal in animals:
        index = [i for i, s in enumerate(control_samples) if s.mouse_id == animal]
        per_animal[animal] = {
            "n_windows": int(len(index)),
            **proportions(index),
            "mean_class_probability": {name: float(ensemble_probs[index, k].mean()) for k, name in enumerate(class_names)},
        }
    summary = {
        "n_windows": int(n),
        "overall": {f"{name}_proportion": float(counts[k] / n) for k, name in enumerate(class_names)},
        "mean_class_probability": {name: float(ensemble_probs[:, k].mean()) for k, name in enumerate(class_names)},
        "per_animal": per_animal,
    }
    return {"summary": summary, "ensemble_probs": ensemble_probs, "preds": preds}


def write_control_outputs(result: dict[str, Any], samples: Sequence[GraphSample], class_names: Sequence[str], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(result["summary"], out_dir / "control_decoding_summary.json")
    with (out_dir / "control_window_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mouse_id", "source_file", "original_file", "window_index",
                         *[f"p_{name}" for name in class_names], "pred_class"])
        probs, preds = result["ensemble_probs"], result["preds"]
        for i, s in enumerate(samples):
            writer.writerow([s.mouse_id, Path(s.source_file).name, s.original_file, s.window_index,
                             *[f"{value:.6f}" for value in probs[i]], int(preds[i])])
