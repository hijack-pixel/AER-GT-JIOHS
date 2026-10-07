"""Core computations for Sec. 2.8 interpretability analyses."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from aergt.data.samples import GraphSample
from aergt.data.scalers import load_standardizer
from aergt.data.splits import subject_level_folds, subset_by_subjects
from aergt.evaluation.metrics import classification_metrics
from aergt.models import AERGT
from aergt.training.runner import _model_kwargs

CLASS_NAMES = ("baseline", "seizure", "recovery")
RECOVERY_IDX = 2


def build_fold_model(artifact_dir: str | Path, fold: int, model_kwargs: dict[str, Any]) -> AERGT:
    ckpt = Path(artifact_dir) / "checkpoints" / f"fold_{fold}_final_model.pth"
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model = AERGT(**model_kwargs)
    model.load_state_dict(state)
    model.eval()
    return model


def fold_scaler(artifact_dir: str | Path, fold: int):
    return load_standardizer(Path(artifact_dir) / "scalers" / f"fold_{fold}_scaler.pkl")


@torch.no_grad()
def forward_batch(model: AERGT, samples, scaler, *, device: torch.device):
    """Standardized forward pass returning (probs, per-node readout weights,
    per-pair interaction matrices)."""
    xs = np.stack([np.asarray(s.x, dtype=np.float64) for s in samples])
    shape = xs.shape
    flat = scaler.transform(xs.reshape(shape[0], -1)).reshape(shape)
    x = torch.tensor(flat, dtype=torch.float32, device=device)
    adj = torch.tensor(np.stack([np.asarray(s.adj) for s in samples]),
                       dtype=torch.float32, device=device)
    logits, alpha = model(x, adj, return_attention=True)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    alpha = alpha.detach().cpu().numpy()                       # B, N (AER readout)
    inter = extract_interaction_matrices(model, x, adj)        # B, N, N
    return probs, alpha, inter


@torch.no_grad()
def extract_interaction_matrices(model: AERGT, x: torch.Tensor, adj: torch.Tensor) -> np.ndarray:
    """ROI-ROI attention matrices from the global attention encoder:
    attention weights averaged across heads, then symmetrized
    ((A + A^T) / 2) into an undirected interaction matrix (Sec. 2.8)."""
    node_out = model.encode_nodes(x, adj)                       # B, N, D
    mats = []
    for b in range(node_out.shape[0]):
        h = node_out[b:b + 1]                                   # 1, N, D
        layer_attn = []
        for layer in model.transformer.layers:
            attn = layer.self_attn(h, h, h, need_weights=True,
                                   average_attn_weights=False)[1]   # (1, H, N, N)
            a = attn.mean(axis=1)[0]                                 # heads averaged -> N, N
            layer_attn.append(a)
        a = np.mean(layer_attn, axis=0)                              # layers averaged
        a = (a + a.T) / 2.0                                          # symmetrized
        mats.append(a)
    return np.stack(mats)


def fold_normalize(v: np.ndarray) -> np.ndarray:
    """Center and normalize across ROIs within a fold (Sec. 2.8)."""
    v = np.asarray(v, dtype=float)
    std = v.std()
    return (v - v.mean()) / std if std > 0 else v - v.mean()


def recovery_f1(samples, probs) -> float:
    preds = probs.argmax(axis=1)
    labels = np.asarray([s.label for s in samples])
    onehot = np.eye(len(CLASS_NAMES))[preds]
    return float(classification_metrics(labels, preds, onehot,
                                        num_classes=len(CLASS_NAMES))["recovery_f1"])


@torch.no_grad()
def roi_masking_recovery_f1_drop(model, samples, scaler, *, device: torch.device) -> np.ndarray:
    """Per-ROI Recovery-F1 drop (percentage points) when the ROI's node
    features are masked (zeroed) one at a time; the fold model is reapplied
    without retraining."""
    base_f1 = recovery_f1_from_model(model, samples, scaler, device)
    drops = np.zeros(len(samples[0].x))
    for roi in range(len(samples[0].x)):
        masked = []
        for s in samples:
            x = np.array(s.x, dtype=np.float64)
            x[roi, :] = 0.0
            masked.append(type(s)(x=x, adj=s.adj, label=s.label,
                                  mouse_id=s.mouse_id, source_file=s.source_file,
                                  original_file=s.original_file,
                                  window_index=s.window_index))
        f1 = recovery_f1_from_model(model, masked, scaler, device)
        drops[roi] = 100.0 * (base_f1 - f1)
    return drops


@torch.no_grad()
def recovery_f1_from_model(model, samples, scaler, device) -> float:
    xs = np.stack([np.asarray(s.x, dtype=np.float64) for s in samples])
    shape = xs.shape
    flat = scaler.transform(xs.reshape(shape[0], -1)).reshape(shape)
    x = torch.tensor(flat, dtype=torch.float32, device=device)
    adj = torch.tensor(np.stack([np.asarray(s.adj) for s in samples]),
                       dtype=torch.float32, device=device)
    probs = torch.softmax(model(x, adj), dim=1).cpu().numpy()
    return recovery_f1(samples, probs)


def run_all(oof_samples, artifact_dir: str | Path, model_kwargs: dict[str, Any],
            *, device: torch.device, top_k: int = 8) -> dict[str, Any]:
    """Full Sec. 2.8 pipeline over the five held-out folds.

    ``oof_samples`` are the labelled samples; for each fold only the
    held-out subjects are analysed, with that fold's model and scaler.
    """
    device = torch.device(device) if isinstance(device, str) else device
    folds = subject_level_folds(oof_samples, 5)

    readout_by_state = []          # per fold: {state: fold-normalized mean alpha}
    interaction_contrast = []      # per fold: recovery-mean minus baseline-mean
    mask_drops = []                # per fold: per-ROI Recovery-F1 drop (pp)

    for fold, _train_subjects, test_subjects in folds:
        test = subset_by_subjects(oof_samples, test_subjects)
        model = build_fold_model(artifact_dir, fold, model_kwargs).to(device)
        scaler = fold_scaler(artifact_dir, fold)
        probs, alpha, inter = forward_batch(model, test, scaler, device=device)

        # 1) readout weights per state, fold-normalized
        per_state = {}
        for k, name in enumerate(CLASS_NAMES):
            idx = [i for i, s in enumerate(test) if s.label == k]
            if idx:
                per_state[name] = fold_normalize(alpha[idx].mean(axis=0))
        readout_by_state.append(per_state)

        # 3) interaction contrast: recovery minus baseline means
        rec_idx = [i for i, s in enumerate(test) if s.label == RECOVERY_IDX]
        bas_idx = [i for i, s in enumerate(test) if s.label == 0]
        if rec_idx and bas_idx:
            interaction_contrast.append(
                inter[rec_idx].mean(axis=0) - inter[bas_idx].mean(axis=0))

        # 2) ROI masking sensitivity
        mask_drops.append(roi_masking_recovery_f1_drop(
            model, test, scaler, device=device))

    # cross-fold averages -------------------------------------------------
    mean_recovery = np.mean([r["recovery"] for r in readout_by_state], axis=0)
    order = np.argsort(mean_recovery)[::-1]
    top_roi = order[:top_k]

    contrasts = {}
    for a, b in (("recovery", "baseline"), ("recovery", "seizure")):
        per_fold = [r[a] - r[b] for r in readout_by_state if a in r and b in r]
        contrasts[f"{a}_minus_{b}"] = {
            "per_roi_mean": (np.mean(per_fold, axis=0) if per_fold else None),
            "top_roi_mean": (float(np.mean([np.mean(per_fold, axis=0)[i] for i in top_roi]))
                             if per_fold else None),
        }

    interaction = np.mean(interaction_contrast, axis=0) if interaction_contrast else None
    edges = None
    if interaction is not None:
        # readout-weighted interaction score: |contrast| weighted by the mean
        # recovery-class readout contribution of the two endpoint ROIs
        score = np.zeros_like(interaction)
        for i in range(interaction.shape[0]):
            for j in range(interaction.shape[1]):
                if j <= i:
                    continue
                score[i, j] = score[j, i] = abs(interaction[i, j]) * 0.5 * (
                    mean_recovery[i] + mean_recovery[j])
        iu = np.triu_indices_from(score, k=1)
        ranked = np.argsort(score[iu])[::-1]
        keep = max(1, int(round(0.10 * len(ranked))))
        edges = {
            "n_pairs": int(len(ranked)),
            "n_displayed": int(keep),
            "top_edges": [
                {"roi_i": int(iu[0][r]), "roi_j": int(iu[1][r]),
                 "contrast": float(interaction[iu[0][r], iu[1][r]]),
                 "readout_weighted_score": float(score[iu[0][r], iu[1][r]])}
                for r in ranked[:keep]
            ],
        }

    return {
        "mean_recovery_readout": mean_recovery.tolist(),
        "top_roi_indices": top_roi.tolist(),
        "mask_recovery_f1_drop_pp_mean": np.mean(mask_drops, axis=0).tolist(),
        "readout_contrasts_topk_mean": {k: v["top_roi_mean"] for k, v in contrasts.items()},
        "interaction_contrast": (interaction.tolist() if interaction is not None else None),
        "edges_top10pct": edges,
    }
