"""Core computations for Sec. 2.8 interpretability analyses."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score

from aergt.data.samples import GraphSample
from aergt.data.scalers import load_standardizer
from aergt.data.splits import subject_level_folds, subset_by_subjects
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


def standardize(samples, scaler) -> np.ndarray:
    """Apply a feature scaler (fitted on the node features) to a set of samples."""
    xs = np.stack([np.asarray(s.x, dtype=np.float64) for s in samples])
    shape = xs.shape
    scaled = scaler.transform(xs.reshape(-1, shape[-1])).reshape(shape)
    return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def tensors(samples, xs: np.ndarray, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(xs, dtype=torch.float32, device=device)
    adj = torch.tensor(np.stack([np.asarray(s.adj) for s in samples]), dtype=torch.float32, device=device)
    return x, adj


@torch.no_grad()
def forward_batch(model: AERGT, samples, scaler, *, device: torch.device):
    """Standardized forward pass returning (probs, per-node readout weights,
    per-pair interaction matrices)."""
    x, adj = tensors(samples, standardize(samples, scaler), device=device)
    logits, alpha = model(x, adj, return_attention=True)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    alpha = alpha.detach().cpu().numpy()                       # B, N (AER readout)
    inter = extract_interaction_matrices(model, x, adj)        # B, N, N
    return probs, alpha, inter


@torch.no_grad()
def extract_interaction_matrices(model: AERGT, x: torch.Tensor, adj: torch.Tensor) -> np.ndarray:
    """ROI-ROI attention matrices of the global attention encoder (Sec. 2.8).

    For every window, the attention weights of each self-attention layer are the
    ones the layer receives as input in the forward pass; they are averaged
    across heads, then symmetrized ((A + A^T) / 2) into an undirected interaction
    matrix, and finally averaged across layers."""
    weights = model.encoder_attention(x, adj)                 # per layer: (B, heads, N, N)
    mats = []
    for attn in weights:
        a = attn.mean(dim=1)                                  # heads averaged -> (B, N, N)
        a = (a + a.transpose(1, 2)) / 2.0                     # symmetrized
        mats.append(a.cpu().numpy())
    return np.mean(mats, axis=0)


def fold_normalize(v: np.ndarray) -> np.ndarray:
    """Center and normalize across ROIs within a fold (Sec. 2.8)."""
    v = np.asarray(v, dtype=float)
    std = v.std()
    return (v - v.mean()) / std if std > 0 else v - v.mean()


@torch.no_grad()
def roi_masking_recovery_f1_drop(model, samples, scaler, *, device: torch.device) -> np.ndarray:
    """Per-ROI Recovery-F1 drop (percentage points) when the node features of one
    ROI are masked (set to zero in the standardized model input, as the mask
    token of pretraining) one at a time; the fold model is reapplied without
    retraining (Sec. 2.8)."""
    labels = np.asarray([s.label for s in samples])
    adjacency = np.stack([np.asarray(s.adj) for s in samples])
    xs = standardize(samples, scaler)

    def recovery_f1_of(x: np.ndarray) -> float:
        logits = model(torch.tensor(x, dtype=torch.float32, device=device),
                       torch.tensor(adjacency, dtype=torch.float32, device=device))
        preds = logits.argmax(dim=1).cpu().numpy()
        return float(f1_score(labels, preds, labels=[RECOVERY_IDX], average="macro", zero_division=0))

    base_f1 = recovery_f1_of(xs)
    drops = np.zeros(xs.shape[1])
    for roi in range(xs.shape[1]):
        masked = xs.copy()
        masked[:, roi, :] = 0.0
        drops[roi] = 100.0 * (base_f1 - recovery_f1_of(masked))
    return drops


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
                # absolute readout-weighted interaction score: |contrast| weighted by
                # the mean recovery-class readout of the two endpoint ROIs
                score[i, j] = score[j, i] = abs(
                    interaction[i, j] * 0.5 * (mean_recovery[i] + mean_recovery[j]))
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
        "readout_contrasts_per_roi": {k: (v["per_roi_mean"].tolist() if v["per_roi_mean"] is not None else None)
                                      for k, v in contrasts.items()},
        "interaction_contrast": (interaction.tolist() if interaction is not None else None),
        "edges_top10pct": edges,
        # fold-wise values behind the cross-fold means (one array per fold)
        "per_fold": {
            "held_out_subjects": [list(test_subjects) for _fold, _train, test_subjects in folds],
            "readout_by_state": {name: [r[name].tolist() if name in r else None for r in readout_by_state]
                                 for name in CLASS_NAMES},
            "mask_recovery_f1_drop_pp": [m.tolist() for m in mask_drops],
            "interaction_contrast_recovery_minus_baseline": [m.tolist() for m in interaction_contrast],
            "readout_contrasts": {f"{a}_minus_{b}": [(r[a] - r[b]).tolist() if a in r and b in r else None
                                                     for r in readout_by_state]
                                  for a, b in (("recovery", "baseline"), ("recovery", "seizure"))},
        },
    }
