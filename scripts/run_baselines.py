"""Run the Table 2 baseline models under the SAME subject-level five-fold
splits as AER-GT (Sec. 2.7).

Feature-based baselines (vectorized ROI features):
    logistic_regression | xgboost | mlp
Graph-based baselines (same node-feature and adjacency matrices as AER-GT):
    supervised_gcn | supervised_gat | self_supervised_gat | self_supervised_bnt

Usage:
    python scripts/run_baselines.py --config configs/full_model.yaml --all
    python scripts/run_baselines.py --config configs/full_model.yaml --only xgboost,mlp

Inputs are read from the paths declared in the config (private ``.mat``
samples under ``data/private/labeled_epilepsy``). Metrics are written to
``outputs/baselines/<name>/`` and are regenerated at run time; no result
tables are versioned in the repository.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from aergt.config import load_config
from aergt.data.samples import apply_scaler, fit_scaler, load_samples
from aergt.data.splits import subject_level_folds, subset_by_subjects
from aergt.evaluation.metrics import classification_metrics, mean_std_ci
from aergt.models.baselines import FEATURE_BASELINES, GRAPH_BASELINES, SelfSupervisedWrapper
from aergt.utils.runtime import resolve_device, set_seed


def _flatten_graph(samples):
    """Graph tensors for the graph-based baselines."""
    x = torch.tensor(np.stack([s.x for s in samples]), dtype=torch.float32)
    adj = torch.tensor(np.stack([s.adj for s in samples]), dtype=torch.float32)
    y = torch.tensor([s.label for s in samples], dtype=torch.long)
    return x, adj, y


def _train_graph_model(model, train_samples, *, device, epochs=40, lr=1e-3,
                       batch_size=64, pretrain_epochs=20):
    """Two-stage training: reconstruction pretraining (self-supervised
    variants only) followed by supervised cross-entropy finetuning."""
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x, adj, y = _flatten_graph(train_samples)
    x, adj, y = x.to(device), adj.to(device), y.to(device)

    if isinstance(model, SelfSupervisedWrapper):
        model.train()
        for _ in range(pretrain_epochs):
            opt.zero_grad()
            recon = model(x, adj, mode="reconstruct")
            loss = torch.nn.functional.mse_loss(recon, x)
            loss.backward()
            opt.step()

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(x, adj)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def _eval_graph_model(model, samples, *, device):
    model.eval()
    x, adj, y = _flatten_graph(samples)
    x, adj, y = x.to(device), adj.to(device), y.to(device)
    logits = model(x, adj)
    preds = logits.argmax(dim=1).cpu().numpy()
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    return classification_metrics(y.cpu().numpy(), preds, probs, num_classes=3)


def run_feature_baseline(name: str, cfg: dict, samples) -> dict:
    from aergt.models.baselines import vectorize_features, vectorize_labels

    folds = subject_level_folds(samples, cfg.get("cv", {}).get("n_splits", 5))
    rows = []
    for _fold, train_subjects, test_subjects in folds:
        train = subset_by_subjects(samples, train_subjects)
        test = subset_by_subjects(samples, test_subjects)
        scaler = fit_scaler(train)
        train_s, test_s = apply_scaler(train, scaler), apply_scaler(test, scaler)
        est = FEATURE_BASELINES[name]()
        est.fit(vectorize_features(train_s), vectorize_labels(train_s))
        preds = est.predict(vectorize_features(test_s))
        try:
            probs = est.predict_proba(vectorize_features(test_s))
        except AttributeError:
            probs = None
        rows.append(classification_metrics(
            vectorize_labels(test_s), preds,
            probs if probs is not None else np.eye(3)[preds], num_classes=3))
    return _summarize(rows)


def run_graph_baseline(name: str, cfg: dict, samples) -> dict:
    device = resolve_device(cfg.get("device", "auto"))
    folds = subject_level_folds(samples, cfg.get("cv", {}).get("n_splits", 5))
    rows = []
    for fold, train_subjects, test_subjects in folds:
        set_seed(cfg.get("seed", 42) + fold)
        train = subset_by_subjects(samples, train_subjects)
        test = subset_by_subjects(samples, test_subjects)
        scaler = fit_scaler(train)
        train_s, test_s = apply_scaler(train, scaler), apply_scaler(test, scaler)
        model = GRAPH_BASELINES[name](
            in_channels=6, num_classes=3,
            hidden_dim=cfg.get("model", {}).get("hidden_dim", 128))
        model = _train_graph_model(model, train_s, device=device)
        rows.append(_eval_graph_model(model, test_s, device=device))
    return _summarize(rows)


def _summarize(rows: list[dict]) -> dict:
    out = {}
    for key in ("accuracy", "macro_f1", "baseline_f1", "seizure_f1",
                "recovery_f1", "macro_auc"):
        if rows and key in rows[0]:
            values = [float(r[key]) for r in rows]
            stats = mean_std_ci(values)
            out[key] = {"mean": stats["mean"], "sd": stats["std"], "folds": values}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Table 2 baselines.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "full_model.yaml"))
    parser.add_argument("--all", action="store_true", help="run every baseline")
    parser.add_argument("--only", default="", help="comma-separated baseline names")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.all:
        names = list(FEATURE_BASELINES) + list(GRAPH_BASELINES)
    else:
        names = [n.strip() for n in args.only.split(",") if n.strip()]
    if not names:
        parser.error("pass --all or --only <names>")

    samples = load_samples(cfg["data"]["labeled_dir"])
    print(f"loaded {len(samples)} labeled samples")

    out_dir = ROOT / "outputs" / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        set_seed(cfg.get("seed", 42))
        if name in FEATURE_BASELINES:
            summary = run_feature_baseline(name, cfg, samples)
        elif name in GRAPH_BASELINES:
            summary = run_graph_baseline(name, cfg, samples)
        else:
            raise SystemExit(f"unknown baseline: {name}")
        out_file = out_dir / f"{name}_summary.json"
        out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"{name}: acc={summary['accuracy']['mean']:.3f} "
              f"± {summary['accuracy']['sd']:.3f} -> {out_file}")


if __name__ == "__main__":
    main()
