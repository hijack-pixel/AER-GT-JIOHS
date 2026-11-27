"""Run the baseline models of Table 2 under the same subject-level folds as AER-GT (Sec. 2.7).

Feature-based baselines (flattened ROI features):
    logistic_regression | xgboost | mlp
Graph-based baselines (node features and adjacency matrices as AER-GT):
    supervised_gcn | supervised_gat | self_supervised_gat | self_supervised_bnt

Usage:
    python scripts/run_baselines.py --config configs/full_model.yaml --all
    python scripts/run_baselines.py --config configs/full_model.yaml --only xgboost,mlp

Inputs are the labeled window samples of the configuration. For every baseline
the per-fold metrics are written in the format of ``fold_metrics.csv`` so that
``scripts/supplementary_analyses/paired_baseline_comparison.py`` can pair them with
the AER-GT run, together with a summary of means and SDs.

Output:
    outputs/<run>/evaluation/baselines/<name>/fold_metrics.csv
    outputs/<run>/evaluation/baselines/<name>/summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aergt.config import load_config  # noqa: E402
from aergt.data.samples import apply_scaler, fit_scaler, load_samples  # noqa: E402
from aergt.data.splits import subject_level_folds, subset_by_subjects  # noqa: E402
from aergt.evaluation.metrics import METRIC_NAMES, classification_metrics, mean_std_ci  # noqa: E402
from aergt.models.baselines import FEATURE_BASELINES, GRAPH_BASELINES, SelfSupervisedWrapper  # noqa: E402
from aergt.utils.runtime import resolve_device, set_seed  # noqa: E402


def _flatten_graph(samples):
    """Graph tensors for the graph-based baselines."""
    x = torch.tensor(np.stack([s.x for s in samples]), dtype=torch.float32)
    adj = torch.tensor(np.stack([s.adj for s in samples]), dtype=torch.float32)
    y = torch.tensor([s.label for s in samples], dtype=torch.long)
    return x, adj, y


def _train_graph_model(model, train_samples, *, device, epochs=40, lr=1e-3, mask_ratio=0.20, pretrain_epochs=20):
    """Two-stage training of the self-supervised baselines: masked node-feature
    reconstruction on the training windows, then supervised fine-tuning. The
    supervised baselines use the fine-tuning stage only."""
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x, adj, y = _flatten_graph(train_samples)
    x, adj, y = x.to(device), adj.to(device), y.to(device)

    if isinstance(model, SelfSupervisedWrapper):
        n_masked = max(1, int(round(mask_ratio * x.shape[1])))
        model.train()
        for _ in range(pretrain_epochs):
            opt.zero_grad()
            mask = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool)
            for index in range(x.shape[0]):
                mask[index, torch.randperm(x.shape[1])[:n_masked]] = True
            mask = mask.to(device)
            x_masked = x.clone()
            x_masked[mask] = 0.0
            recon = model(x_masked, adj, mode="reconstruct")
            loss = torch.nn.functional.mse_loss(recon[mask], x[mask])
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
def _eval_graph_model(model, samples, *, device, num_classes):
    model.eval()
    x, adj, y = _flatten_graph(samples)
    x, adj, y = x.to(device), adj.to(device), y.to(device)
    logits = model(x, adj)
    preds = logits.argmax(dim=1).cpu().numpy()
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    return classification_metrics(y.cpu().numpy(), preds, probs, num_classes=num_classes)


def run_feature_baseline(name: str, cfg: dict, samples, folds) -> list[dict]:
    from aergt.models.baselines import vectorize_features, vectorize_labels

    seed = int(cfg["project"]["seed"])
    scaler_cfg = cfg["training"]
    rows = []
    for fold, train_subjects, test_subjects in folds:
        train = subset_by_subjects(samples, train_subjects)
        test = subset_by_subjects(samples, test_subjects)
        scaler = fit_scaler(train, max_samples=int(scaler_cfg["max_samples_for_scaler"]), seed=seed)
        train_s, test_s = apply_scaler(train, scaler), apply_scaler(test, scaler)
        estimator = FEATURE_BASELINES[name](random_state=seed)
        estimator.fit(vectorize_features(train_s), vectorize_labels(train_s))
        preds = estimator.predict(vectorize_features(test_s))
        try:
            probs = estimator.predict_proba(vectorize_features(test_s))
        except AttributeError:
            probs = None
        metrics = classification_metrics(vectorize_labels(test_s), preds,
                                         probs if probs is not None else np.eye(int(cfg["model"]["num_classes"]))[preds],
                                         num_classes=int(cfg["model"]["num_classes"]))
        rows.append({"fold": fold, "held_out_subjects": ";".join(test_subjects), "n_windows": len(test), **_flat(metrics)})
    return rows


def _flat(metrics: dict) -> dict:
    return {key: value for key, value in metrics.items() if key != "confusion_matrix"}


def run_graph_baseline(name: str, cfg: dict, samples, folds) -> list[dict]:
    device = resolve_device(str(cfg["hardware"]["device"]))
    seed = int(cfg["project"]["seed"])
    scaler_cfg = cfg["training"]
    num_classes = int(cfg["model"]["num_classes"])
    rows = []
    for fold, train_subjects, test_subjects in folds:
        set_seed(seed + fold)
        train = subset_by_subjects(samples, train_subjects)
        test = subset_by_subjects(samples, test_subjects)
        scaler = fit_scaler(train, max_samples=int(scaler_cfg["max_samples_for_scaler"]), seed=seed)
        train_s, test_s = apply_scaler(train, scaler), apply_scaler(test, scaler)
        model = GRAPH_BASELINES[name](in_channels=int(cfg["model"]["in_channels"]), num_classes=num_classes,
                                      hidden_dim=int(cfg["model"]["hidden_dim"]))
        model = _train_graph_model(model, train_s, device=device, mask_ratio=float(cfg["training"]["mask_ratio"]))
        rows.append({"fold": fold, "held_out_subjects": ";".join(test_subjects), "n_windows": len(test),
                     **_flat(_eval_graph_model(model, test_s, device=device, num_classes=num_classes))})
    return rows


def _summarize(rows: list[dict]) -> dict:
    out = {}
    for key in METRIC_NAMES:
        if rows and key in rows[0]:
            stats = mean_std_ci([float(r[key]) for r in rows])
            out[key] = {"mean": stats["mean"], "sd": stats["std"], "ci95": [stats["ci95_low"], stats["ci95_high"]],
                        "folds": [float(r[key]) for r in rows]}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Table 2 baselines.")
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
    unknown = [name for name in names if name not in FEATURE_BASELINES and name not in GRAPH_BASELINES]
    if unknown:
        parser.error(f"unknown baseline(s): {unknown}")

    samples = load_samples(ROOT / cfg["data"]["labeled_dir"], labeled=True, num_nodes=int(cfg["model"]["num_nodes"]),
                           in_channels=int(cfg["model"]["in_channels"]), use_abs_adj=bool(cfg["data"]["use_abs_adjacency"]))
    folds = subject_level_folds(samples, int(cfg["training"]["num_folds"]))
    set_seed(int(cfg["project"]["seed"]))

    for name in names:
        rows = (run_feature_baseline(name, cfg, samples, folds) if name in FEATURE_BASELINES
                else run_graph_baseline(name, cfg, samples, folds))
        out_dir = ROOT / cfg["project"]["output_dir"] / "evaluation" / "baselines" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "fold_metrics.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = _summarize(rows)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"{name}: acc={summary['accuracy']['mean']:.3f} +/- {summary['accuracy']['sd']:.3f} -> {out_dir}")


if __name__ == "__main__":
    main()
