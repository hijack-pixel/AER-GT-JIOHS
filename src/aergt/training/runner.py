from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from torch import nn

from aergt.data.samples import (
    GraphSample,
    apply_scaler,
    fit_scaler,
    load_samples,
    make_loader,
)
from aergt.data.scalers import save_standardizer
from aergt.data.splits import subject_level_folds, subjects, subset_by_subjects
from aergt.models import AERGT
from aergt.training.losses import aer_regularization
from aergt.utils.runtime import configure_torch, resolve_device, runtime_manifest, set_seed, write_json


def _make_grad_scaler(device: torch.device, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device.type, enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(device: torch.device, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def _model_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    model_cfg = cfg["model"]
    ablation_cfg = cfg.get("ablation", {})
    return {
        "in_channels": model_cfg["in_channels"],
        "num_classes": model_cfg["num_classes"],
        "num_nodes": model_cfg["num_nodes"],
        "hidden_dim": model_cfg["hidden_dim"],
        "num_heads": model_cfg["num_heads"],
        "num_layers": model_cfg["num_layers"],
        "transformer_dropout": model_cfg["transformer_dropout"],
        "attention_temperature": model_cfg["attention_temperature"],
        "attention_dropout": model_cfg["attention_dropout"],
        "add_self_loop_in_norm": model_cfg["add_self_loop_in_norm"],
        "use_topology_guided_aggregation": ablation_cfg.get("use_topology_guided_aggregation", True),
    }


def _batch_to_device(batch: dict[str, object], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = batch["x"].to(device, non_blocking=True)
    adj = batch["adj"].to(device, non_blocking=True)
    y = batch["y"].to(device, non_blocking=True)
    return x, adj, y


def pretrain_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler,
    *,
    device: torch.device,
    cfg: dict[str, Any],
) -> float:
    model.train()
    criterion = nn.MSELoss()
    mask_ratio = float(cfg["training"]["mask_ratio"])
    max_grad_norm = float(cfg["training"]["max_grad_norm"])
    amp_enabled = bool(cfg["hardware"]["amp"]) and device.type == "cuda"

    total_loss = 0.0
    valid_batches = 0
    for batch in loader:
        x, adj, _ = _batch_to_device(batch, device)
        original_x = x.clone()
        x_masked = x.clone()
        mask = torch.rand(x.shape[:2], device=device) < mask_ratio
        if mask.sum() == 0:
            continue
        x_masked[mask] = 0.0

        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp_enabled):
            pred = model(x_masked, adj, mode="reconstruct")
            loss = criterion(pred[mask], original_x[mask])
        if not torch.isfinite(loss):
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.detach().cpu())
        valid_batches += 1
    return total_loss / max(valid_batches, 1)


def finetune_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler,
    *,
    device: torch.device,
    cfg: dict[str, Any],
) -> float:
    model.train()
    ce_loss = nn.CrossEntropyLoss()
    max_grad_norm = float(cfg["training"]["max_grad_norm"])
    amp_enabled = bool(cfg["hardware"]["amp"]) and device.type == "cuda"
    num_nodes = int(cfg["model"]["num_nodes"])
    aer_weight = float(cfg["training"]["aer_lambda"])
    if not cfg.get("ablation", {}).get("use_aer_regularization", True):
        aer_weight = 0.0

    total_loss = 0.0
    valid_batches = 0
    for batch in loader:
        x, adj, y = _batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp_enabled):
            logits, alpha = model(x, adj, mode="classify", return_attention=True)
            loss = ce_loss(logits, y) + aer_regularization(alpha, num_nodes=num_nodes, weight=aer_weight)
        if not torch.isfinite(loss):
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.detach().cpu())
        valid_batches += 1
    return total_loss / max(valid_batches, 1)


def _write_history(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["fold", "stage", "epoch", "loss"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _save_checkpoint(path: Path, model: nn.Module, cfg: dict[str, Any], fold: int, stage: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "fold": fold,
            "stage": stage,
        },
        path,
    )


def train_from_config(cfg: dict[str, Any], *, repo_root: str | Path) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    output_dir = (repo_root / cfg["project"]["output_dir"]).resolve()
    checkpoint_dir = output_dir / "checkpoints"
    scaler_dir = output_dir / "scalers"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    scaler_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(cfg["project"]["seed"]), deterministic=bool(cfg["hardware"]["deterministic"]))
    configure_torch(tf32=bool(cfg["hardware"]["tf32"]))
    device = resolve_device(str(cfg["hardware"]["device"]))

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    hardware_cfg = cfg["hardware"]
    ablation_cfg = cfg.get("ablation", {})

    labeled_samples = load_samples(
        repo_root / data_cfg["labeled_dir"],
        labeled=True,
        num_nodes=int(model_cfg["num_nodes"]),
        in_channels=int(model_cfg["in_channels"]),
        use_abs_adj=bool(data_cfg["use_abs_adjacency"]),
    )
    # The unlabeled pretraining samples provide the standardization parameters in
    # every configuration (Sec. 2.6); the pretraining stage itself runs unless it
    # is ablated.
    pretrain_dir = repo_root / data_cfg["pretrain_dir"]
    pretrain_samples: list[GraphSample] = []
    if pretrain_dir.is_dir() and any(pretrain_dir.glob("*.mat")):
        pretrain_samples = load_samples(
            pretrain_dir,
            labeled=False,
            num_nodes=int(model_cfg["num_nodes"]),
            in_channels=int(model_cfg["in_channels"]),
            use_abs_adj=bool(data_cfg["use_abs_adjacency"]),
        )
    use_pretraining = bool(ablation_cfg.get("use_pretraining", True))
    if not pretrain_samples:
        if use_pretraining:
            print("NOTE: no pretraining samples found; pretraining is skipped and the scaler "
                  "is fitted on the training windows.")
        use_pretraining = False

    folds = subject_level_folds(labeled_samples, int(train_cfg["num_folds"]))
    history_rows: list[dict[str, object]] = []
    manifest: dict[str, Any] = {
        "mode": "train",
        "runtime": runtime_manifest(repo_root, cfg),
        "device": str(device),
        "subjects": subjects(labeled_samples),
        "folds": [],
    }

    for fold_idx, train_subjects, test_subjects in folds:
        print("=" * 80)
        print(f"Fold {fold_idx}/{train_cfg['num_folds']}")
        print(f"Train subjects: {train_subjects}")
        print(f"Held-out subjects: {test_subjects}")

        train_raw = subset_by_subjects(labeled_samples, train_subjects)
        test_raw = subset_by_subjects(labeled_samples, test_subjects)
        scaler_source = pretrain_samples if pretrain_samples else train_raw
        standardizer = fit_scaler(
            scaler_source,
            max_samples=int(train_cfg["max_samples_for_scaler"]),
            seed=int(cfg["project"]["seed"]),
        )
        scaler_path = scaler_dir / f"fold_{fold_idx}_scaler.pkl"
        save_standardizer(standardizer, scaler_path)

        train_scaled = apply_scaler(train_raw, standardizer)
        pretrain_scaled = apply_scaler(pretrain_samples, standardizer) if use_pretraining else []

        model = AERGT(**_model_kwargs(cfg)).to(device)
        amp_scaler = _make_grad_scaler(device, enabled=bool(hardware_cfg["amp"]) and device.type == "cuda")

        if pretrain_scaled:
            pre_loader = make_loader(
                pretrain_scaled,
                batch_size=int(train_cfg["pretrain_batch_size"]),
                shuffle=True,
                num_workers=int(hardware_cfg["num_workers"]),
                pin_memory=bool(hardware_cfg["pin_memory"]) and device.type == "cuda",
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg["pretrain_lr"]))
            for epoch in range(1, int(train_cfg["pretrain_epochs"]) + 1):
                loss = pretrain_epoch(model, pre_loader, optimizer, amp_scaler, device=device, cfg=cfg)
                history_rows.append({"fold": fold_idx, "stage": "pretrain", "epoch": epoch, "loss": loss})
                print(f"[fold {fold_idx}] pretrain {epoch:03d}: {loss:.6f}")
            _save_checkpoint(checkpoint_dir / f"fold_{fold_idx}_pretrained.pth", model, cfg, fold_idx, "pretrained")

        train_loader = make_loader(
            train_scaled,
            batch_size=int(train_cfg["finetune_batch_size"]),
            shuffle=True,
            num_workers=int(hardware_cfg["num_workers"]),
            pin_memory=bool(hardware_cfg["pin_memory"]) and device.type == "cuda",
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(train_cfg["finetune_lr"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )
        for epoch in range(1, int(train_cfg["finetune_epochs"]) + 1):
            loss = finetune_epoch(model, train_loader, optimizer, amp_scaler, device=device, cfg=cfg)
            history_rows.append({"fold": fold_idx, "stage": "finetune", "epoch": epoch, "loss": loss})
            if epoch == 1 or epoch % 10 == 0 or epoch == int(train_cfg["finetune_epochs"]):
                print(f"[fold {fold_idx}] finetune {epoch:03d}: {loss:.6f}")

        final_model_path = checkpoint_dir / f"fold_{fold_idx}_final_model.pth"
        _save_checkpoint(final_model_path, model, cfg, fold_idx, "final")
        manifest["folds"].append(
            {
                "fold": fold_idx,
                "train_subjects": train_subjects,
                "test_subjects": test_subjects,
                "scaler_path": str(scaler_path.relative_to(output_dir)),
                "final_model_path": str(final_model_path.relative_to(output_dir)),
                "pretrain_samples": len(pretrain_scaled),
                "train_samples": len(train_raw),
                "test_samples": len(test_raw),
            }
        )
        write_json(manifest, output_dir / "train_manifest.json")
        _write_history(output_dir / "training_history.csv", history_rows)

    write_json(manifest, output_dir / "train_manifest.json")
    _write_history(output_dir / "training_history.csv", history_rows)
    return manifest
