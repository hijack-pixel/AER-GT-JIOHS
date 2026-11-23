import os
import re
import glob
import json
import math
import random
import csv
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import joblib
import numpy as np
import scipy.io
import torch
import torch.nn as nn
from torch.nn import Linear, TransformerEncoder, TransformerEncoderLayer
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


# =========================================================
# 0. Global config
# =========================================================
# Lambda-sensitivity copy of aergft/aergft_train_only_ascii.py (original untouched).
# Answers Reviewer 3, major comment 5: only AER_LAMBDA and OUTPUT_DIR differ
# from the original. Pretraining is lambda-independent (same SEED=42), so the
# pretrained backbone matches the main run.
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(PACKAGE_DIR)

PRETRAIN_PATH = "data/private/pretrain_con1_con2"   # adjust to your pretraining dataset
LABELED_PATH = "data/private/labeled_epilepsy"   # adjust to your labeled dataset
OUTPUT_DIR = os.path.join(PIPELINE_DIR, "aer_gft_cv_outputs_split_lambda0p2")

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_NODES = 22
IN_CHANNELS = 6
NUM_CLASSES = 3
CLASS_NAMES = ["Baseline", "Seizure", "Recovery"]

DIM_HIDDEN = 128
N_HEADS = 4
N_LAYERS = 2
TRANSFORMER_DROPOUT = 0.30
ATTN_TEMPERATURE = 1.5
ATTN_DROPOUT = 0.10
AER_LAMBDA = 0.2

PRETRAIN_EPOCHS = 20
FINETUNE_EPOCHS = 100
PRETRAIN_LR = 1e-3
FINETUNE_LR = 2e-3
WEIGHT_DECAY = 5e-3
PRETRAIN_BATCH_SIZE = 64
FINETUNE_BATCH_SIZE = 32
MASK_RATIO = 0.20
MAX_GRAD_NORM = 5.0

USE_ABS_ADJ = True
ADD_SELF_LOOP_IN_NORM = True
MAX_PRETRAIN_SAMPLES_FOR_SCALER = 5000
NUM_FOLDS = 5


# =========================================================
# 1. Reproducibility
# =========================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# 2. MATLAB field utilities
# =========================================================
def safe_get_string(mat_field) -> str:
    try:
        value = str(mat_field[0, 0][0])
    except Exception:
        try:
            value = str(mat_field[0])
        except Exception:
            value = str(mat_field)
    return value.strip()


def normalize_mouse_id(mouse_id: str) -> str:
    mouse_id = str(mouse_id).strip()
    mouse_id = re.sub(r"\s+", " ", mouse_id)
    return mouse_id


def safe_get_label(label_field) -> int:
    arr = np.asarray(label_field).squeeze()
    return int(arr.item())


@dataclass
class GraphSample:
    data: Data
    mouse_id: str
    source_file: str


# =========================================================
# 3. Model
# =========================================================
class AERGFT(nn.Module):
    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        num_classes: int = NUM_CLASSES,
        num_nodes: int = NUM_NODES,
        dim_hidden: int = DIM_HIDDEN,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        transformer_dropout: float = TRANSFORMER_DROPOUT,
        attn_temperature: float = ATTN_TEMPERATURE,
        attn_dropout: float = ATTN_DROPOUT,
        add_self_loop_in_norm: bool = ADD_SELF_LOOP_IN_NORM,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.dim_hidden = dim_hidden
        self.attn_temperature = float(attn_temperature)
        self.add_self_loop_in_norm = bool(add_self_loop_in_norm)
        self.attn_dropout = nn.Dropout(float(attn_dropout))

        self.node_embedding = Linear(in_channels, dim_hidden)
        self.res_proj = Linear(in_channels, dim_hidden)
        self.topo_proj = Linear(dim_hidden, dim_hidden)

        encoder_layer = TransformerEncoderLayer(
            d_model=dim_hidden,
            nhead=n_heads,
            dim_feedforward=dim_hidden * 2,
            dropout=transformer_dropout,
            batch_first=True,
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.reconstructor = nn.Sequential(
            Linear(dim_hidden, dim_hidden // 2),
            nn.ReLU(),
            Linear(dim_hidden // 2, in_channels),
        )

        self.attn_gate = nn.Sequential(
            Linear(dim_hidden, dim_hidden // 2),
            nn.Tanh(),
            Linear(dim_hidden // 2, 1),
        )

        self.classifier = nn.Sequential(
            Linear(dim_hidden, 64),
            nn.ReLU(),
            Linear(64, num_classes),
        )

    def normalize_adj(self, adj: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        adj = torch.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)

        if self.add_self_loop_in_norm:
            eye = torch.eye(adj.size(-1), device=adj.device).unsqueeze(0)
            adj = adj + eye

        deg = adj.sum(dim=-1)
        deg = torch.clamp(deg, min=eps)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt = torch.diag_embed(deg_inv_sqrt)

        adj_norm = torch.bmm(torch.bmm(deg_inv_sqrt, adj), deg_inv_sqrt)
        adj_norm = torch.nan_to_num(adj_norm, nan=0.0, posinf=0.0, neginf=0.0)
        return adj_norm

    def _attn_pool(self, node_out: torch.Tensor, return_alpha: bool = False):
        eps = 1e-12
        scores = self.attn_gate(node_out).squeeze(-1)
        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        tau = max(self.attn_temperature, 1e-6)
        alpha = torch.softmax(scores / tau, dim=1)
        alpha = self.attn_dropout(alpha)
        alpha = alpha / (alpha.sum(dim=1, keepdim=True) + eps)

        graph_feat = torch.sum(node_out * alpha.unsqueeze(-1), dim=1)
        graph_feat = torch.nan_to_num(graph_feat, nan=0.0, posinf=0.0, neginf=0.0)

        if return_alpha:
            return graph_feat, alpha
        return graph_feat

    def forward(self, data: Data, mode: str = "classify", return_attn: bool = False):
        x = data.x.view(-1, self.num_nodes, self.in_channels)
        adj = data.adj.view(-1, self.num_nodes, self.num_nodes)
        adj_norm = self.normalize_adj(adj)

        res = self.res_proj(x)
        h0 = self.node_embedding(x)

        h_topo = self.topo_proj(torch.bmm(adj_norm, h0))
        h_global = self.transformer(h0)
        node_out = h_topo + h_global + res
        node_out = torch.nan_to_num(node_out, nan=0.0, posinf=0.0, neginf=0.0)

        if mode == "reconstruct":
            out = self.reconstructor(node_out)
            out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            return out.view(-1, self.in_channels)

        graph_feat, alpha = self._attn_pool(node_out, return_alpha=True)
        logits = self.classifier(graph_feat)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

        if return_attn:
            return logits, alpha
        return logits


# =========================================================
# 4. Data loading
# =========================================================
def read_aert_sample(file_path: str, is_labeled: bool = True) -> Optional[GraphSample]:
    try:
        mat = scipy.io.loadmat(file_path)["AERT_Sample"]
    except Exception:
        return None

    try:
        feat = np.asarray(mat["NodeFeats"][0, 0], dtype=np.float32)
        adj = np.asarray(mat["Adjacency"][0, 0], dtype=np.float32)
        mouse_id = normalize_mouse_id(safe_get_string(mat["MouseID"]))
    except Exception:
        return None

    if feat.shape != (NUM_NODES, IN_CHANNELS):
        return None
    if adj.shape != (NUM_NODES, NUM_NODES):
        return None

    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    adj = np.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if USE_ABS_ADJ:
        adj = np.abs(adj)

    rows, cols = np.where(np.abs(adj) > 0)
    if len(rows) == 0:
        rows = np.arange(NUM_NODES)
        cols = np.arange(NUM_NODES)

    edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    y_val = safe_get_label(mat["Label"]) if is_labeled else -1

    data = Data(
        x=torch.tensor(feat, dtype=torch.float32),
        y=torch.tensor([y_val], dtype=torch.long),
        adj=torch.tensor(adj, dtype=torch.float32),
        edge_index=edge_index,
    )
    return GraphSample(data=data, mouse_id=mouse_id, source_file=file_path)


def load_graph_samples(folder: str, is_labeled: bool = True) -> List[GraphSample]:
    files = sorted(glob.glob(os.path.join(folder, "*.mat")))
    if not files:
        raise FileNotFoundError(f"No .mat files found in: {folder}")

    samples: List[GraphSample] = []
    skipped = 0

    for f in files:
        sample = read_aert_sample(f, is_labeled=is_labeled)
        if sample is None:
            skipped += 1
            continue
        samples.append(sample)

    print(f"Loaded {len(samples)} samples from {folder}; skipped {skipped} invalid files.")

    if not samples:
        raise RuntimeError(f"No valid graph samples loaded from: {folder}")

    return samples


def fit_scaler_from_pretrain_samples(pretrain_samples: Sequence[GraphSample]) -> StandardScaler:
    sample_subset = list(pretrain_samples)

    if len(sample_subset) > MAX_PRETRAIN_SAMPLES_FOR_SCALER:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(sample_subset), size=MAX_PRETRAIN_SAMPLES_FOR_SCALER, replace=False)
        sample_subset = [sample_subset[i] for i in idx]

    feats = [s.data.x.cpu().numpy() for s in sample_subset]
    scaler = StandardScaler()
    scaler.fit(np.vstack(feats))
    return scaler


def apply_scaler_to_samples(samples: Sequence[GraphSample], scaler: StandardScaler) -> List[GraphSample]:
    scaled_samples: List[GraphSample] = []

    for s in samples:
        x = s.data.x.cpu().numpy()
        x_scaled = scaler.transform(x)
        x_scaled = np.nan_to_num(x_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        data = Data(
            x=torch.tensor(x_scaled, dtype=torch.float32),
            y=s.data.y.clone(),
            adj=s.data.adj.clone(),
            edge_index=s.data.edge_index.clone(),
        )

        scaled_samples.append(GraphSample(data=data, mouse_id=s.mouse_id, source_file=s.source_file))

    return scaled_samples


# =========================================================
# 5. Fold utilities
# =========================================================
def get_subjects(samples: Sequence[GraphSample]) -> List[str]:
    return sorted({s.mouse_id for s in samples})


def subset_by_subjects(samples: Sequence[GraphSample], subjects: Sequence[str]) -> List[GraphSample]:
    subject_set = set(subjects)
    return [s for s in samples if s.mouse_id in subject_set]


def build_fold_splits(labeled_samples: Sequence[GraphSample], n_splits: int = NUM_FOLDS):
    subjects = get_subjects(labeled_samples)

    if len(subjects) < n_splits:
        raise ValueError(
            f"Need at least {n_splits} unique subjects for {n_splits}-fold CV, "
            f"got {len(subjects)}: {subjects}"
        )

    subject_rows = np.arange(len(subjects))
    subject_groups = np.array(subjects)
    splitter = GroupKFold(n_splits=n_splits)

    folds = []

    for fold_idx, (train_sub_idx, test_sub_idx) in enumerate(
        splitter.split(subject_rows, groups=subject_groups), start=1
    ):
        train_subjects = [subjects[i] for i in train_sub_idx]
        test_subjects = [subjects[i] for i in test_sub_idx]
        folds.append((fold_idx, train_subjects, test_subjects))

    return folds


# =========================================================
# 6. Training helpers
# =========================================================
def make_loader(samples: Sequence[GraphSample], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader([s.data for s in samples], batch_size=batch_size, shuffle=shuffle)


def pretrain_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int = PRETRAIN_EPOCHS,
) -> List[Dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=PRETRAIN_LR)
    criterion = nn.MSELoss()
    model.train()

    history = []

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        valid_batches = 0

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            orig_x = batch.x.view(-1, NUM_NODES, IN_CHANNELS).clone()
            x_masked = orig_x.clone()

            mask = torch.rand(orig_x.shape[:2], device=device) < MASK_RATIO
            if mask.sum() == 0:
                continue

            x_masked[mask] = 0.0
            batch.x = x_masked.view(-1, IN_CHANNELS)

            pred_x = model(batch, mode="reconstruct").view(-1, NUM_NODES, IN_CHANNELS)
            loss = criterion(pred_x[mask], orig_x[mask])

            if not torch.isfinite(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
            optimizer.step()

            total_loss += float(loss.item())
            valid_batches += 1

        avg_loss = total_loss / max(valid_batches, 1)
        history.append({"epoch": epoch, "pretrain_mse": avg_loss})
        print(f"    [Pretrain] Epoch {epoch:02d}/{epochs} | MSE: {avg_loss:.6f}")

    return history


def compute_aer_loss(
    alpha: torch.Tensor,
    num_nodes: int = NUM_NODES,
    lam: float = AER_LAMBDA,
) -> torch.Tensor:
    eps = 1e-12
    entropy = -(alpha * torch.log(alpha + eps)).sum(dim=1).mean()
    max_entropy = math.log(num_nodes)
    penalty = max_entropy - entropy
    return lam * penalty


def finetune_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = FINETUNE_EPOCHS,
) -> List[Dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)
    ce_loss = nn.CrossEntropyLoss()
    model.train()

    history = []

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        valid_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            labels = batch.y.view(-1)

            optimizer.zero_grad()

            logits, alpha = model(batch, mode="classify", return_attn=True)
            loss = ce_loss(logits, labels) + compute_aer_loss(alpha, num_nodes=NUM_NODES, lam=AER_LAMBDA)

            if not torch.isfinite(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
            optimizer.step()

            total_loss += float(loss.item())
            valid_batches += 1

        avg_loss = total_loss / max(valid_batches, 1)
        history.append({"epoch": epoch, "finetune_loss": avg_loss})

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(f"    [Finetune] Epoch {epoch:03d}/{epochs} | Loss: {avg_loss:.6f}")

    return history


# =========================================================
# 7. Output helpers
# =========================================================
def save_json(obj: Dict[str, object], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=True, indent=2)


def write_training_history_csv(output_dir: str, rows: List[Dict[str, object]]) -> None:
    path = os.path.join(output_dir, "training_history.csv")
    if not rows:
        return

    fieldnames = ["fold", "stage", "epoch", "pretrain_mse", "finetune_loss"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f">>> Training history saved to: {path}")


# =========================================================
# 8. Train-only pipeline
# =========================================================
def run_training_only(
    pretrain_path: str = PRETRAIN_PATH,
    labeled_path: str = LABELED_PATH,
    output_dir: str = OUTPUT_DIR,
) -> Dict[str, object]:
    os.makedirs(output_dir, exist_ok=True)
    set_seed(SEED)

    print("\n>>> Loading datasets")
    pretrain_raw = load_graph_samples(pretrain_path, is_labeled=False)
    labeled_raw = load_graph_samples(labeled_path, is_labeled=True)

    folds = build_fold_splits(labeled_raw, n_splits=NUM_FOLDS)
    subjects = get_subjects(labeled_raw)

    print(f">>> Found {len(subjects)} labeled subjects for {NUM_FOLDS}-fold subject-level CV")

    manifest: Dict[str, object] = {
        "mode": "train_only",
        "pretrain_path": pretrain_path,
        "labeled_path": labeled_path,
        "output_dir": output_dir,
        "seed": SEED,
        "device": str(DEVICE),
        "num_folds": NUM_FOLDS,
        "subjects": subjects,
        "folds": [],
        "config": {
            "num_nodes": NUM_NODES,
            "in_channels": IN_CHANNELS,
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "dim_hidden": DIM_HIDDEN,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "transformer_dropout": TRANSFORMER_DROPOUT,
            "attn_temperature": ATTN_TEMPERATURE,
            "attn_dropout": ATTN_DROPOUT,
            "aer_lambda": AER_LAMBDA,
            "pretrain_epochs": PRETRAIN_EPOCHS,
            "finetune_epochs": FINETUNE_EPOCHS,
            "pretrain_lr": PRETRAIN_LR,
            "finetune_lr": FINETUNE_LR,
            "weight_decay": WEIGHT_DECAY,
            "pretrain_batch_size": PRETRAIN_BATCH_SIZE,
            "finetune_batch_size": FINETUNE_BATCH_SIZE,
            "mask_ratio": MASK_RATIO,
            "max_grad_norm": MAX_GRAD_NORM,
            "use_abs_adj": USE_ABS_ADJ,
            "add_self_loop_in_norm": ADD_SELF_LOOP_IN_NORM,
            "max_pretrain_samples_for_scaler": MAX_PRETRAIN_SAMPLES_FOR_SCALER,
        },
    }

    all_history_rows: List[Dict[str, object]] = []

    for fold_idx, train_subjects, test_subjects in folds:
        print("\n" + "=" * 80)
        print(f"Fold {fold_idx}/{NUM_FOLDS}")
        print(f"  Train subjects: {train_subjects}")
        print(f"  Test subjects : {test_subjects}")

        scaler = fit_scaler_from_pretrain_samples(pretrain_raw)
        scaler_path = os.path.join(output_dir, f"fold_{fold_idx}_scaler.pkl")
        joblib.dump(scaler, scaler_path)

        pretrain_samples = apply_scaler_to_samples(pretrain_raw, scaler)
        train_raw = subset_by_subjects(labeled_raw, train_subjects)
        test_raw = subset_by_subjects(labeled_raw, test_subjects)
        train_samples = apply_scaler_to_samples(train_raw, scaler)

        print(f"  Pretrain samples: {len(pretrain_samples)}")
        print(f"  Train samples   : {len(train_samples)}")
        print(f"  Test samples    : {len(test_raw)} [held out; not evaluated here]")

        model = AERGFT().to(DEVICE)

        pre_loader = make_loader(pretrain_samples, PRETRAIN_BATCH_SIZE, shuffle=True)
        train_loader = make_loader(train_samples, FINETUNE_BATCH_SIZE, shuffle=True)

        pre_hist = pretrain_model(model, pre_loader, DEVICE, epochs=PRETRAIN_EPOCHS)
        pretrained_path = os.path.join(output_dir, f"fold_{fold_idx}_pretrained.pth")
        torch.save(model.state_dict(), pretrained_path)

        fine_hist = finetune_model(model, train_loader, DEVICE, epochs=FINETUNE_EPOCHS)
        final_model_path = os.path.join(output_dir, f"fold_{fold_idx}_final_model.pth")
        torch.save(model.state_dict(), final_model_path)

        for item in pre_hist:
            all_history_rows.append({"fold": fold_idx, "stage": "pretrain", **item})
        for item in fine_hist:
            all_history_rows.append({"fold": fold_idx, "stage": "finetune", **item})

        manifest["folds"].append(
            {
                "fold": fold_idx,
                "train_subjects": train_subjects,
                "test_subjects": test_subjects,
                "scaler_path": scaler_path,
                "pretrained_model_path": pretrained_path,
                "final_model_path": final_model_path,
                "pretrain_samples": len(pretrain_samples),
                "train_samples": len(train_samples),
                "test_samples": len(test_raw),
            }
        )

        save_json(manifest, os.path.join(output_dir, "train_manifest.json"))
        write_training_history_csv(output_dir, all_history_rows)

    print("\n>>> Training-only pipeline finished.")
    save_json(manifest, os.path.join(output_dir, "train_manifest.json"))
    write_training_history_csv(output_dir, all_history_rows)

    print(f">>> Models and scalers saved to: {output_dir}")
    print(f">>> Train manifest saved to: {os.path.join(output_dir, 'train_manifest.json')}")

    return manifest


def main() -> None:
    run_training_only(
        pretrain_path=PRETRAIN_PATH,
        labeled_path=LABELED_PATH,
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
