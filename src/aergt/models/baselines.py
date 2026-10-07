"""Baseline models compared against AER-GT in Table 2 (Sec. 2.7).

Two groups, exactly as described in the manuscript:

* Feature-based models (vectorized ROI features, i.e. the flattened
  ``[num_nodes * in_channels]`` node-feature matrix):
  - Logistic Regression
  - XGBoost
  - MLP
* Graph-based models (same node-feature matrices and adjacency matrices as
  AER-GT):
  - supervised GCN
  - supervised GAT
  - self-supervised GAT (two-stage: reconstruction pretraining + finetuning)
  - self-supervised BNT (Brain-Network-Transformer-style backbone with
    two-stage training)

All baselines consume the same subject-level five-fold splits as AER-GT
(see ``aergt.data.splits.subject_level_folds``); nothing in this module
performs its own splitting.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# feature vectorization (Sec. 2.7: "Feature-based models used vectorized
# ROI features")
# --------------------------------------------------------------------------- #

def vectorize_features(samples) -> np.ndarray:
    """Flatten the per-window node-feature matrix ``x`` into one vector."""
    return np.stack([np.asarray(s.x, dtype=np.float64).reshape(-1) for s in samples])


def vectorize_labels(samples) -> np.ndarray:
    return np.asarray([s.label for s in samples], dtype=np.int64)


# --------------------------------------------------------------------------- #
# feature-based baselines (sklearn / xgboost)
# --------------------------------------------------------------------------- #

def make_logistic_regression():
    """Logistic Regression baseline (multinomial)."""
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=2000, C=1.0)


def make_xgboost():
    """XGBoost baseline. Requires the optional ``xgboost`` dependency."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "The XGBoost baseline requires the 'xgboost' package "
            "(pip install xgboost>=1.7)."
        ) from exc
    return XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.1,
        eval_metric="mlogloss", use_label_encoder=False,
    )


def make_mlp():
    """MLP baseline."""
    from sklearn.neural_network import MLPClassifier

    return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, early_stopping=True)


FEATURE_BASELINES = {
    "logistic_regression": make_logistic_regression,
    "xgboost": make_xgboost,
    "mlp": make_mlp,
}

# --------------------------------------------------------------------------- #
# graph-based baselines (PyTorch)
# --------------------------------------------------------------------------- #

def _sym_normalize_adj(adj: torch.Tensor) -> torch.Tensor:
    """D^(-1/2) (A + I) D^(-1/2) normalization used by the GCN baseline."""
    a = adj.clone()
    a = a + torch.eye(a.shape[-1], device=a.device, dtype=a.dtype)
    deg = a.sum(dim=-1).clamp_min(1e-8).pow(-0.5)
    return a * deg.unsqueeze(-1) * deg.unsqueeze(-2)


class GCN(nn.Module):
    """Supervised two-layer graph convolutional network baseline."""

    def __init__(self, *, in_channels: int = 6, num_classes: int = 3, hidden_dim: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, *, return_attention: bool = False):
        h = F.relu(torch.matmul(_sym_normalize_adj(adj), self.fc1(x)))
        h = F.relu(torch.matmul(_sym_normalize_adj(adj), self.fc2(h)))
        pooled = h.mean(dim=1)                       # mean over ROIs
        logits = self.classifier(pooled)
        if return_attention:                         # uniform, kept for API parity
            alpha = torch.full((x.shape[0], x.shape[1]), 1.0 / x.shape[1],
                               device=x.device, dtype=x.dtype)
            return logits, alpha
        return logits


class GATLayer(nn.Module):
    """Single-head graph attention layer (Velickovic et al.)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Linear(out_dim, 1, bias=False)
        self.a_dst = nn.Linear(out_dim, 1, bias=False)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.W(x)                                            # B,N,D
        # additive attention score e_ij = leaky_relu(a_src*h_i + a_dst*h_j)
        s = self.a_src(h)                                        # B,N,1
        d = self.a_dst(h).transpose(1, 2)                        # B,1,N
        e = F.leaky_relu(s + d, negative_slope=0.2)
        mask = (adj > 0)
        e = e.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(e, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        out = torch.matmul(attn, h)
        return out, attn


class GAT(nn.Module):
    """Supervised graph attention network baseline (two GAT layers, mean pool)."""

    def __init__(self, *, in_channels: int = 6, num_classes: int = 3, hidden_dim: int = 128) -> None:
        super().__init__()
        self.layer1 = GATLayer(in_channels, hidden_dim)
        self.layer2 = GATLayer(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, *, return_attention: bool = False):
        h, _ = self.layer1(x, adj)
        h = F.elu(h)
        h, attn2 = self.layer2(h, adj)
        pooled = h.mean(dim=1)
        logits = self.classifier(pooled)
        if return_attention:
            return logits, attn2.mean(dim=-1)
        return logits


class BNTBackbone(nn.Module):
    """Brain-Network-Transformer-style backbone (Kan et al., ref. 36):
    ROI embedding + global Transformer encoder + orthonormal clustering
    readout, used here as the self-supervised BNT baseline."""

    def __init__(self, *, in_channels: int = 6, hidden_dim: int = 128,
                 num_heads: int = 4, num_clusters: int = 4,
                 transformer_dropout: float = 0.3) -> None:
        super().__init__()
        self.embedding = nn.Linear(in_channels, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=4 * hidden_dim, dropout=transformer_dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        # orthonormal clustering head (orthogonal basis of ROI loadings)
        self.cluster_basis = nn.Parameter(torch.randn(num_clusters, hidden_dim))

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        h = self.embedding(x)
        h = self.encoder(h)
        basis, _ = torch.linalg.qr(self.cluster_basis.T)        # orthonormalize
        scores = torch.softmax(h @ basis, dim=-1)               # B,N,K
        pooled = scores.transpose(1, 2) @ h                     # B,K,D
        return pooled.reshape(pooled.shape[0], -1), h


class SelfSupervisedWrapper(nn.Module):
    """Two-stage (self-supervised pretraining + supervised finetuning)
    wrapper shared by the self-supervised GAT and BNT baselines.

    Pretraining objective: masked node-feature reconstruction, mirroring the
    AER-GT two-stage protocol so that the only difference between the
    self-supervised baselines and AER-GT is the backbone."""

    def __init__(self, backbone: str, *, in_channels: int = 6, num_classes: int = 3,
                 hidden_dim: int = 128) -> None:
        super().__init__()
        self.backbone_name = backbone
        if backbone == "gat":
            self.gat1 = GATLayer(in_channels, hidden_dim)
            self.gat2 = GATLayer(hidden_dim, hidden_dim)
            feat_dim = hidden_dim
        elif backbone == "bnt":
            self.bnt = BNTBackbone(in_channels=in_channels, hidden_dim=hidden_dim)
            feat_dim = hidden_dim * self.bnt.cluster_basis.shape[0]
        else:
            raise ValueError(f"unknown backbone: {backbone}")
        self.reconstructor = nn.Linear(hidden_dim, in_channels)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def encode(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if self.backbone_name == "gat":
            h, _ = self.gat1(x, adj)
            h = F.elu(h)
            h, attn2 = self.gat2(h, adj)
            return h.mean(dim=1)
        pooled, _ = self.bnt(x, adj)
        return pooled

    def node_encode(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h, _ = self.gat1(x, adj)
        return F.elu(h)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, *, mode: str = "classify",
                return_attention: bool = False):
        if mode == "reconstruct":
            if self.backbone_name == "gat":
                h = self.node_encode(x, adj)
                recon = self.reconstructor(h)
            else:
                _, h = self.bnt(x, adj)
                recon = self.reconstructor(h)
            return torch.nan_to_num(recon, nan=0.0, posinf=0.0, neginf=0.0)
        feat = self.encode(x, adj)
        logits = self.classifier(feat)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        if return_attention:
            alpha = torch.full((x.shape[0], x.shape[1]), 1.0 / x.shape[1],
                               device=x.device, dtype=x.dtype)
            return logits, alpha
        return logits


GRAPH_BASELINES = {
    "supervised_gcn": lambda in_channels, num_classes, hidden_dim: GCN(
        in_channels=in_channels, num_classes=num_classes, hidden_dim=hidden_dim),
    "supervised_gat": lambda in_channels, num_classes, hidden_dim: GAT(
        in_channels=in_channels, num_classes=num_classes, hidden_dim=hidden_dim),
    "self_supervised_gat": lambda in_channels, num_classes, hidden_dim: SelfSupervisedWrapper(
        "gat", in_channels=in_channels, num_classes=num_classes, hidden_dim=hidden_dim),
    "self_supervised_bnt": lambda in_channels, num_classes, hidden_dim: SelfSupervisedWrapper(
        "bnt", in_channels=in_channels, num_classes=num_classes, hidden_dim=hidden_dim),
}
