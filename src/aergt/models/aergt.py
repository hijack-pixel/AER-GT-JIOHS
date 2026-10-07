from __future__ import annotations

import torch
from torch import nn


class AERGT(nn.Module):
    """Attention-enhanced ROI graph transformer used for IOS state decoding."""

    def __init__(
        self,
        *,
        in_channels: int = 6,
        num_classes: int = 3,
        num_nodes: int = 22,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        transformer_dropout: float = 0.30,
        attention_temperature: float = 1.5,
        attention_dropout: float = 0.10,
        add_self_loop_in_norm: bool = True,
        use_topology_guided_aggregation: bool = True,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.attention_temperature = float(attention_temperature)
        self.add_self_loop_in_norm = bool(add_self_loop_in_norm)
        self.use_topology_guided_aggregation = bool(use_topology_guided_aggregation)

        self.node_embedding = nn.Linear(in_channels, hidden_dim)
        # Keep these names compatible with the paper-trained state_dict files.
        self.res_proj = nn.Linear(in_channels, hidden_dim)
        self.topo_proj = nn.Linear(hidden_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=transformer_dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.reconstructor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, in_channels),
        )
        self.attn_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def normalize_adjacency(self, adj: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        adj = torch.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)
        if self.add_self_loop_in_norm:
            eye = torch.eye(adj.size(-1), device=adj.device, dtype=adj.dtype).unsqueeze(0)
            adj = adj + eye

        deg = torch.clamp(adj.sum(dim=-1), min=eps)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt = torch.diag_embed(deg_inv_sqrt)
        adj_norm = torch.bmm(torch.bmm(deg_inv_sqrt, adj), deg_inv_sqrt)
        return torch.nan_to_num(adj_norm, nan=0.0, posinf=0.0, neginf=0.0)

    def encode_nodes(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        adj_norm = self.normalize_adjacency(adj)
        residual = self.res_proj(x)
        h0 = self.node_embedding(x)

        h_topology = self.topo_proj(torch.bmm(adj_norm, h0))
        h_global = self.transformer(h0)
        if self.use_topology_guided_aggregation:
            node_out = h_topology + h_global + residual
        else:
            node_out = h_global + residual
        return torch.nan_to_num(node_out, nan=0.0, posinf=0.0, neginf=0.0)

    def attention_pool(self, node_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        eps = 1e-12
        scores = self.attn_gate(node_out).squeeze(-1)
        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        tau = max(self.attention_temperature, 1e-6)
        alpha = torch.softmax(scores / tau, dim=1)
        alpha = self.attn_dropout(alpha)
        alpha = alpha / (alpha.sum(dim=1, keepdim=True) + eps)
        graph_feat = torch.sum(node_out * alpha.unsqueeze(-1), dim=1)
        graph_feat = torch.nan_to_num(graph_feat, nan=0.0, posinf=0.0, neginf=0.0)
        return graph_feat, alpha

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        *,
        mode: str = "classify",
        return_attention: bool = False,
    ):
        node_out = self.encode_nodes(x, adj)
        if mode == "reconstruct":
            recon = self.reconstructor(node_out)
            return torch.nan_to_num(recon, nan=0.0, posinf=0.0, neginf=0.0)

        graph_feat, alpha = self.attention_pool(node_out)
        logits = self.classifier(graph_feat)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        if return_attention:
            return logits, alpha
        return logits
