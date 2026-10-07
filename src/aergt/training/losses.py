from __future__ import annotations

import math

import torch


def aer_regularization(alpha: torch.Tensor, *, num_nodes: int, weight: float) -> torch.Tensor:
    if weight <= 0:
        return alpha.new_tensor(0.0)
    eps = 1e-12
    entropy = -(alpha * torch.log(alpha + eps)).sum(dim=1).mean()
    max_entropy = math.log(num_nodes)
    return float(weight) * (max_entropy - entropy)

