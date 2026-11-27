from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.io
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aergt.config import load_config
from aergt.data.samples import read_aert_sample
from aergt.models.aergt import AERGT


def test_demo_config_inherits_full_model() -> None:
    cfg = load_config(ROOT / "configs" / "demo.yaml")

    assert cfg["model"]["num_nodes"] == 22
    assert cfg["model"]["in_channels"] == 6
    assert cfg["training"]["num_folds"] == 3
    assert cfg["data"]["labeled_dir"] == "data/demo/labeled_epilepsy"


def test_read_aert_sample_schema(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample.mat"
    aert_sample = {
        "NodeFeats": np.ones((22, 6), dtype=np.float32),
        "Adjacency": np.eye(22, dtype=np.float32),
        "Label": np.array([[2]], dtype=np.int32),
        "MouseID": "Mouse 1",
        "OriginalFile": "recording_1.tif",
        "WindowIndex": np.array([[1]], dtype=np.int32),
    }
    scipy.io.savemat(sample_path, {"AERT_Sample": aert_sample})

    sample = read_aert_sample(sample_path, labeled=True)

    assert sample is not None
    assert sample.x.shape == (22, 6)
    assert sample.adj.shape == (22, 22)
    assert sample.label == 2
    assert sample.mouse_id == "Mouse 1"


def test_aergt_forward_shapes() -> None:
    model = AERGT(hidden_dim=32, num_heads=4, num_layers=1, transformer_dropout=0.0, attention_dropout=0.0)
    x = torch.randn(2, 22, 6)
    adj = torch.eye(22).repeat(2, 1, 1)

    logits, attention = model(x, adj, return_attention=True)
    recon = model(x, adj, mode="reconstruct")

    assert logits.shape == (2, 3)
    assert attention.shape == (2, 22)
    assert recon.shape == (2, 22, 6)
