from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import scipy.io
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class GraphSample:
    x: np.ndarray
    adj: np.ndarray
    label: int
    mouse_id: str
    source_file: str
    original_file: str = ""
    window_index: int = -1


def normalize_mouse_id(mouse_id: object) -> str:
    value = str(mouse_id).strip()
    return re.sub(r"\s+", " ", value)


def _field(sample, name: str):
    try:
        return sample[name][0, 0]
    except Exception:
        try:
            return getattr(sample[0, 0], name)
        except Exception as exc:
            raise KeyError(name) from exc


def _string_field(sample, name: str, default: str = "") -> str:
    try:
        value = _field(sample, name)
    except KeyError:
        return default
    arr = np.asarray(value).squeeze()
    if arr.size == 0:
        return default
    if arr.dtype.kind in {"U", "S"}:
        return "".join(arr.astype(str).ravel()).strip()
    if arr.shape == ():
        return str(arr.item()).strip()
    return str(arr.ravel()[0]).strip()


def _int_field(sample, name: str, default: int = -1) -> int:
    try:
        value = _field(sample, name)
    except KeyError:
        return default
    arr = np.asarray(value).squeeze()
    if arr.size == 0:
        return default
    return int(arr.item())


def read_aert_sample(
    path: str | Path,
    *,
    labeled: bool,
    num_nodes: int = 22,
    in_channels: int = 6,
    use_abs_adj: bool = True,
) -> GraphSample | None:
    path = Path(path)
    try:
        mat = scipy.io.loadmat(path)
        sample = mat["AERT_Sample"]
        x = np.asarray(_field(sample, "NodeFeats"), dtype=np.float32)
        adj = np.asarray(_field(sample, "Adjacency"), dtype=np.float32)
    except Exception:
        return None

    if x.shape != (num_nodes, in_channels):
        return None
    if adj.shape != (num_nodes, num_nodes):
        return None

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    adj = np.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if use_abs_adj:
        adj = np.abs(adj)

    label = _int_field(sample, "Label", -1) if labeled else -1
    mouse_id = normalize_mouse_id(_string_field(sample, "MouseID", "unknown"))
    original_file = _string_field(sample, "OriginalFile", "")
    window_index = _int_field(sample, "WindowIndex", -1)

    return GraphSample(
        x=x,
        adj=adj,
        label=label,
        mouse_id=mouse_id,
        source_file=str(path),
        original_file=original_file,
        window_index=window_index,
    )


def load_samples(
    folder: str | Path,
    *,
    labeled: bool,
    num_nodes: int,
    in_channels: int,
    use_abs_adj: bool,
) -> list[GraphSample]:
    folder = Path(folder)
    files = sorted(folder.glob("*.mat"))
    if not files:
        raise FileNotFoundError(f"No .mat files found in {folder}")

    samples: list[GraphSample] = []
    skipped = 0
    for path in files:
        sample = read_aert_sample(
            path,
            labeled=labeled,
            num_nodes=num_nodes,
            in_channels=in_channels,
            use_abs_adj=use_abs_adj,
        )
        if sample is None:
            skipped += 1
            continue
        samples.append(sample)

    if not samples:
        raise RuntimeError(f"No valid AERT_Sample records were loaded from {folder}")
    print(f"Loaded {len(samples)} samples from {folder}; skipped {skipped}.")
    return samples


def fit_scaler(
    samples: Sequence[GraphSample],
    *,
    max_samples: int,
    seed: int,
) -> StandardScaler:
    selected = list(samples)
    if max_samples > 0 and len(selected) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(selected), size=max_samples, replace=False)
        selected = [selected[i] for i in idx]
    feats = np.vstack([sample.x for sample in selected])
    scaler = StandardScaler()
    scaler.fit(feats)
    return scaler


def apply_scaler(samples: Sequence[GraphSample], scaler: StandardScaler) -> list[GraphSample]:
    scaled: list[GraphSample] = []
    for sample in samples:
        x = scaler.transform(sample.x)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        scaled.append(
            GraphSample(
                x=x,
                adj=sample.adj,
                label=sample.label,
                mouse_id=sample.mouse_id,
                source_file=sample.source_file,
                original_file=sample.original_file,
                window_index=sample.window_index,
            )
        )
    return scaled


class GraphDataset(Dataset):
    def __init__(self, samples: Sequence[GraphSample]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> GraphSample:
        return self.samples[idx]


def collate_graph_samples(batch: Sequence[GraphSample]) -> dict[str, object]:
    x = torch.tensor(np.stack([sample.x for sample in batch]), dtype=torch.float32)
    adj = torch.tensor(np.stack([sample.adj for sample in batch]), dtype=torch.float32)
    y = torch.tensor([sample.label for sample in batch], dtype=torch.long)
    return {
        "x": x,
        "adj": adj,
        "y": y,
        "mouse_id": [sample.mouse_id for sample in batch],
        "source_file": [sample.source_file for sample in batch],
        "original_file": [sample.original_file for sample in batch],
        "window_index": [sample.window_index for sample in batch],
    }


def make_loader(
    samples: Sequence[GraphSample],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        GraphDataset(samples),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_graph_samples,
    )

