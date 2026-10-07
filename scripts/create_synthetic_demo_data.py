from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import scipy.io

ROOT = Path(__file__).resolve().parents[1]


def _save_sample(path: Path, *, x: np.ndarray, adj: np.ndarray, label: int, mouse_id: str, window_index: int) -> None:
    sample = {
        "NodeFeats": x.astype(np.float32),
        "Adjacency": adj.astype(np.float32),
        "Label": np.array([[label]], dtype=np.int32),
        "MouseID": mouse_id,
        "OriginalFile": f"{mouse_id}_synthetic.tif",
        "WindowIndex": np.array([[window_index]], dtype=np.int32),
        "FPS": np.array([[10]], dtype=np.int32),
        "WinSec": np.array([[20]], dtype=np.int32),
        "StepSec": np.array([[4]], dtype=np.int32),
    }
    scipy.io.savemat(path, {"AERT_Sample": sample})


def build_demo(root: Path, *, overwrite: bool) -> None:
    demo_root = root / "data" / "demo"
    pretrain_dir = demo_root / "pretrain_con1_con2"
    labeled_dir = demo_root / "labeled_epilepsy"
    if overwrite and demo_root.exists():
        shutil.rmtree(demo_root)
    pretrain_dir.mkdir(parents=True, exist_ok=True)
    labeled_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    num_nodes = 22
    in_channels = 6
    prototypes = {
        0: np.array([0.0, 0.5, 0.0, 0.0, 0.2, 0.1], dtype=np.float32),
        1: np.array([1.4, 0.9, 0.3, 0.2, 0.6, 0.7], dtype=np.float32),
        2: np.array([0.7, 0.7, 0.1, 0.1, 0.4, 0.3], dtype=np.float32),
    }

    for mouse in range(1, 5):
        for window in range(18):
            x = rng.normal(0.0, 0.4, size=(num_nodes, in_channels)).astype(np.float32)
            adj_noise = rng.uniform(0.0, 0.25, size=(num_nodes, num_nodes)).astype(np.float32)
            adj = (adj_noise + adj_noise.T) / 2.0
            np.fill_diagonal(adj, 1.0)
            _save_sample(
                pretrain_dir / f"CON_{mouse}_Win{window:03d}.mat",
                x=x,
                adj=adj,
                label=-1,
                mouse_id=f"CON_{mouse}",
                window_index=window,
            )

    for mouse in range(1, 4):
        for label in range(3):
            for window in range(12):
                x = rng.normal(0.0, 0.25, size=(num_nodes, in_channels)).astype(np.float32)
                x += prototypes[label]
                adj_noise = rng.uniform(0.0, 0.35, size=(num_nodes, num_nodes)).astype(np.float32)
                adj = (adj_noise + adj_noise.T) / 2.0
                adj += 0.08 * label
                np.fill_diagonal(adj, 1.0)
                _save_sample(
                    labeled_dir / f"Mouse_{mouse}_L{label}_Win{window:03d}.mat",
                    x=x,
                    adj=adj,
                    label=label,
                    mouse_id=f"Mouse {mouse}",
                    window_index=window,
                )

    print(f"Synthetic demo data written to: {demo_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create synthetic non-biological demo .mat files.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build_demo(ROOT, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

