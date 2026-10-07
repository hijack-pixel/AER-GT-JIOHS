from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_state_probability_trajectories(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    eval_dir = run_dir / "evaluation"
    out_dir = run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = eval_dir / "oof_predictions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    prob_cols = [col for col in df.columns if col.startswith("prob_")]
    if not prob_cols:
        raise ValueError(f"No probability columns found in {csv_path}")

    sort_cols = [col for col in ["mouse_id", "original_file", "window_index"] if col in df.columns]
    df = df.sort_values(sort_cols) if sort_cols else df

    fig, ax = plt.subplots(figsize=(8, 4))
    grouped = df.groupby("label", sort=True)
    for label, group in grouped:
        x = np.linspace(0, 1, len(group))
        for col in prob_cols:
            ax.plot(x, group[col].to_numpy(dtype=float), alpha=0.35, label=f"label {label} {col}")
    ax.set_xlabel("Normalized window order")
    ax.set_ylabel("Predicted probability")
    ax.set_title("State probability trajectories")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = out_dir / "state_probability_trajectories.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_roi_attention_summary(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    eval_dir = run_dir / "evaluation"
    out_dir = run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_files = sorted(eval_dir.glob("fold_*_interpretability.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No fold interpretability files found in {eval_dir}")

    attention = []
    for path in npz_files:
        data = np.load(path, allow_pickle=True)
        attention.append(data["attention"])
    attn = np.vstack(attention)
    mean_attn = attn.mean(axis=0)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(np.arange(1, mean_attn.size + 1), mean_attn)
    ax.set_xlabel("ROI index")
    ax.set_ylabel("Mean attention weight")
    ax.set_title("ROI attention summary")
    fig.tight_layout()
    path = out_dir / "roi_attention_summary.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path

