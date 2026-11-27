"""Figures for the time-based negative control.

Renders the summaries written by ``time_negative_control_decode.py``: the AUC of
the true three-state task against the early/late pseudo-class AUC of the folds
that contain both pseudo-classes, with the chance level, and the per-fold
pseudo-class AUCs.

Usage:
    python scripts/supplementary_analyses/time_negative_control_figures.py \
        --config configs/full_model.yaml --run outputs/time_negative_control \
        [--nb-run outputs/time_negative_control_nb]

Input:
    <run>/per_fold.csv          pseudo-class decoding of the baseline windows (ictal-transition group)
    <nb-run>/per_fold.csv       optional, pseudo-class decoding of the normative-baseline (NB saline) windows
    outputs/<main run>/evaluation/fold_metrics.csv   Macro-AUC of the three-state task

Output (written into ``<run>``):
    auc_summary.png / .pdf
    per_fold_auc.png / .pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _common import ROOT, evaluation_dir, load_run_config, mean_sd, read_fold_metrics, resolve_path  # noqa: E402

RED, ORANGE, GREY = "#C00000", "#E6A817", "#8a8a8a"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/full_model.yaml", help="configuration of the three-state task")
    parser.add_argument("--run", default="outputs/time_negative_control",
                        help="output directory of the decode script (baseline windows)")
    parser.add_argument("--nb-run", default="", help="output directory of the decode script for the NB cohort")
    args = parser.parse_args()

    run = resolve_path(args.run)
    sources = [("Ictal baseline windows", pd.read_csv(run / "per_fold.csv"), ORANGE)]
    if args.nb_run:
        sources.append(("NB saline windows", pd.read_csv(resolve_path(args.nb_run) / "per_fold.csv"), GREY))
    valid = [(name, frame.dropna(subset=["auc"]), color) for name, frame, color in sources]
    valid = [(name, frame, color) for name, frame, color in valid if len(frame)]
    if not valid:
        raise SystemExit("no fold with both pseudo-classes")

    true_auc = read_fold_metrics(evaluation_dir(load_run_config(args.config)))["macro_auc"]
    true_mean, true_sd = mean_sd(true_auc)

    # ---------- Figure 1: AUC summary ----------
    labels = ["True three-state task"] + [name for name, _, _ in valid]
    values = [true_mean] + [float(frame["auc"].mean()) for _, frame, _ in valid]
    errors = [true_sd] + [float(frame["auc"].std(ddof=1)) if len(frame) > 1 else 0.0 for _, frame, _ in valid]
    colors = [RED] + [color for _, _, color in valid]
    fig, ax = plt.subplots(figsize=(2.2 + 3.2 * len(labels), 4.4))
    ax.bar(np.arange(len(labels)), values, yerr=errors, capsize=5, width=0.52, color=colors, edgecolor="black", lw=0.8,
           zorder=3, error_kw=dict(lw=1.2))
    for index, (value, error) in enumerate(zip(values, errors)):
        ax.text(index, value + error + 0.03, f"{value:.3f} \u00b1 {error:.3f}", ha="center", fontsize=9.5)
    ax.axhline(0.5, color="#555555", ls="--", lw=1.2, zorder=2)
    ax.text(len(labels) - 0.45, 0.515, "chance (0.500)", fontsize=9, color="#555555", ha="right")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("AUC")
    ax.set_ylim(0, 1.10)
    ax.tick_params(labelsize=9.5)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(run / f"auc_summary.{extension}", **({"dpi": 300} if extension == "png" else {}))
    plt.close(fig)

    # ---------- Figure 2: per-fold pseudo-class AUC ----------
    names, aucs, fold_colors = [], [], []
    for name, frame, color in valid:
        for _, row in frame.iterrows():
            names.append(f"{row['held_out']} ({name.split(' ')[0]})")
            aucs.append(float(row["auc"]))
            fold_colors.append(color)
    fig, ax = plt.subplots(figsize=(max(4.8, 1.3 * len(names) + 1.8), 4.2))
    x = np.arange(len(names))
    ax.scatter(x, aucs, s=90, c=fold_colors, edgecolor="black", lw=0.8, zorder=3)
    for index, value in enumerate(aucs):
        ax.annotate(f"{value:.3f}", (index, value), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
    ax.axhline(0.5, color="#555555", ls="--", lw=1.1, zorder=2)
    ax.text(len(names) - 0.55, 0.504, "chance (0.500)", fontsize=9, color="#555555", ha="right", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("Pseudo-class AUC")
    ax.set_xlabel("Held-out fold")
    low = min([0.44] + [value - 0.03 for value in aucs])
    high = max([0.56] + [value + 0.05 for value in aucs])
    ax.set_ylim(low, high)
    ax.tick_params(labelsize=9.5)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(run / f"per_fold_auc.{extension}", **({"dpi": 300} if extension == "png" else {}))
    plt.close(fig)

    for name, frame, _ in valid:
        mean, sd = mean_sd(frame["auc"])
        print(f"{name}: pseudo-class AUC {mean:.3f} +/- {sd:.3f} over {len(frame)} folds")
    print(f"true task Macro-AUC {true_mean:.3f} +/- {true_sd:.3f}")
    print(f"saved -> {run.relative_to(ROOT) if ROOT in run.parents else run}")


if __name__ == "__main__":
    main()
