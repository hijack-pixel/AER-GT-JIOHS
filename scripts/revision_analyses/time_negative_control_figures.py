# -*- coding: utf-8 -*-
"""
Figures for the early/late pseudo-class time negative control
(Reviewer 3, Comment 4).

Renders the summary of the pseudo-class decoding run (see
``time_negative_control_decode.py``): the true three-state AUC versus the
early/late pseudo-class AUCs against the chance line, and the per-fold
pseudo-class AUC.

Input: per-fold AUC CSV with columns ``held_out,auc`` (one row per held-out
fold; the ictal-transition folds and the NB saline folds).

Usage:
    python scripts/revision_analyses/time_negative_control_figures.py \
        --csv outputs/time_negative_control/per_fold_auc.csv

Outputs (written next to the CSV):
    auc_summary.png / .pdf
    per_fold_auc_scatter.png / .pdf
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RED, ORANGE, GRAY = "#C00000", "#E6A817", "#8a8a8a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="per-fold AUC CSV of the decode run")
    parser.add_argument("--true-auc", type=float, default=0.984,
                        help="AUC of the true three-state task (reference bar)")
    args = parser.parse_args()

    out = os.path.dirname(os.path.abspath(args.csv))
    df = pd.read_csv(args.csv)

    ictal = df[df["held_out"].str.startswith("Mouse")]["auc"].values
    nb = df[df["held_out"].str.startswith("NB")]["auc"].values
    m_ictal, sd_ictal = ictal.mean(), ictal.std(ddof=1)
    m_nb, sd_nb = nb.mean(), nb.std(ddof=1)
    print(f"[summary] ictal pseudo-class AUC = {m_ictal:.3f} +/- {sd_ictal:.3f}")
    print(f"[summary] NB pseudo-class AUC     = {m_nb:.3f} +/- {sd_nb:.3f}")

    # ---------- Figure 1: AUC summary ----------
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    labels = ["True three-state task", "Early/late pseudo-class\n(ictal-transition group)",
              "Early/late pseudo-class\n(NB saline group)"]
    vals = [args.true_auc, m_ictal, m_nb]
    errs = [0.008, sd_ictal, sd_nb]
    ax.bar(range(3), vals, yerr=errs, capsize=5, width=0.56,
           color=[RED, ORANGE, GRAY], edgecolor="black", lw=0.8, zorder=3,
           error_kw=dict(lw=1.2))
    for i, (v, e) in enumerate(zip(vals, errs)):
        ax.text(i, v + e + 0.02, f"{v:.3f}", ha="center", fontsize=10,
                fontweight="bold" if i == 0 else "normal")
    ax.axhline(0.5, color="#555555", ls="--", lw=1.2, zorder=2)
    ax.text(2.42, 0.515, "chance (0.500)", fontsize=9, color="#555555", ha="right")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("AUC", fontsize=11)
    ax.set_ylim(0, 1.10)
    ax.grid(alpha=0.25, axis="y")
    ax.tick_params(labelsize=9.5)
    ax.set_title("Time negative control: the identical pipeline cannot decode time",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "auc_summary.png"), dpi=300)
    fig.savefig(os.path.join(out, "auc_summary.pdf"))

    # ---------- Figure 2: per-fold scatter ----------
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    names = list(df["held_out"])
    aucs = list(df["auc"])
    cols = [ORANGE] * len(ictal) + [GRAY] * len(nb)
    ax.scatter(range(len(names)), aucs, s=90, c=cols, edgecolor="black",
               lw=0.8, zorder=3)
    for i, (n, v) in enumerate(zip(names, aucs)):
        ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)
    ax.axhline(0.5, color="#555555", ls="--", lw=1.1, zorder=2)
    ax.text(len(names) - 0.45, 0.506, "chance (0.500)", fontsize=9,
            color="#555555", ha="right", va="bottom")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("Mouse", "M") for n in names], fontsize=10)
    ax.set_ylabel("Pseudo-class AUC", fontsize=11)
    ax.set_xlabel("Held-out fold", fontsize=11)
    ax.set_ylim(0.44, 0.58)
    ax.grid(alpha=0.25, axis="y")
    ax.tick_params(labelsize=9.5)
    ax.set_title("Time negative control: per-fold AUC (all folds at chance)",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "per_fold_auc_scatter.png"), dpi=300)
    fig.savefig(os.path.join(out, "per_fold_auc_scatter.pdf"))

    print("saved ->", out)


if __name__ == "__main__":
    main()
