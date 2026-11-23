"""Paired fold-wise comparison of AER-GT against a baseline (Reviewer 2, Comment 2b).

Because each cross-validation fold leaves out exactly one animal, the five
fold-level Macro-F1 values of two methods constitute a naturally paired
comparison. This script reads two fold-metrics CSVs (AER-GT and one
baseline; e.g. the strongest baseline, self-supervised BNT) and reports:

- the per-fold paired differences and the winning method per fold,
- the mean advantage and its range over the wins,
- a one-sided sign test and a one-sided Wilcoxon signed-rank test
  (at n = 5 the minimum attainable one-sided p values are 0.031 for 5/5
  wins and 0.0625 for Wilcoxon, the resolution floor acknowledged in the
  manuscript).

Expected input schema: one CSV per method with columns
``test_subject, macro_f1`` (the ``fold_metrics.csv`` files written by
``train_full_model.py`` / ``run_baselines.py`` runs on identical splits).

Usage:
    python scripts/revision_analyses/paired_baseline_comparison.py \
        --aergt outputs/full_model/fold_metrics.csv \
        --baseline outputs/baselines/self_supervised_bnt/fold_metrics.csv

Outputs:
    outputs/paired_comparison/
        paired_table.csv   per-fold values, difference, winner
        paired_figure.png  per-fold paired dot-line plot
        summary.json       mean advantage, sign/Wilcoxon p values
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aergt", required=True, help="AER-GT fold_metrics.csv")
    parser.add_argument("--baseline", required=True,
                        help="baseline fold_metrics.csv (identical splits)")
    parser.add_argument("--metric", default="macro_f1")
    parser.add_argument("--out", default="outputs/paired_comparison")
    args = parser.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    a = pd.read_csv(args.aergt).set_index("test_subject")[args.metric]
    b = pd.read_csv(args.baseline).set_index("test_subject")[args.metric]
    subjects = sorted(set(a.index) & set(b.index))
    if len(subjects) < 2:
        raise SystemExit("The two CSVs must share at least two test subjects.")

    diff = np.array([a[s] - b[s] for s in subjects])
    wins = diff > 0
    n = len(subjects)

    # one-sided sign test: P(X >= wins | n, 0.5)
    from math import comb
    sign_p = sum(comb(n, k) for k in range(int(wins.sum()), n + 1)) / 2 ** n
    try:
        wil_p = wilcoxon(diff, alternative="greater").pvalue
    except ValueError:
        wil_p = float("nan")

    table = pd.DataFrame({
        "test_subject": subjects,
        "AER-GT": [a[s] for s in subjects],
        "baseline": [b[s] for s in subjects],
        "difference": diff,
        "AER-GT wins": ["Yes" if w else "No" for w in wins],
    })
    table.to_csv(out / "paired_table.csv", index=False)
    print(table.to_string(index=False))

    pos = diff[diff > 0]
    summary = {
        "metric": args.metric,
        "wins": f"{int(wins.sum())}/{n}",
        "mean_advantage": round(float(diff.mean()), 4),
        "advantage_range_wins": ([round(float(pos.min()), 4),
                                  round(float(pos.max()), 4)]
                                 if len(pos) else None),
        "sign_test_one_sided_p": round(float(sign_p), 4),
        "wilcoxon_one_sided_p": (round(float(wil_p), 4)
                                 if wil_p == wil_p else None),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(n)
    for i, s in enumerate(subjects):
        ax.plot([i, i], [b[s], a[s]], color="#c9c9c9", lw=1.2, zorder=1)
    ax.scatter(x, [b[s] for s in subjects], s=70, color="#548235",
               edgecolor="black", lw=0.7, zorder=3, label="baseline")
    ax.scatter(x, [a[s] for s in subjects], s=70, color="#C00000",
               edgecolor="black", lw=0.7, zorder=3, label="AER-GT")
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=20, ha="right")
    ax.set_ylabel(args.metric)
    ax.grid(alpha=0.25, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "paired_figure.png", dpi=300)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
