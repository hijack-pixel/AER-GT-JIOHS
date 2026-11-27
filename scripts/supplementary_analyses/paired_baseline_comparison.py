"""Paired fold-wise comparison of AER-GT with a baseline.

Because the cross-validation is subject-level, fold k of every method leaves out
the same animal, so the fold-level scores of two methods form paired
observations. For one metric this script reports, per held-out animal, the two
scores and their difference, and tests the differences with

- a one-sided sign test,
- a one-sided Wilcoxon signed-rank test,
- an exact paired sign-flip permutation test (all 2**n sign assignments),
- the 95% confidence interval of the mean paired difference (percentile
  bootstrap over animals, as for the ablation contrasts, and the t-interval).

With n = 5 animals the smallest attainable one-sided p values are 0.031 for the
sign test (five of five differences positive) and 0.0625 for the sign-flip test.

Usage:
    python scripts/supplementary_analyses/paired_baseline_comparison.py \
        --config configs/full_model.yaml --baseline self_supervised_bnt --metric macro_f1

Input (written by ``scripts/run_inference.py`` and ``scripts/run_baselines.py``):
    outputs/<run>/evaluation/fold_metrics.csv
    outputs/<run>/evaluation/baselines/<baseline name>/fold_metrics.csv

Output:
    outputs/supplementary_analyses/paired_comparison/
        paired_table.csv     per-animal scores, difference, which method is higher
        paired_figure.png    paired dot-line plot
        summary.json         mean paired difference with its 95% CI, range over the positive differences, p values
"""
from __future__ import annotations

import argparse
import json
from math import comb

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from _common import ROOT, display_map, evaluation_dir, exact_sign_flip_test, load_run_config, read_fold_metrics  # noqa: E402


def one_sided_sign_test(differences: np.ndarray) -> float:
    """P(X >= number of positive differences | n, 0.5)."""
    n, positives = int(differences.size), int(np.sum(differences > 0))
    return sum(comb(n, k) for k in range(positives, n + 1)) / 2 ** n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/full_model.yaml",
                        help="run configuration of AER-GT; the baselines were run with the same configuration")
    parser.add_argument("--baseline", required=True,
                        help="baseline name as used by scripts/run_baselines.py (e.g. self_supervised_bnt)")
    parser.add_argument("--metric", default="macro_f1")
    parser.add_argument("--out", default="outputs/supplementary_analyses/paired_comparison")
    args = parser.parse_args()

    eval_dir = evaluation_dir(load_run_config(args.config))
    frames = []
    for label, source in (("AER-GT", read_fold_metrics(eval_dir)),
                          (args.baseline, read_fold_metrics(eval_dir / "baselines" / args.baseline))):
        if args.metric not in source.columns:
            raise SystemExit(f"metric {args.metric!r} is not a column of fold_metrics.csv")
        frames.append(source.rename(columns={args.metric: label})[["held_out_subjects", label]])
    merged = frames[0].merge(frames[1], on="held_out_subjects", validate="one_to_one")
    if len(merged) < 2:
        raise SystemExit("The two runs must share at least two held-out animals.")

    names = display_map(merged["held_out_subjects"])
    merged["animal"] = merged["held_out_subjects"].astype(str).map(names)
    merged = merged.sort_values("animal").reset_index(drop=True)
    diff = (merged["AER-GT"] - merged[args.baseline]).to_numpy()
    higher = diff > 0
    merged["difference"] = diff
    merged["AER-GT higher"] = np.where(higher, "Yes", "No")

    sign_p = one_sided_sign_test(diff)
    try:
        wilcoxon_p = float(wilcoxon(diff, alternative="greater").pvalue)
    except ValueError:            # all differences are zero
        wilcoxon_p = float("nan")
    exceed, assignments = exact_sign_flip_test(diff, alternative="greater")
    positive = diff[diff > 0]

    # 95% confidence intervals of the mean paired difference: the percentile bootstrap used
    # for the ablation contrasts (Sec. 2.5) and, for reference, the t-interval over animals
    from aergt.evaluation.ablation_stats import bootstrap_ci_mean  # noqa: E402
    from scipy.stats import t as student_t  # noqa: E402

    ci_low, ci_high = bootstrap_ci_mean(diff)
    half_width = float(student_t.ppf(0.975, len(diff) - 1)) * float(diff.std(ddof=1)) / np.sqrt(len(diff))

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    table = merged[["animal", "AER-GT", args.baseline, "difference", "AER-GT higher"]]
    table.to_csv(out / "paired_table.csv", index=False)
    summary = {
        "metric": args.metric,
        "n_animals": int(len(diff)),
        "aergt_higher": f"{int(higher.sum())}/{int(len(diff))}",
        "mean_difference": round(float(diff.mean()), 4),
        "mean_difference_ci95_bootstrap": [round(ci_low, 4), round(ci_high, 4)],
        "mean_difference_ci95_t": [round(float(diff.mean() - half_width), 4), round(float(diff.mean() + half_width), 4)],
        "range_of_positive_differences": ([round(float(positive.min()), 4), round(float(positive.max()), 4)]
                                          if positive.size else None),
        "sign_test_one_sided_p": round(float(sign_p), 4),
        "wilcoxon_signed_rank_one_sided_p": (round(wilcoxon_p, 4) if wilcoxon_p == wilcoxon_p else None),
        "sign_flip_permutation_one_sided_p": round(exceed / assignments, 4),
        "sign_flip_assignments": assignments,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(max(4.8, 1.4 * len(merged) + 1.6), 4.2))
    x = np.arange(len(merged))
    for i, (_, row) in enumerate(merged.iterrows()):
        ax.plot([i, i], [row[args.baseline], row["AER-GT"]], color="#c9c9c9", lw=1.2, zorder=1)
    ax.scatter(x, merged[args.baseline], s=70, color="#548235", edgecolor="black", lw=0.7, zorder=3, label=args.baseline)
    ax.scatter(x, merged["AER-GT"], s=70, color="#C00000", edgecolor="black", lw=0.7, zorder=3, label="AER-GT")
    ax.set_xticks(x)
    ax.set_xticklabels(merged["animal"], rotation=20, ha="right")
    ax.set_ylabel(args.metric)
    ax.set_ylim(min(merged[args.baseline].min(), merged["AER-GT"].min()) - 0.03,
                max(merged[args.baseline].max(), merged["AER-GT"].max()) + 0.03)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "paired_figure.png", dpi=300)
    fig.savefig(out / "paired_figure.pdf")

    print(table.to_string(index=False))
    print(json.dumps(summary, indent=2))
    print(f"\nsaved -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
