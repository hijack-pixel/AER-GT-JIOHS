"""Per-animal decoding performance.

Builds the per-animal performance table of the revised manuscript (Table 4) from a
trained run, adds animal-level 95% confidence intervals, and decomposes the
between-animal variability of Baseline-F1 into the precision and recall of the
baseline class.

Because the cross-validation is subject-level, every fold leaves out exactly one
animal, so the per-fold metrics of a run are per-animal metrics. The confidence
intervals are t-intervals over animals (n - 1 degrees of freedom), as reported in
the revised manuscript.

Usage:
    python scripts/supplementary_analyses/per_animal_performance.py \
        --config configs/full_model.yaml

Input (written by ``scripts/run_inference.py``):
    outputs/<run>/evaluation/subject_metrics_for_stats.csv
    outputs/<run>/evaluation/oof_predictions.csv

Output:
    outputs/supplementary_analyses/per_animal_performance/
        per_animal_table.csv            per-animal metrics, mean +/- SD row
        confidence_intervals.csv        mean, SD, 95% t-interval, range per metric
        baseline_f1_decomposition.csv   baseline precision and recall per animal
        summary.txt
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from _common import (METRICS, METRIC_LABELS, ROOT, class_names, display_map, evaluation_dir,
                     load_run_config, mean_sd, read_oof_predictions, read_subject_metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/full_model.yaml", help="run configuration")
    parser.add_argument("--out", default="outputs/supplementary_analyses/per_animal_performance")
    args = parser.parse_args()

    cfg = load_run_config(args.config)
    eval_dir = evaluation_dir(cfg)
    names = class_names(cfg)
    baseline_idx, recovery_idx = names.index("Baseline"), names.index("Recovery")

    subject = read_subject_metrics(eval_dir)
    oof = read_oof_predictions(eval_dir)
    labels = display_map(subject["mouse_id"])
    subject = subject.assign(animal=subject["mouse_id"].astype(str).map(labels)).sort_values("animal")

    # ---------- per-animal table ----------
    table = subject[["animal", "n_windows", *METRICS]].rename(
        columns={"animal": "Held-out animal", "n_windows": "Windows (n)",
                 **dict(zip(METRICS, METRIC_LABELS))})
    for column in METRIC_LABELS:
        table[column] = table[column].map(lambda value: f"{value:.3f}")
    mean_row, sd_row = ["Mean"], ["SD"]
    mean_row.append(f"{subject['n_windows'].mean():.0f}")
    sd_row.append("-")
    for metric in METRICS:
        mean, sd = mean_sd(subject[metric])
        mean_row.append(f"{mean:.3f}")
        sd_row.append(f"{sd:.3f}")
    table = pd.concat([table, pd.DataFrame([mean_row, sd_row], columns=table.columns)], ignore_index=True)

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "per_animal_table.csv", index=False)

    # ---------- animal-level confidence intervals ----------
    n = len(subject)
    critical = stats.t.ppf(0.975, n - 1)
    intervals = []
    lines = [f"Per-animal metrics over {n} held-out animals: mean +/- SD, 95% t-interval (t = {critical:.3f})", ""]
    for metric, label in zip(METRICS, METRIC_LABELS):
        mean, sd = mean_sd(subject[metric])
        half_width = critical * sd / np.sqrt(n)
        intervals.append({"metric": label, "n": n, "mean": round(mean, 4), "sd": round(sd, 4),
                          "ci95_low": round(mean - half_width, 4), "ci95_high": round(mean + half_width, 4),
                          "min": round(float(subject[metric].min()), 4), "max": round(float(subject[metric].max()), 4)})
        lines.append(f"  {label:12s} {mean:.3f} +/- {sd:.3f}   95% CI [{mean - half_width:.3f}, {mean + half_width:.3f}]"
                     f"   range [{subject[metric].min():.3f}, {subject[metric].max():.3f}]")
    pd.DataFrame(intervals).to_csv(out / "confidence_intervals.csv", index=False)

    # ---------- Baseline-F1 decomposition ----------
    oof["animal"] = oof["mouse_id"].astype(str).map(labels)
    rows = []
    for animal, group in oof.groupby("animal"):
        true, pred = group["label"].to_numpy(), group["pred"].to_numpy()
        true_baseline, pred_baseline = true == baseline_idx, pred == baseline_idx
        tp = int(np.sum(true_baseline & pred_baseline))
        fp = int(np.sum(~true_baseline & pred_baseline))
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / int(true_baseline.sum()) if true_baseline.sum() else float("nan")
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")
        rows.append({"animal": animal, "n_windows": len(group),
                     "baseline_f1": float(subject.loc[subject["animal"] == animal, "baseline_f1"].iloc[0]),
                     "baseline_precision": precision, "baseline_recall": recall,
                     "n_baseline": int(true_baseline.sum()), "n_baseline_correct": tp,
                     "n_baseline_predicted": tp + fp,
                     "recovery_to_baseline_errors": int(np.sum((true == recovery_idx) & pred_baseline))})
    decomposition = pd.DataFrame(rows)
    decomposition["f1_from_precision_recall"] = 2 * decomposition["baseline_precision"] * decomposition["baseline_recall"] / (
        decomposition["baseline_precision"] + decomposition["baseline_recall"])
    decomposition.to_csv(out / "baseline_f1_decomposition.csv", index=False)

    lines += ["", "Baseline-F1 decomposition per animal:", ""]
    for _, row in decomposition.iterrows():
        lines.append(f"  {row['animal']:8s} Baseline-F1 {row['baseline_f1']:.3f} | precision {row['baseline_precision']:.2f}"
                     f" ({row['n_baseline_correct']}/{row['n_baseline_predicted']}),"
                     f" recall {row['baseline_recall']:.2f} ({row['n_baseline_correct']}/{row['n_baseline']}),"
                     f" recovery windows assigned to baseline {row['recovery_to_baseline_errors']}")
    lowest = decomposition.loc[decomposition["baseline_f1"].idxmin()]
    _, sd_without = mean_sd(decomposition.loc[decomposition["animal"] != lowest["animal"], "baseline_f1"])
    _, sd_all = mean_sd(decomposition["baseline_f1"])
    lines += ["",
              f"Lowest Baseline-F1: {lowest['animal']} ({lowest['baseline_f1']:.3f}). "
              f"Baseline-F1 SD over all animals {sd_all:.3f}; without this animal {sd_without:.3f}."]
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(table.to_string(index=False))
    print()
    print(text)
    print(f"\nsaved -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
