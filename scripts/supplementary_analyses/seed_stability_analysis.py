"""Seed-stability analysis.

Aggregates the paper training run and its reruns with further random seeds into
three tables and a two-panel figure:

- Table A: mean of each metric over the held-out animals, per seed, and the
  mean +/- SD of these means over seeds;
- Table B: seed x animal detail (each cell is one held-out animal);
- Table C: between-animal SD (over the held-out animals of one run) against
  between-seed SD (over the seed means) for each metric.

The reruns use configurations derived from the main one with a different
``project.seed`` (``configs/seed_123.yaml``, ``configs/seed_2024.yaml``); nothing
else differs.

Usage:
    python scripts/supplementary_analyses/seed_stability_analysis.py \
        --config configs/full_model.yaml \
        --rerun configs/seed_123.yaml --rerun configs/seed_2024.yaml

Input (per run, written by ``scripts/run_inference.py``):
    outputs/<run>/evaluation/fold_metrics.csv

Output:
    outputs/supplementary_analyses/seed_stability/
        tableA_seed_stability.csv
        tableB_seed_by_animal.csv
        tableC_variance_decomposition.csv
        seed_stability_figure.png / .pdf
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _common import METRICS, METRIC_LABELS, ROOT, display_map, evaluation_dir, load_run_config, read_fold_metrics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/full_model.yaml", help="the main run")
    parser.add_argument("--rerun", action="append", default=[], help="rerun configuration (repeatable)")
    parser.add_argument("--out", default="outputs/supplementary_analyses/seed_stability")
    args = parser.parse_args()

    runs = {}
    for config in [args.config, *args.rerun]:
        cfg = load_run_config(config)
        seed = int(cfg["project"]["seed"])
        if seed in runs:
            raise SystemExit(f"seed {seed} is given twice; each run must use its own seed")
        runs[seed] = read_fold_metrics(evaluation_dir(cfg))
    names = display_map(sum((frame["held_out_subjects"].astype(str).tolist() for frame in runs.values()), []))

    # consistency check: Macro-F1 is the mean of the three class-wise F1 values
    for seed, frame in runs.items():
        residual = np.abs(frame["macro_f1"] - frame[["baseline_f1", "seizure_f1", "recovery_f1"]].mean(axis=1)).max()
        if residual > 1e-5:
            raise SystemExit(f"seed {seed}: Macro-F1 differs from the mean of the class-wise F1 values (max {residual:.2e})")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    # ---------- Table A ----------
    rows = [{"Seed": seed, **{label: f"{frame[metric].mean():.4f}" for metric, label in zip(METRICS, METRIC_LABELS)}}
            for seed, frame in runs.items()]
    rows.append({"Seed": "mean +/- SD", **{
        label: f"{np.mean([frame[metric].mean() for frame in runs.values()]):.4f} +/- "
               f"{np.std([frame[metric].mean() for frame in runs.values()], ddof=1):.4f}" if len(runs) > 1 else ""
        for metric, label in zip(METRICS, METRIC_LABELS)}})
    table_a = pd.DataFrame(rows)
    table_a.to_csv(out / "tableA_seed_stability.csv", index=False)

    # ---------- Table B ----------
    rows = []
    for seed, frame in runs.items():
        for _, record in frame.iterrows():
            animal = names[str(record["held_out_subjects"])]
            rows.append({"Animal": animal, "Seed": seed, "Windows (n)": int(record["n_windows"]),
                         **{label: round(float(record[metric]), 4) for metric, label in zip(METRICS, METRIC_LABELS)}})
    table_b = pd.DataFrame(rows).sort_values(["Animal", "Seed"]).reset_index(drop=True)
    table_b.to_csv(out / "tableB_seed_by_animal.csv", index=False)
    ranges = (table_b.groupby("Animal")[["Macro-F1", "Baseline-F1"]]
              .agg(lambda values: float(values.max() - values.min()))
              .rename(columns={"Macro-F1": "Macro-F1 range over seeds",
                               "Baseline-F1": "Baseline-F1 range over seeds"})
              .reset_index())

    # ---------- Table C ----------
    main = runs[min(runs)]
    rows = []
    for metric, label in zip(METRICS, METRIC_LABELS):
        animal_sd = float(main[metric].std(ddof=1))
        seed_means = [float(frame[metric].mean()) for frame in runs.values()]
        seed_sd = float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0
        rows.append({"Metric": label, "Between-animal SD": round(animal_sd, 4), "Between-seed SD": round(seed_sd, 4),
                     "Ratio": round(animal_sd / seed_sd, 1) if seed_sd > 0 else None})
    table_c = pd.DataFrame(rows)
    table_c.to_csv(out / "tableC_variance_decomposition.csv", index=False)

    # ---------- figure ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    x, width = np.arange(len(METRICS)), 0.8 / len(runs)
    colors = ["#1f77b4", "#2ca02c", "#E6A817", "#C00000"]
    for index, (seed, frame) in enumerate(runs.items()):
        axes[0].bar(x - 0.4 + width * (index + 0.5), [frame[metric].mean() for metric in METRICS], width,
                    label=f"seed {seed}", color=colors[index % len(colors)], alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(METRIC_LABELS, fontsize=8, rotation=30, ha="right")
    axes[0].set_ylabel("Score")
    lowest = min(float(frame[metric].mean()) for frame in runs.values() for metric in METRICS)
    axes[0].set_ylim(0.8 if lowest >= 0.8 else max(0.0, lowest - 0.1), 1.01)
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].set_title("(a) Per-seed means", fontsize=10)
    axes[0].grid(alpha=0.25, axis="y")

    x2 = np.arange(len(METRICS))
    animal_sds = [main[metric].std(ddof=1) for metric in METRICS]
    seed_sds = [np.std([frame[metric].mean() for frame in runs.values()], ddof=1) if len(runs) > 1 else 0.0 for metric in METRICS]
    axes[1].bar(x2 - 0.18, animal_sds, 0.35, label="Between-animal SD", color="#548235", alpha=0.85)
    axes[1].bar(x2 + 0.18, seed_sds, 0.35, label="Between-seed SD", color="#C00000", alpha=0.85)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(METRIC_LABELS, fontsize=8, rotation=30, ha="right")
    axes[1].set_ylabel("SD")
    axes[1].legend(fontsize=9, frameon=False)
    axes[1].set_title("(b) Between-animal and between-seed variability", fontsize=10)
    axes[1].grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(out / "seed_stability_figure.png", dpi=250)
    fig.savefig(out / "seed_stability_figure.pdf")

    print("Table A"); print(table_a.to_string(index=False))
    print("\nCross-seed ranges per animal"); print(ranges.to_string(index=False))
    print("\nTable C"); print(table_c.to_string(index=False))
    print(f"\nsaved -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
