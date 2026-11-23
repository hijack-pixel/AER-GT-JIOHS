# -*- coding: utf-8 -*-
"""
Seed-stability analysis (Reviewer 3, Comment 8a).

Aggregates the paper training run (seed 42) and the two seed reruns
(123, 2024) into: Table A (per-seed metrics), Table B (seed x animal
detail with cross-seed ranges), Table C (between-animal vs between-seed
SD decomposition), and a two-panel summary figure.

Inputs (read-only; adjust the three paths to your run directories):
    outputs/full_model/fold_metrics.csv       (seed 42, the paper run)
    outputs/seed_123/fold_metrics.csv
    outputs/seed_2024/fold_metrics.csv

Outputs:
    outputs/seed_stability/
        tableA_seed_stability.csv
        tableB_seed_by_animal.csv
        tableC_variance_decomposition.csv
        seed_stability_figure.png / .pdf
"""
import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "outputs", "seed_stability")
os.makedirs(OUT, exist_ok=True)

SEEDS = {
    "42":   pd.read_csv(os.path.join(ROOT, "outputs/full_model/fold_metrics.csv")),
    "123":  pd.read_csv(os.path.join(ROOT, "outputs/seed_123/fold_metrics.csv")),
    "2024": pd.read_csv(os.path.join(ROOT, "outputs/seed_2024/fold_metrics.csv")),
}
METRICS = ["accuracy", "macro_auc", "macro_f1", "baseline_f1", "seizure_f1", "recovery_f1"]
MLABELS = ["ACC", "Macro-AUC", "Macro-F1", "Baseline-F1", "Seizure-F1", "Recovery-F1"]
# manuscript display numbering (data IDs 1/4/5/8/9 -> manuscript 1-5)
REMAP = {"Mouse 1": "Mouse 1", "Mouse 4": "Mouse 2", "Mouse 5": "Mouse 3",
         "Mouse 8": "Mouse 4", "Mouse 9": "Mouse 5"}

# ===== consistency check: Macro-F1 equals the mean of the three class F1s =====
for sn, df in SEEDS.items():
    for _, r in df.iterrows():
        lhs = r["macro_f1"]
        rhs = (r["baseline_f1"] + r["seizure_f1"] + r["recovery_f1"]) / 3
        assert abs(lhs - rhs) < 1e-5, f"seed{sn} fold{int(r['fold'])}"
print("consistency: 15/15 folds pass")

# ===== Table A: per-seed metrics =====
rows_a = []
for sn, df in SEEDS.items():
    row = {"Seed": sn}
    for m, ml in zip(METRICS, MLABELS):
        row[ml] = f"{df[m].mean():.4f}"
    rows_a.append(row)
summary = {"Seed": "mean +/- SD"}
for m, ml in zip(METRICS, MLABELS):
    vals = [SEEDS[sn][m].mean() for sn in SEEDS]
    summary[ml] = f"{np.mean(vals):.4f} +/- {np.std(vals, ddof=1):.4f}"
rows_a.append(summary)
tA = pd.DataFrame(rows_a)
tA.to_csv(os.path.join(OUT, "tableA_seed_stability.csv"), index=False)
print("\nTable A:")
print(tA.to_string(index=False))

# ===== Table B: seed x animal detail =====
rows_b = []
for subj in ["Mouse 1", "Mouse 4", "Mouse 5", "Mouse 8", "Mouse 9"]:
    for sn in ["42", "123", "2024"]:
        df = SEEDS[sn]
        r = df[df.test_subject == subj].iloc[0]
        row = {"Animal": REMAP[subj], "data_id": subj, "Seed": sn}
        for m, ml in zip(METRICS, MLABELS):
            row[ml] = round(r[m], 4)
        rows_b.append(row)
tB = pd.DataFrame(rows_b)
tB.to_csv(os.path.join(OUT, "tableB_seed_by_animal.csv"), index=False)

ranges = tB.groupby("Animal").agg(
    mF1_min=("Macro-F1", "min"), mF1_max=("Macro-F1", "max"),
    BF1_min=("Baseline-F1", "min"), BF1_max=("Baseline-F1", "max")).reset_index()
ranges["mF1_range"] = ranges.mF1_max - ranges.mF1_min
ranges["BF1_range"] = ranges.BF1_max - ranges.BF1_min
print("\nTable B summary (cross-seed ranges):")
print(ranges[["Animal", "mF1_range", "BF1_range"]].to_string(index=False))

# ===== Table C: between-animal vs between-seed SD =====
rows_c = []
for m, ml in zip(METRICS, MLABELS):
    animal_sd = SEEDS["42"][m].std(ddof=1)
    seed_vals = [SEEDS[sn][m].mean() for sn in SEEDS]
    seed_sd = np.std(seed_vals, ddof=1)
    rows_c.append({"Metric": ml, "animal_SD": round(animal_sd, 4),
                   "seed_SD": round(seed_sd, 4),
                   "ratio": f"{animal_sd/seed_sd:.1f}x"})
tC = pd.DataFrame(rows_c)
tC.to_csv(os.path.join(OUT, "tableC_variance_decomposition.csv"), index=False)
print("\nTable C:")
print(tC.to_string(index=False))

# ===== Figure: two panels =====
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

ax = axes[0]
x = np.arange(len(MLABELS))
w = 0.25
colors = {"42": "#1f77b4", "123": "#2ca02c", "2024": "#E6A817"}
for i, (sn, df) in enumerate(SEEDS.items()):
    vals = [df[m].mean() for m in METRICS]
    ax.bar(x + i * w, vals, w, label=f"seed {sn}", color=colors[sn], alpha=0.8)
ax.set_xticks(x + w)
ax.set_xticklabels(MLABELS, fontsize=8, rotation=30, ha="right")
ax.set_ylabel("Score", fontsize=10)
ax.set_ylim(0.80, 1.01)
ax.legend(fontsize=8)
ax.set_title("(a) Per-seed metrics", fontsize=10)
ax.grid(alpha=0.25, axis="y")

ax = axes[1]
animal_sds = [SEEDS["42"][m].std(ddof=1) for m in METRICS]
seed_sds = [np.std([SEEDS[sn][m].mean() for sn in SEEDS], ddof=1) for m in METRICS]
x2 = np.arange(len(MLABELS))
ax.bar(x2 - 0.18, animal_sds, 0.35, label="Between-animal SD", color="#548235", alpha=0.85)
ax.bar(x2 + 0.18, seed_sds, 0.35, label="Between-seed SD", color="#C00000", alpha=0.85)
ax.set_xticks(x2)
ax.set_xticklabels(MLABELS, fontsize=8, rotation=30, ha="right")
ax.set_ylabel("SD", fontsize=10)
ax.legend(fontsize=9)
ax.set_title("(b) Animal vs seed variability", fontsize=10)
ax.grid(alpha=0.25, axis="y")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "seed_stability_figure.pdf"))
fig.savefig(os.path.join(OUT, "seed_stability_figure.png"), dpi=250)
plt.close()

print(f"\nsaved -> {OUT}")
