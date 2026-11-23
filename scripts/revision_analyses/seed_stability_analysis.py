# -*- coding: utf-8 -*-
"""
v2_r38_seed_analysis.py  (NEW, paper-v2-add)

R3.8 三种子完整分析（全真实数据）+ 表 A/B/C + 图 + 解读文档。
输入（只读）:
  E:\\paper\\AERT\\AERT_cv_outputs_split\\fold_metrics.csv            (seed 42)
  E:\\paper\\paper-v2-add\\aer_gft_cv_outputs_split_seed123\\fold_metrics.csv
  E:\\paper\\paper-v2-add\\aer_gft_cv_outputs_split_seed2024\\fold_metrics.csv
输出 -> E:\\paper\\paper-v2-add\\R3.8_seed\\
"""
import os
import numpy as np
import pandas as pd
from scipy import stats as sps

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = {
    "42":  pd.read_csv(r"E:\paper\AERT\AERT_cv_outputs_split\fold_metrics.csv"),
    "123": pd.read_csv(r"E:\paper\paper-v2-add\aer_gft_cv_outputs_split_seed123\fold_metrics.csv"),
    "2024": pd.read_csv(r"E:\paper\paper-v2-add\aer_gft_cv_outputs_split_seed2024\fold_metrics.csv"),
}
METRICS = ["accuracy", "macro_auc", "macro_f1", "baseline_f1", "seizure_f1", "recovery_f1"]
MLABELS = ["ACC", "Macro-AUC", "Macro-F1", "Baseline-F1", "Seizure-F1", "Recovery-F1"]
REMAP = {"Mouse 1": "Mouse 1", "Mouse 4": "Mouse 2", "Mouse 5": "Mouse 3",
         "Mouse 8": "Mouse 4", "Mouse 9": "Mouse 5"}

# ===== 恒等式验证 =====
for sn, df in SEEDS.items():
    for _, r in df.iterrows():
        lhs = r["macro_f1"]
        rhs = (r["baseline_f1"] + r["seizure_f1"] + r["recovery_f1"]) / 3
        assert abs(lhs - rhs) < 1e-5, f"seed{sn} fold{int(r['fold'])}"
print("恒等式: 15/15 folds pass ✓")

# ===== 表 A =====
rows_a = []
for sn, df in SEEDS.items():
    row = {"Seed": sn}
    for m, ml in zip(METRICS, MLABELS):
        row[ml] = round(df[m].mean(), 4)
    rows_a.append(row)
# rebuild properly
rows_a = []
for sn, df in SEEDS.items():
    row = {"Seed": sn}
    for m, ml in zip(METRICS, MLABELS):
        row[ml] = f"{df[m].mean():.4f}"
    rows_a.append(row)
summary = {"Seed": "mean ± SD"}
for m, ml in zip(METRICS, MLABELS):
    vals = [SEEDS[sn][m].mean() for sn in SEEDS]
    summary[ml] = f"{np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}"
rows_a.append(summary)
tA = pd.DataFrame(rows_a)
tA.to_csv(os.path.join(HERE, "r38_表A_种子稳定性.csv"), index=False, encoding="utf-8-sig")
print("\n表 A:")
print(tA.to_string(index=False))

# ===== 表 B：种子×动物明细 =====
rows_b = []
for subj in ["Mouse 1", "Mouse 4", "Mouse 5", "Mouse 8", "Mouse 9"]:
    for sn in ["42", "123", "2024"]:
        df = SEEDS[sn]
        r = df[df.test_subject == subj].iloc[0]
        row = {"Animal": REMAP[subj], "data_id": subj, "Seed": sn}
        for m, ml in zip(METRICS, MLABELS):
            row[ml] = round(r[m], 4)
        row["R_to_B_err"] = int(r["cm_20"])
        rows_b.append(row)
tB = pd.DataFrame(rows_b)
tB.to_csv(os.path.join(HERE, "r38_表B_种子×动物明细.csv"), index=False, encoding="utf-8-sig")

# per-animal cross-seed range
ranges = tB.groupby("Animal").agg(
    mF1_min=("Macro-F1", "min"), mF1_max=("Macro-F1", "max"),
    BF1_min=("Baseline-F1", "min"), BF1_max=("Baseline-F1", "max")).reset_index()
ranges["mF1_range"] = ranges.mF1_max - ranges.mF1_min
ranges["BF1_range"] = ranges.BF1_max - ranges.BF1_min
print("\n表 B 摘要（跨种子极差）:")
print(ranges[["Animal", "mF1_range", "BF1_range"]].to_string(index=False))

# ===== 表 C：方差分解 =====
rows_c = []
for m, ml in zip(METRICS, MLABELS):
    animal_sd = SEEDS["42"][m].std(ddof=1)
    seed_vals = [SEEDS[sn][m].mean() for sn in SEEDS]
    seed_sd = np.std(seed_vals, ddof=1)
    rows_c.append({"Metric": ml, "animal_SD": round(animal_sd, 4),
                   "seed_SD": round(seed_sd, 4),
                   "ratio": f"{animal_sd/seed_sd:.1f}x"})
tC = pd.DataFrame(rows_c)
tC.to_csv(os.path.join(HERE, "r38_表C_方差分解.csv"), index=False, encoding="utf-8-sig")
print("\n表 C:")
print(tC.to_string(index=False))

# ===== 图：双面板 =====
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

# panel a: per-seed bar
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

# panel b: animal vs seed SD
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
fig.savefig(os.path.join(HERE, "r38_三种子图.pdf"))
fig.savefig(os.path.join(HERE, "r38_三种子图.png"), dpi=250)
plt.close()

print("\n图已保存: r38_三种子图.png/pdf")
print("全部产物:", sorted(f for f in os.listdir(HERE) if f.startswith("r38")))
