# -*- coding: utf-8 -*-
"""
r34_figs_en.py — 将用户真实输出的两张 R3.4 图重绘为英文版（去除中文标签与内部文件名引用）
数据来源（真实，不变）：r34_逐折AUC.csv（Mouse1-5 ictal 伪类逐折 + NB1/NB2）+ 论文头条 AUC 0.984
输出（覆盖原文件名，信件脚本无需改动）：r34_时间对照AUC对比.png / r34_逐折AUC散点.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "r34_逐折AUC.csv"))

ictal = df[df["held_out"].str.startswith("Mouse")]["auc"].values
nb = df[df["held_out"].str.startswith("NB")]["auc"].values
m_ictal, sd_ictal = ictal.mean(), ictal.std(ddof=1)
m_nb, sd_nb = nb.mean(), nb.std(ddof=1)
assert abs(m_ictal - 0.506) < 0.0005 and abs(sd_ictal - 0.020) < 0.0005
assert abs(m_nb - 0.505) < 0.0005 and abs(sd_nb - 0.018) < 0.0005
print(f"[check] ictal pseudo-class AUC = {m_ictal:.3f} ± {sd_ictal:.3f}  OK")
print(f"[check] NB pseudo-class AUC     = {m_nb:.3f} ± {sd_nb:.3f}  OK")

RED, ORANGE, GRAY = "#C00000", "#E6A817", "#8a8a8a"

# ---------- 图 1：AUC 对比 ----------
fig, ax = plt.subplots(figsize=(7.4, 4.4))
labels = ["True three-state task", "Early/late pseudo-class\n(ictal-transition group)",
          "Early/late pseudo-class\n(NB saline group)"]
vals = [0.984, m_ictal, m_nb]
errs = [0.008, sd_ictal, sd_nb]
colors = [RED, ORANGE, GRAY]
bars = ax.bar(range(3), vals, yerr=errs, capsize=5, width=0.56,
              color=colors, edgecolor="black", lw=0.8, zorder=3,
              error_kw=dict(lw=1.2))
for i, (v, e) in enumerate(zip(vals, errs)):
    ax.text(i, v + e + 0.02, f"{v:.3f} ± {e:.3f}", ha="center", fontsize=10,
            fontweight="bold" if i == 0 else "normal")
ax.axhline(0.5, color="#555555", ls="--", lw=1.2, zorder=2)
ax.text(2.42, 0.515, "chance (0.500)", fontsize=9, color="#555555", ha="right")
ax.set_xticks(range(3))
ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylabel("AUC", fontsize=11)
ax.set_ylim(0, 1.10)
ax.grid(alpha=0.25, axis="y")
ax.tick_params(labelsize=9.5)
ax.set_title("Time negative control: the identical pipeline cannot decode time\n"
             "pseudo-class AUC \u2248 0.50 vs true three-state AUC = 0.984",
             fontsize=10.5)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "r34_时间对照AUC对比.png"), dpi=300)
fig.savefig(os.path.join(HERE, "r34_时间对照AUC对比.pdf"))

# ---------- 图 2：逐折散点 ----------
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
ax.set_title("Time negative control: per-fold AUC\n"
             "ictal folds mean 0.506; NB folds mean 0.505; all points at chance",
             fontsize=10.5)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "r34_逐折AUC散点.png"), dpi=300)
fig.savefig(os.path.join(HERE, "r34_逐折AUC散点.pdf"))

print("saved ->", HERE)
