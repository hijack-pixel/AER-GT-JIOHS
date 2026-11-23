# -*- coding: utf-8 -*-
"""
v2_r35a_rank_correlation.py  (NEW, paper-v2-add)

R3.5a: Spearman rank correlation between the recovery-class readout ranking
(Fig. 7a, all 22 ROIs) and the ROI-masking Recovery-F1 drop ranking
(Fig. 7b, all 22 ROIs), with a permutation null (n=1000), plus the Top-8
sub-analysis and a fold-level stability check on the masking side.

Inputs (existing, read-only):
  E:\\paper\\AERT\\AERT_cv_outputs_split\\roi_contribution_data.csv        (readout, 22 ROIs)
  E:\\paper\\AERT\\AERT_cv_outputs_split\\recovery_f1_drop_pp_data.csv     (masking drop, 22 ROIs)
Outputs -> E:\\paper\\paper-v2-add\\R3.5a_output\\
"""
import os
import numpy as np
import pandas as pd
from scipy import stats as sps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "R3.5a_output")
os.makedirs(OUT, exist_ok=True)

CV = r"E:\paper\AERT\AERT_cv_outputs_split"
readout = pd.read_csv(os.path.join(CV, "roi_contribution_data.csv"))[["roi", "mean"]].rename(
    columns={"mean": "readout_mean"})
masking = pd.read_csv(os.path.join(CV, "recovery_f1_drop_pp_data.csv"))[["roi", "mean", "points"]].rename(
    columns={"mean": "mask_drop_mean"})

df = readout.merge(masking, on="roi", how="inner")
assert len(df) == 22, f"expected 22 ROIs, got {len(df)}"
df = df.sort_values("readout_mean", ascending=False).reset_index(drop=True)
df["readout_rank"] = df["readout_mean"].rank(ascending=False).astype(int)
df["mask_rank"] = df["mask_drop_mean"].rank(ascending=False).astype(int)

rng = np.random.default_rng(42)
NPERM = 10000

def spearman_p_perm(x, y, nperm=NPERM, seed_rng=rng):
    rho, _ = sps.spearmanr(x, y)
    null = np.empty(nperm)
    y = np.asarray(y, float)
    for i in range(nperm):
        null[i] = sps.spearmanr(x, rng.permutation(y))[0]
    p_greater = float((null >= rho).mean())
    p_two = float((np.abs(null) >= abs(rho)).mean())
    return rho, p_greater, p_two, null

# ---- main analysis: all 22 ROIs ----
rho22, pg22, pt22, null22 = spearman_p_perm(df["readout_mean"], df["mask_drop_mean"])

# ---- Top-8 sub-analysis (the paper's central claim region) ----
top8 = df.head(8)
rho8, pg8, pt8, _ = spearman_p_perm(top8["readout_mean"], top8["mask_drop_mean"])

# ---- fold-level stability of the masking ranking (per-fold points available) ----
pts = masking.set_index("roi")["points"].str.split(";").apply(
    lambda v: [float(x) for x in v])
fold_mat = np.vstack(pts.loc[df["roi"]].values)          # 22 x 5
fold_rank = np.vstack([sps.rankdata(-fold_mat[:, j]) for j in range(5)]).T
pair_r = [sps.spearmanr(fold_rank[:, a], fold_rank[:, b])[0]
          for a in range(5) for b in range(a + 1, 5)]
stab_obs = float(np.mean(pair_r))
null_stab = []
for _ in range(2000):
    pr = []
    m = rng.permuted(fold_mat, axis=0)
    fr = np.vstack([sps.rankdata(-m[:, j]) for j in range(5)]).T
    pr = [sps.spearmanr(fr[:, a], fr[:, b])[0] for a in range(5) for b in range(a + 1, 5)]
    null_stab.append(np.mean(pr))
p_stab = float((np.array(null_stab) >= stab_obs).mean())

# ---- outputs ----
df_out = df[["roi", "readout_mean", "readout_rank", "mask_drop_mean", "mask_rank"]]
df_out.to_csv(os.path.join(OUT, "r35a_roi_ranking_table.csv"), index=False, encoding="utf-8-sig")

summary = [
    "R3.5a  readout ranking vs masking-drop ranking (all 22 ROIs)",
    f"  Spearman rho = {rho22:.3f}   perm p (greater) = {pg22:.4f}   two-sided = {pt22:.4f}  (n_perm={NPERM})",
    "Top-8 sub-analysis (paper's central region):",
    f"  Spearman rho = {rho8:.3f}   perm p (greater) = {pg8:.4f}   two-sided = {pt8:.4f}",
    "Fold-level stability of masking ranking (mean pairwise Spearman across 5 folds):",
    f"  observed = {stab_obs:.3f}   perm p = {p_stab:.4f}",
    "",
    "Rank table (sorted by readout):",
    df_out.to_string(index=False),
]
txt = "\n".join(summary)
open(os.path.join(OUT, "r35a_summary.txt"), "w", encoding="utf-8").write(txt)
print(txt)

# ---- scatter plot ----
fig, ax = plt.subplots(figsize=(6.2, 5.2))
ax.scatter(df["readout_mean"], df["mask_drop_mean"], s=42, color="#2ca02c", zorder=3,
           label=f"22 ROIs  rho = {rho22:.2f}")
for _, r in df.iterrows():
    if r["readout_rank"] <= 8 or r["mask_rank"] <= 8:
        ax.annotate(r["roi"], (r["readout_mean"], r["mask_drop_mean"]),
                    fontsize=8, xytext=(4, 3), textcoords="offset points")
ax.set_xlabel("Recovery-class readout contribution (Fig. 7a)", fontsize=10.5)
ax.set_ylabel("Recovery-F1 drop when masked (pp, Fig. 7b)", fontsize=10.5)
ax.set_title("Readout vs masking-drop, all 22 ROIs", fontsize=11)
ax.grid(alpha=0.25)
ax.legend(fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "r35a_scatter.pdf"))
fig.savefig(os.path.join(OUT, "r35a_scatter.png"), dpi=300)
print("\nsaved ->", OUT)
