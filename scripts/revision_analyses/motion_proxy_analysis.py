# -*- coding: utf-8 -*-
"""
v2_r23_motion.py  (NEW, paper-v2-add)

Experiment 9 (R2.3-alternative + R3.7): locomotion-proxy analyses linking the
per-window frame-difference motion index to decoding outputs.

  (a) Spearman correlation of motion with predicted baseline probability
      (overall and per true state, per animal);
  (b) decomposition of the reversed error-window motion result
      (median 59.9 incorrect vs 76.6 correct);
  (c) motion-quintile-stratified and quiescent-only (lowest-quintile)
      baseline-vs-recovery decoding accuracy;
  (d) motion as a predictor of the predicted label within the B/R subset (AUC).

Inputs (read-only):
  AERT_cv_outputs_split\\oof_predictions.csv                      (1485 rows)
  AERT_motion_quality_five_mice\\per_window_motion_five_mice.csv  (1518 rows; inner join -> 1485)
Output -> E:\\paper\\paper-v2-add\\R2.3_motion\\
"""
import os
import re
import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OOF = r"E:\paper\AERT\AERT_cv_outputs_split\oof_predictions.csv"
MOT = r"E:\paper\AERT\AERT_motion_quality_five_mice\per_window_motion_five_mice.csv"

oof = pd.read_csv(OOF)
mot = pd.read_csv(MOT)

# ---- join keys from oof source_file: "...\\Mouse_9_10min_tif_Win1.mat" ----
pat = re.compile(r"Mouse_(\d+)_(.+)_Win(\d+)\.mat$")
parsed = oof["source_file"].str.extract(pat)
oof["mouse_num"] = parsed[0].astype(int)
oof["win"] = parsed[2].astype(int)

def fkey(s):
    # normalize across file-alias conventions: strip state/batch suffixes, then
    # collapse everything except [a-z0-9]; finally fold "3h-1" style onto "3h"
    t = str(s).lower()
    t = t.replace("癫痫发作", "").replace("seizure", "").replace("transfer", "")
    t = re.sub(r"[^a-z0-9]", "", t)
    return re.sub(r"(h)1$", r"\1", t)

oof["fkey"] = parsed[1].map(fkey)
mot["fkey"] = mot["file"].map(fkey)

# empirical alignment of dataset mouse ids (oof) to raw recording dirs (motion):
# for each oof mouse, pick the motion mouse with the largest overlapping file-key set
oof_sets = {m: set(oof.loc[oof.mouse_num == m, "fkey"]) for m in oof.mouse_num.unique()}
mot_sets = {m: set(mot.loc[mot.mouse == m, "fkey"]) for m in mot.mouse.unique()}
DS2RAW = {}
for m, ks in oof_sets.items():
    best = max(mot_sets, key=lambda r: len(ks & mot_sets[r]))
    DS2RAW[m] = best
oof["raw_mouse"] = oof["mouse_num"].map(DS2RAW)

df = oof.merge(mot, left_on=["raw_mouse", "fkey", "win"],
               right_on=["mouse", "fkey", "win"], how="inner")
align = {f"dataset Mouse_{k} -> raw dir mouse{v}": len(oof_sets[k] & mot_sets[v])
         for k, v in sorted(DS2RAW.items())}
n_join = len(df)
assert df[["motion_diff"]].notna().all().all()
lines = [f"joined windows: {n_join} (oof 1485, motion 1518 -> the 33-window "
         f"extra recording enters the R3.1 accounting)"]

# ---------- (a) motion vs predicted baseline probability ----------
lines.append("")
lines.append("(a) Spearman rho: motion vs predicted baseline probability")
rows_a = []
for scope, sub in [("all", df)] + [(f"true={s}", df[df.true_label_name == s])
                                    for s in ["Baseline", "Seizure", "Recovery"]]:
    rho, p = sps.spearmanr(sub["motion_diff"], sub["p_baseline"])
    rows_a.append((scope, len(sub), rho, p))
    lines.append(f"  {scope:16s} n={len(sub):4d}  rho={rho:+.3f}  p={p:.3g}")
pd.DataFrame(rows_a, columns=["scope", "n", "rho", "p"]).to_csv(
    os.path.join(HERE, "r23_motion_vs_pbaseline.csv"), index=False)

# ---------- (b) decomposition of error-window motion ----------
lines.append("")
lines.append("(b) median motion by true state x correctness (explains 59.9 vs 76.6)")
rows_b = []
for st in ["Baseline", "Seizure", "Recovery"]:
    for corr in [1, 0]:
        sub = df[(df.true_label_name == st) & (df.correct == corr)]
        label = f"{st}{' correct' if corr else ' INCORRECT'}"
        rows_b.append((label, len(sub), sub.motion_diff.median(),
                       sub.motion_diff.quantile(.25), sub.motion_diff.quantile(.75)))
        lines.append(f"  {label:22s} n={len(sub):4d}  median={sub.motion_diff.median():6.1f}  "
                     f"IQR=({sub.motion_diff.quantile(.25):.0f},{sub.motion_diff.quantile(.75):.0f})")
# error composition: where do the low-motion error windows come from?
err = df[df.correct == 0]
comp = err.groupby("true_label_name").motion_diff.agg(["count", "median"])
lines.append("  error composition: " + "; ".join(
    f"{k}: n={int(v['count'])}, median={v['median']:.0f}" for k, v in comp.iterrows()))
pd.DataFrame(rows_b, columns=["group", "n", "median", "q25", "q75"]).to_csv(
    os.path.join(HERE, "r23_error_motion_decomposition.csv"), index=False)

# ---------- (c) motion-stratified and quiescent-only B/R accuracy ----------
br = df[df.true_label_name.isin(["Baseline", "Recovery"])].copy()
br["is_recovery"] = (br.true_label_name == "Recovery").astype(int)
br["br_correct"] = (br.pred_label_name == br.true_label_name).astype(int)
br["motion_quintile"] = pd.qcut(br.motion_diff, 5,
                                labels=["Q1 (quiescent)", "Q2", "Q3", "Q4", "Q5"])
lines.append("")
lines.append("(c) B-vs-R accuracy by motion quintile (quiescent-only = Q1)")
rows_c = []
for q, sub in br.groupby("motion_quintile", observed=True):
    accB = sub[sub.true_label_name == "Baseline"].br_correct.mean()
    accR = sub[sub.true_label_name == "Recovery"].br_correct.mean()
    rows_c.append((str(q), len(sub), sub.motion_diff.median(), accB, accR))
    lines.append(f"  {str(q):16s} n={len(sub):4d}  motion-median={sub.motion_diff.median():5.1f}  "
                 f"acc(B)={accB:.3f}  acc(R)={accR:.3f}")
q1 = br[br.motion_quintile == "Q1 (quiescent)"]
lines.append(f"  quiescent-only pooled B/R accuracy: {q1.br_correct.mean():.3f} (n={len(q1)})")
# motion-matched summary: mean of per-quintile class accuracies
matched_acc = np.mean([r[3] for r in rows_c] + [r[4] for r in rows_c])
lines.append(f"  motion-matched (quintile-averaged) B/R accuracy: {matched_acc:.3f} "
             f"(unmatched pooled: {br.br_correct.mean():.3f})")
pd.DataFrame(rows_c, columns=["quintile", "n", "motion_median", "acc_baseline", "acc_recovery"]).to_csv(
    os.path.join(HERE, "r23_quintile_accuracy.csv"), index=False)

# ---------- (d) can motion alone predict the label within B/R? ----------
auc = roc_auc_score(br["is_recovery"], -br["motion_diff"])   # lower motion -> recovery prediction?
auc_pos = roc_auc_score(br["is_recovery"], br["motion_diff"])
lines.append("")
lines.append("(d) motion alone predicting true B/R label: AUC = "
             f"{max(auc, auc_pos):.3f} (|direction-agnostic|; 0.5 = no information)")

# ---------- figure ----------
fig, ax = plt.subplots(figsize=(6.4, 4.6))
qs = [r[0] for r in rows_c]
ax.plot(range(5), [r[3] for r in rows_c], "o-", color="#548235", label="Baseline windows")
ax.plot(range(5), [r[4] for r in rows_c], "s-", color="#E6A817", label="Recovery windows")
ax.set_xticks(range(5))
ax.set_xticklabels(["Q1\n(quiescent)", "Q2", "Q3", "Q4", "Q5"], fontsize=9)
ax.set_ylabel("Decoding accuracy (B vs R)", fontsize=10.5)
ax.set_xlabel("Motion-index quintile", fontsize=10.5)
ax.set_ylim(0.5, 1.02)
ax.grid(alpha=0.25)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "r23_quintile_accuracy.pdf"))
fig.savefig(os.path.join(HERE, "r23_quintile_accuracy.png"), dpi=300)

txt = "\n".join(lines)
open(os.path.join(HERE, "r23_summary.txt"), "w", encoding="utf-8").write(txt)
print(txt)
print("\nsaved ->", HERE)
