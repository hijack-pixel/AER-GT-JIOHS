# -*- coding: utf-8 -*-
"""
v2_r22a_per_animal.py  (NEW, paper-v2-add)

R2.2a / R3-minor5: per-animal (per held-out fold) performance table for AER-GT,
animal-level 95% CIs, and the decomposition explaining the large Baseline-F1
SD (0.080). Display uses the manuscript's renumbered IDs (1->1, 4->2, 5->3,
8->4, 9->5); internal data IDs unchanged.

Input : E:\\paper\\AERT\\AERT_cv_outputs_split\\fold_metrics.csv
Output-> E:\\paper\\paper-v2-add\\R2.2_per_animal\\
"""
import os
import numpy as np
import pandas as pd
from scipy import stats as sps

HERE = os.path.dirname(os.path.abspath(__file__))
CV = r"E:\paper\AERT\AERT_cv_outputs_split"
REMAP = {"Mouse 1": "Mouse 1", "Mouse 4": "Mouse 2", "Mouse 5": "Mouse 3",
         "Mouse 8": "Mouse 4", "Mouse 9": "Mouse 5"}

df = pd.read_csv(os.path.join(CV, "fold_metrics.csv"))
df["display"] = df["test_subject"].map(REMAP)

metrics = ["accuracy", "macro_auc", "macro_f1", "baseline_f1", "seizure_f1", "recovery_f1"]
hdr = ["n_windows"] + metrics

# ---------- per-animal table ----------
tab = df[["display", "n_windows"] + metrics].copy()
tab = tab.sort_values("display").reset_index(drop=True)
tab.columns = ["Held-out animal", "n windows", "ACC", "Macro-AUC", "Macro-F1",
               "Baseline-F1", "Seizure-F1", "Recovery-F1"]
for c in tab.columns[2:]:
    tab[c] = tab[c].map(lambda v: f"{v:.3f}")

mean_row = ["Mean"] + [f"{df['n_windows'].mean():.0f}"] + \
           [f"{df[m].mean():.3f}" for m in metrics]
sd_row = ["SD"] + ["-"] + [f"{df[m].std(ddof=1):.3f}" for m in metrics]
tab_display = pd.concat([tab, pd.DataFrame([mean_row, sd_row], columns=tab.columns)],
                        ignore_index=True)
tab_display.to_csv(os.path.join(HERE, "r22a_per_animal_table.csv"),
                   index=False, encoding="utf-8-sig")

# ---------- CIs and verification ----------
lines = []
tcrit = sps.t.ppf(0.975, 4)
paper = {"accuracy": (0.914, 0.030), "macro_auc": (0.984, 0.008),
         "macro_f1": (0.904, 0.042), "recovery_f1": (0.914, 0.041),
         "baseline_f1": (None, 0.080), "seizure_f1": (None, None)}
lines.append("Per-animal metrics: mean ± SD and animal-level 95% t-CI (n=5)")
for m in metrics:
    mu, sd = df[m].mean(), df[m].std(ddof=1)
    ci = tcrit * sd / np.sqrt(5)
    ref = paper.get(m, (None, None))
    ok = ""
    if ref[0] is not None:
        ok = " | matches paper" if abs(mu - ref[0]) < 0.0005 and abs(sd - ref[1]) < 0.0005 else \
             f" | CHECK vs paper {ref}"
    elif ref[1] is not None:
        ok = " | SD matches paper" if abs(sd - ref[1]) < 0.0005 else f" | CHECK vs paper SD {ref[1]}"
    lines.append(f"  {m:12s}: {mu:.3f} ± {sd:.3f}   95%CI [{mu-ci:.3f}, {mu+ci:.3f}]"
                 f"   range [{df[m].min():.3f}, {df[m].max():.3f}]{ok}")

# ---------- Baseline-F1 SD driver ----------
lines.append("")
lines.append("Baseline-F1 SD decomposition (why 0.080):")
for _, r in df.sort_values("baseline_f1").iterrows():
    tp, fp_s, fp_r = r["cm_00"], r["cm_10"], r["cm_20"]
    nB = r["cm_00"] + r["cm_01"] + r["cm_02"]
    prec = tp / (tp + fp_s + fp_r)
    rec = tp / nB
    lines.append(f"  {REMAP[r['test_subject']]:9s} (data {r['test_subject']}): "
                 f"Baseline-F1 {r['baseline_f1']:.3f} | precision {prec:.2f} "
                 f"({tp}/{int(tp+fp_s+fp_r)}), recall {rec:.2f} ({int(tp)}/{int(nB)}), "
                 f"recovery->baseline errors {int(fp_r)}")
low = df.loc[df["baseline_f1"].idxmin()]
lines.append("")
lines.append(f"Driver: {REMAP[low['test_subject']]} (data id {low['test_subject']}) with "
             f"Baseline-F1 {low['baseline_f1']:.3f}; excluding it, the remaining four animals "
             f"give SD {df[df.baseline_f1 != low['baseline_f1']].baseline_f1.std(ddof=1):.3f}.")
lines.append(f"Its Macro-AUC remains {low['macro_auc']:.3f}: the deficit is a threshold/precision "
             f"effect on the baseline class (recovery windows pulled into baseline), not a "
             f"ranking failure.")

txt = "\n".join(lines)
open(os.path.join(HERE, "r22a_summary.txt"), "w", encoding="utf-8").write(txt)
print(tab_display.to_string(index=False))
print()
print(txt)
