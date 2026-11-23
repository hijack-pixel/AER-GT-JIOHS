# -*- coding: utf-8 -*-
"""
Segment-position sensitivity analysis (revision).

R3.2: segment-position sensitivity of the Fig. 4 state-characterization stats.

Fig. 4 originally used one fixed 20-s segment per recording (frames 101-300).
Here every "20-s segment" is realized as a block of 5 consecutive windows
(5 x 40-frame stride = 200 frames) from the existing window-level dataset.
Blocks start at window index b = 1..29 (29 possible positions).

  (a) full-recording aggregation (all 33 windows);
  (b) random-position blocks x 1000 draws -> distribution of Friedman p and
      dissociation reproduction rate;
  (c) position sweep b = 1..29 -> p-vs-position curves.

Metrics per (mouse, state): window mean dF/F (NodeFeats[:,0].mean()) and
network deviation 1 - r(A, baseline template), template = mean adjacency of
the animal's baseline windows at the SAME block positions (mirrors the
paper's animal-own-baseline-template design).

Input (read-only): outputs/AERT_Dataset_FiveMice_Rebuilt\\*.mat
Output -> outputs/R3.2_segment\\
"""
import os
import glob
import re
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats as sps

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DS = r"outputs/AERT_Dataset_FiveMice_Rebuilt"
STATE = {0: "Baseline", 1: "Seizure", 2: "Recovery"}
BLK = 5          # windows per 20-s segment
NWIN = 33        # windows per recording
NDRAW = 1000

# ---------- load all windows ----------
rows = []
for f in sorted(glob.glob(os.path.join(DS, "*.mat"))):
    s = sio.loadmat(f)["AERT_Sample"][0, 0]
    name = os.path.basename(f)
    m = re.match(r"Mouse_(\d+)_(.+)_Win(\d+)\.mat", name)
    adj = s["Adjacency"].astype(float)
    rows.append({
        "mouse": int(m.group(1)), "file": m.group(2), "win": int(m.group(3)),
        "label": int(s["Label"][0, 0]),
        "amp": float(np.mean(s["NodeFeats"][:, 0])),
        "adj": adj[np.triu_indices(22, k=1)],   # 231-dim vector
    })
df = pd.DataFrame(rows)
print(f"loaded {len(df)} windows, {df.groupby(['mouse','file']).ngroups} recordings")

iu = np.triu_indices(22, k=1)
def r_vec(a, b):
    return float(np.corrcoef(a, b)[0, 1])

# per-recording arrays, ordered by win
recs = {}
for (mo, fi), g in df.groupby(["mouse", "file"]):
    g = g.sort_values("win")
    recs[(mo, fi)] = {"label": int(g.label.iloc[0]),
                      "amp": g.amp.to_numpy(),
                      "adj": np.vstack(g.adj)}

def block_vals(rec, b):
    sl = slice(b - 1, b - 1 + BLK)
    return rec["amp"][sl].mean(), rec["adj"][sl].mean(axis=0)

def animal_state_table(b=None, full=False):
    """returns per-(mouse,state) amp and deviation for given block start b (or full)"""
    tab = {}
    for mo in sorted({k[0] for k in recs}):
        my = [recs[k] for k in recs if k[0] == mo]
        # baseline template at this position
        bl_recs = [r for r in my if r["label"] == 0]
        if full:
            tmpl = np.mean([r["adj"].mean(axis=0) for r in bl_recs], axis=0)
        else:
            tmpl = np.mean([block_vals(r, b)[1] for r in bl_recs], axis=0)
        for lab in (0, 1, 2):
            rs = [r for r in my if r["label"] == lab]
            if not rs:
                continue
            if full:
                amps = [r["amp"].mean() for r in rs]
                devs = [1 - r_vec(r["adj"].mean(axis=0), tmpl) for r in rs]
            else:
                bv = [block_vals(r, b) for r in rs]
                amps = [a for a, _ in bv]
                devs = [1 - r_vec(d, tmpl) for _, d in bv]
            tab[(mo, STATE[lab])] = (float(np.mean(amps)), float(np.mean(devs)))
    return tab

def friedman(tab, key):
    M = np.array([[tab[(mo, STATE[l])][key] for l in (0, 1, 2)]
                  for mo in sorted({k[0] for k in tab})])
    stat, p = sps.friedmanchisquare(*[M[:, i] for i in range(3)])
    W = stat / (M.shape[0] * 2)
    return stat, p, W

def dissociation(tab):
    _, p_amp, _ = friedman(tab, 0)
    _, p_dev, W_dev = friedman(tab, 1)
    return p_amp, p_dev, W_dev, (p_amp >= 0.05 and p_dev < 0.05)

lines = []
# ---- (a) full aggregation ----
tab_full = animal_state_table(full=True)
pa, pdv, Wd, dis = dissociation(tab_full)
lines.append(f"(a) full-recording aggregation: amplitude Friedman p={pa:.3f}; "
             f"deviation p={pdv:.4f} (W={Wd:.2f}); dissociation={dis}")

# ---- original fixed-segment reference (from the paper) ----
lines.append("(ref) original fixed segment (paper): amplitude p=0.449; deviation p=0.022 (W=0.76)")

# ---- (b) random-position draws ----
rng = np.random.default_rng(42)
res = []
for _ in range(NDRAW):
    tab = animal_state_table(b=int(rng.integers(1, NWIN - BLK + 2)))
    res.append(dissociation(tab))
res = np.array([(r[0], r[1], r[2], float(r[3])) for r in res])
rate = res[:, 3].mean()
lines.append(f"(b) random 20-s segments x{NDRAW}: dissociation reproduced in {rate*100:.1f}% of draws")
lines.append(f"    amplitude p: median={np.median(res[:,0]):.3f}, %>=0.05 = {(res[:,0]>=0.05).mean()*100:.1f}%")
lines.append(f"    deviation p: median={np.median(res[:,1]):.4f}, %<0.05  = {(res[:,1]<0.05).mean()*100:.1f}%")
lines.append(f"    deviation W: median={np.median(res[:,2]):.2f}")

# ---- (c) position sweep ----
sweep = []
for b in range(1, NWIN - BLK + 2):
    tab = animal_state_table(b=b)
    pa, pdv, Wd, dis = dissociation(tab)
    sweep.append((b, pa, pdv, Wd, float(dis)))
sw = pd.DataFrame(sweep, columns=["block_start", "p_amp", "p_dev", "W_dev", "dissoc"])
sw.to_csv(os.path.join(HERE, "r32_position_sweep.csv"), index=False)
lines.append(f"(c) position sweep b=1..29: dissociation at {sw.dissoc.mean()*100:.0f}% of positions; "
             f"deviation p<0.05 at {(sw.p_dev<0.05).mean()*100:.0f}%; amplitude p>=0.05 at {(sw.p_amp>=0.05).mean()*100:.0f}%")

txt = "\n".join(lines)
open(os.path.join(HERE, "r32_summary.txt"), "w", encoding="utf-8").write(txt)
print(txt)

# ---- figure: sweep curves + draw histograms ----
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax = axes[0]
ax.plot(sw.block_start, sw.p_amp, "o-", color="#2d5a1e", label="amplitude (ΔF/F)")
ax.plot(sw.block_start, sw.p_dev, "s-", color="#C00000", label="network deviation")
ax.axhline(0.05, color="grey", ls="--", lw=0.8)
ax.set_xlabel("20-s segment start (window index)", fontsize=10)
ax.set_ylabel("Friedman omnibus p", fontsize=10)
ax.set_yscale("log")
ax.legend(fontsize=9); ax.grid(alpha=0.25)
ax.set_title("Position sweep", fontsize=11)
ax = axes[1]
ax.hist(res[:, 0], bins=40, alpha=0.6, color="#2d5a1e", label="amplitude p")
ax.hist(res[:, 1], bins=40, alpha=0.6, color="#C00000", label="deviation p")
ax.axvline(0.05, color="grey", ls="--", lw=0.8)
ax.set_xlabel("Friedman omnibus p (random segment)", fontsize=10)
ax.set_ylabel("draws", fontsize=10)
ax.legend(fontsize=9); ax.grid(alpha=0.25)
ax.set_title(f"Random segments x{NDRAW}", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "r32_sensitivity.pdf"))
fig.savefig(os.path.join(HERE, "r32_sensitivity.png"), dpi=300)
print("\nsaved ->", HERE)
