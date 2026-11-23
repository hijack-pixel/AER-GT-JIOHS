# -*- coding: utf-8 -*-
"""
AERT_timeline_states_v3.py   (NEW — replaces v2 for the revision letter)

Timeline of the FIVE ictal-transition animals, drawn from the TRAINING-DATASET
per-recording labels (AERT_Dataset_for_1_4_5_8_9, 45 recordings = 429 B +
462 S + 594 R = 1485 windows), i.e. the exact selection behind every number
in Table 1 and the letter's per-animal table.  Differs from the earlier schedule-based variant, which drew | the full on-disk recording list (46 sessions) rather than the 45 | recordings that entered training.

Per-animal schedules (minutes post-injection; B/S/R = state label of the
whole recording, 33 windows each):
  real 1 (display Mouse 1): B 10,20,30 | S 40,60,90 | R 120,150,180,240   (330 win)
  real 4 (display Mouse 2): B 10,20,29.5 | S 35,45,60 | R 120,240        (264 win)
  real 5 (display Mouse 3): B 10,20,30 | S 40,60,90 | R 120,150,180,240   (330 win)
  real 8 (display Mouse 4): B 10,20,29.5 | S 35,45,60 | R 120,240        (264 win)
  real 9 (display Mouse 5): B 10 | S 20,30 | R 60,90,150,180,210,240     (297 win)

Probe annotations at 30 / 60 min are unchanged from v2 (verified against the
training labels: 30 min -> 4 B + 1 S; 60 min -> 4 S + 1 R).

Output (NEW): outputs/timeline_states.png / .pdf
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

OUT = "outputs/state_timeline"      # adjust to your output directory
os.makedirs(OUT, exist_ok=True)

# --- training-dataset per-recording labels (time in min, state) ---
TL = {
    1: [(10, "B"), (20, "B"), (30, "B"), (40, "S"), (60, "S"), (90, "S"),
        (120, "R"), (150, "R"), (180, "R"), (240, "R")],
    4: [(10, "B"), (20, "B"), (29.5, "B"), (35, "S"), (45, "S"), (60, "S"),
        (120, "R"), (240, "R")],
    5: [(10, "B"), (20, "B"), (30, "B"), (40, "S"), (60, "S"), (90, "S"),
        (120, "R"), (150, "R"), (180, "R"), (240, "R")],
    8: [(10, "B"), (20, "B"), (29.5, "B"), (35, "S"), (45, "S"), (60, "S"),
        (120, "R"), (240, "R")],
    9: [(10, "B"), (20, "S"), (30, "S"), (60, "R"), (90, "R"),
        (150, "R"), (180, "R"), (210, "R"), (240, "R")],
}
MICE = [1, 4, 5, 8, 9]
REMAP = {1: 1, 4: 2, 5: 3, 8: 4, 9: 5}   # display-level renumbering only
WINS = {1: 330, 4: 264, 5: 330, 8: 264, 9: 297}

# ---- consistency asserts against Table 1 ----
nB = sum(1 for m in TL for _, s in TL[m] if s == "B")
nS = sum(1 for m in TL for _, s in TL[m] if s == "S")
nR = sum(1 for m in TL for _, s in TL[m] if s == "R")
assert (nB, nS, nR) == (13, 14, 18), (nB, nS, nR)
assert nB * 33 == 429 and nS * 33 == 462 and nR * 33 == 594
assert sum(len(TL[m]) for m in TL) == 45 and sum(WINS.values()) == 1485
for m in TL:
    assert len(TL[m]) * 33 == WINS[m]
print(f"[check] 45 recordings = 13 B + 14 S + 18 R -> 429/462/594 = 1485  OK")
print(f"[check] per-animal windows 330/264/330/264/297 = 1485  OK")

FILL = {"B": "#b8d8b8", "S": "#f2b8b8", "R": "#f5e6b8"}
EDGE = {"B": "#548235", "S": "#C00000", "R": "#E6A817"}
NAME = {"B": "Baseline", "S": "Seizure", "R": "Recovery"}

BLOCK_W = 5.0
H = 0.52

fig, ax = plt.subplots(figsize=(10.0, 4.1))
fig.subplots_adjust(left=0.115, right=0.90, top=0.94, bottom=0.13)

for r, m in enumerate(MICE):
    y = len(MICE) - 1 - r          # Mouse 1 on top
    recs = TL[m]
    xs = [t for t, _ in recs]
    ax.plot([min(xs), max(xs)], [y, y], color="#c9c9c9", lw=1.1, zorder=1)
    for t, s in recs:
        ax.add_patch(FancyBboxPatch(
            (t - BLOCK_W / 2, y - H / 2), BLOCK_W, H,
            boxstyle="round,pad=0,rounding_size=1.2",
            facecolor=FILL[s], edgecolor=EDGE[s], lw=1.3, zorder=3))
    ax.text(-8, y, f"Mouse {REMAP[m]}", ha="right", va="center", fontsize=10.5)
    ax.text(249, y, f"{WINS[m]} win", ha="left", va="center",
            fontsize=8.2, color="#666666")

for x, lab, yb in [(30, "30 min:\n4 baseline, 1 ictal", 5.02),
                   (60, "60 min:\n4 ictal, 1 recovery", 6.15)]:
    ax.axvline(x, color="#555555", ls="--", lw=1.2, zorder=2)
    ax.text(x, yb, lab, ha="center", va="bottom", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="#555555", lw=0.9))

ax.annotate("Pilocarpine\ninjection", xy=(0, 4.42), xytext=(0, 6.55),
            ha="center", va="bottom", fontsize=9.5,
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.3))

ax.set_xlim(-6, 258)
ax.set_ylim(-0.75, 7.65)
ax.set_yticks([])
ax.set_xticks([0, 30, 60, 90, 120, 150, 180, 210, 240])
ax.set_xlabel("Time after injection (min)", fontsize=10.5)
ax.tick_params(labelsize=9.5)

ax.legend(handles=[Patch(facecolor=FILL[s], edgecolor=EDGE[s], lw=1.2, label=NAME[s])
                   for s in ("B", "S", "R")],
          loc="upper right", fontsize=8.5, framealpha=0.95)

fig.savefig(os.path.join(OUT, "timeline_states.png"), dpi=300)
fig.savefig(os.path.join(OUT, "timeline_states.pdf"))
print("saved ->", OUT)
