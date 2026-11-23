"""Pharmacological-control decomposition (Reviewer 2, Comment 4).

Decomposes the recovery assignments of the seizure-free pilocarpine control
cohort by animal and post-injection recording, and tests whether the
control windows assigned to recovery carry the network-deviation signature
of true recovery windows (revised Sec. 2.9 / 3.3 and the response letter).

Method (revised manuscript Secs. 2.3 and 2.9):

(a) Time decomposition. For every control window the trained fold models
    assign a class; windows are grouped by (animal, recording) and the
    recovery-assignment rate and mean recovery-class probability are
    computed per recording, together with the per-animal means.

(b) Deviation analysis. The network deviation of a window is
    1 - r between its within-window ROI correlation matrix and the same
    ANIMAL's baseline connectivity template (the mean correlation matrix of
    that animal's baseline-labeled reference recording). Control windows
    are split by their assigned class (control->recovery vs
    control->baseline) and compared against ictal recovery and ictal
    baseline windows with Mann-Whitney U tests.

Expected input schema (example paths; produce with
``scripts/run_control_inference.py`` and the labeled private dataset):

    --control-dir   control .mat AERT_Sample windows (Label = -1),
                    filenames ``<MouseID>_<recording>_Win<k>.mat``
    --predictions   per-window CSV with columns
                    file, pred_class, p_baseline, p_seizure, p_recovery
    --labeled-dir   labeled ictal .mat windows for the deviation reference
    --meta          CSV with columns recording, t_post_inj_min  (per animal)

Outputs:
    outputs/control_decomposition/
        decomposition_table.csv   animal x recording rates/probabilities
        deviation_groups.csv      group medians/IQR/n and p-values
        decomposition_figure.png  two-panel time-decomposition figure
        deviation_boxplot.png     four-group deviation boxplot
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aergt.data.samples import load_samples  # noqa: E402


def upper_tri(mat: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(mat, k=1)
    return mat[iu]


def deviation(adj: np.ndarray, template: np.ndarray) -> float:
    r = np.corrcoef(upper_tri(adj), upper_tri(template))[0, 1]
    return float(1.0 - r)


def baseline_templates(labeled_dir: Path) -> dict:
    """Mean correlation template per animal from its baseline recording."""
    samples = load_samples(labeled_dir, labeled=True, num_nodes=22,
                           in_channels=6, use_abs_adj=False)
    templates: dict[str, np.ndarray] = {}
    per_mouse: dict[str, list] = defaultdict(list)
    for s in samples:
        if s.label == 0:
            per_mouse[s.mouse_id].append(s.adj)
    for mouse, mats in per_mouse.items():
        templates[mouse] = np.mean(mats, axis=0)
    return templates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-dir", required=True,
                        help="control AERT_Sample .mat windows (Label = -1)")
    parser.add_argument("--predictions", required=True,
                        help="per-window prediction CSV from run_control_inference.py")
    parser.add_argument("--labeled-dir", required=True,
                        help="labeled ictal .mat windows (deviation reference)")
    parser.add_argument("--meta", required=True,
                        help="CSV with recording, t_post_inj_min per animal")
    parser.add_argument("--out", default="outputs/control_decomposition")
    args = parser.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    preds = pd.read_csv(args.predictions)
    meta = pd.read_csv(args.meta)
    tmap = {str(r.recording): float(r.t_post_inj_min) for _, r in meta.iterrows()}

    control = load_samples(args.control_dir, labeled=False, num_nodes=22,
                           in_channels=6, use_abs_adj=False)
    pmap = {Path(r.file).stem: r for _, r in preds.iterrows()}

    # ---------- (a) time decomposition ----------
    groups: dict[tuple, dict] = defaultdict(
        lambda: {"n": 0, "n_rec": 0, "p_rec": []})
    for s in control:
        key = Path(s.source_file).stem
        if key not in pmap:
            continue
        row = pmap[key]
        rec = f"{s.mouse_id}_{s.original_file}" if s.original_file else key
        g = groups[(s.mouse_id, rec)]
        g["n"] += 1
        if str(row.pred_class) == "2":
            g["n_rec"] += 1
        g["p_rec"].append(float(row.p_recovery))

    rows = []
    for (mouse, rec), g in sorted(groups.items()):
        t = tmap.get(rec, np.nan)
        rows.append({
            "animal": mouse, "recording": rec, "t_post_inj_min": t,
            "n_windows": g["n"], "n_recovery": g["n_rec"],
            "pct_recovery": g["n_rec"] / g["n"],
            "mean_p_recovery": float(np.mean(g["p_rec"])),
        })
    t7 = pd.DataFrame(rows)
    t7.to_csv(out / "decomposition_table.csv", index=False)
    print(t7.to_string(index=False))

    # ---------- (b) deviation analysis ----------
    templates = baseline_templates(Path(args.labeled_dir))
    labeled = load_samples(args.labeled_dir, labeled=True, num_nodes=22,
                           in_channels=6, use_abs_adj=False)

    def ictal_dev(label: int) -> list:
        return [deviation(s.adj, templates[s.mouse_id])
                for s in labeled if s.label == label and s.mouse_id in templates]

    ctrl_rec, ctrl_base = [], []
    for s in control:
        key = Path(s.source_file).stem
        if key not in pmap or s.mouse_id not in templates:
            continue
        d = deviation(s.adj, templates[s.mouse_id])
        (ctrl_rec if str(pmap[key].pred_class) == "2" else ctrl_base).append(d)

    from scipy.stats import mannwhitneyu
    p_internal = mannwhitneyu(ctrl_rec, ctrl_base).pvalue if ctrl_rec else np.nan
    p_vs_ictal = (mannwhitneyu(ctrl_rec, ictal_dev(2)).pvalue
                  if ctrl_rec else np.nan)

    stats = []
    for name, vals in [("ictal_recovery", ictal_dev(2)),
                       ("ictal_baseline", ictal_dev(0)),
                       ("control_to_recovery", ctrl_rec),
                       ("control_to_baseline", ctrl_base)]:
        v = np.asarray(vals)
        stats.append({"group": name, "n": len(v), "median": np.median(v),
                      "q25": np.percentile(v, 25), "q75": np.percentile(v, 75)})
    t8 = pd.DataFrame(stats)
    t8.to_csv(out / "deviation_groups.csv", index=False)
    print(t8.to_string(index=False))
    print(f"control->recovery vs control->baseline: p = {p_internal:.3f}")
    print(f"control->recovery vs ictal recovery:    p = {p_vs_ictal:.3e}")

    # ---------- figures ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for mouse, grp in t7.groupby("animal"):
        grp = grp.sort_values("t_post_inj_min")
        axes[0].plot(grp.t_post_inj_min, grp.pct_recovery * 100, "o-",
                     label=mouse)
        axes[1].plot(grp.t_post_inj_min, grp.mean_p_recovery, "o-", label=mouse)
    axes[0].set_xlabel("Time after injection (min)")
    axes[0].set_ylabel("Recovery-assigned (%)")
    axes[1].set_xlabel("Time after injection (min)")
    axes[1].set_ylabel("Mean recovery-class probability")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / "decomposition_figure.png", dpi=300)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.boxplot([ictal_dev(2), ictal_dev(0), ctrl_rec, ctrl_base],
               tick_labels=["ictal\nrecovery", "ictal\nbaseline",
                            "control->recovery", "control->baseline"],
               showfliers=False)
    ax.set_ylabel("Network deviation (1 - r)")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out / "deviation_boxplot.png", dpi=300)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
