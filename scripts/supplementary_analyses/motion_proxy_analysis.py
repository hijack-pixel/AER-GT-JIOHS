"""Motion-proxy analysis.

Links the per-window frame-difference motion index to the out-of-fold decoding
outputs of a run:

(a) Spearman correlation between the motion index and the predicted
    baseline-class probability: pooled over all windows (descriptive) and
    within each animal, for all windows and within each true state. Windows
    of a recording overlap, so no window-level p value is reported;
(b) median motion of correctly and incorrectly decoded windows by true state,
    and the state composition of the incorrectly decoded windows;
(c) baseline-versus-recovery decoding accuracy by motion quintile, including
    the lowest-motion quintile, and a motion-stratified balanced accuracy (the
    mean of the two class accuracies within each quintile, averaged over the
    five quintiles with equal weight) alongside the pooled baseline-versus-
    recovery accuracy.

The motion index is a proxy for image-plane displacement; it is not a measure
of locomotion, and the analyses describe association at this sample size.

Usage:
    python scripts/supplementary_analyses/motion_proxy_analysis.py \
        --config configs/full_model.yaml \
        [--motion-csv <per_window_motion.csv>]

Inputs:
    outputs/<run>/motion_quality/per_window_motion.csv   written by scripts/compute_motion_quality.py
    outputs/<run>/evaluation/oof_predictions.csv   written by scripts/run_inference.py

Output:
    outputs/supplementary_analyses/motion_proxy/
        motion_vs_baseline_probability.csv
        error_motion_decomposition.csv
        quintile_accuracy.csv
        quintile_accuracy.png / .pdf
        summary.txt
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from _common import (ROOT, class_names, display_map, evaluation_dir, load_run_config, read_oof_predictions,  # noqa: E402
                     resolve_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/full_model.yaml")
    parser.add_argument("--motion-csv", default="", help="default: outputs/<run>/motion_quality/per_window_motion.csv")
    parser.add_argument("--out", default="outputs/supplementary_analyses/motion_proxy")
    args = parser.parse_args()

    cfg = load_run_config(args.config)
    states = [str(name).lower() for name in class_names(cfg)]
    baseline, recovery = states.index("baseline"), states.index("recovery")
    oof = read_oof_predictions(evaluation_dir(cfg))
    oof["mouse_id"] = oof["mouse_id"].astype(str).str.strip()
    motion_path = resolve_path(args.motion_csv) if args.motion_csv else (evaluation_dir(cfg).parent / "motion_quality" / "per_window_motion.csv")
    motion = pd.read_csv(motion_path)
    required = {"mouse_id", "original_file", "window_index", "motion_diff"}
    if not required.issubset(motion.columns):
        raise SystemExit(f"motion CSV lacks columns {sorted(required - set(motion.columns))}")
    motion["mouse_id"] = motion["mouse_id"].astype(str).str.strip()

    df = oof.merge(motion[["mouse_id", "original_file", "window_index", "motion_diff"]],
                   on=["mouse_id", "original_file", "window_index"], how="inner", validate="one_to_one")
    if df.empty:
        raise SystemExit("no out-of-fold window matched a motion row; check the identifiers of both files")
    if len(df) < len(oof):
        print(f"NOTE: {len(oof) - len(df)} of {len(oof)} out-of-fold windows have no motion value and are excluded")
    df["state"] = [states[index].capitalize() for index in df["label"]]
    df["correct"] = (df["pred"] == df["label"]).astype(int)
    df["p_baseline"] = df["prob_baseline"]

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    lines = [f"windows joined: {len(df)}"]

    # ---------- (a) motion vs predicted baseline probability ----------
    # Windows of a recording overlap, so no window-level p value is reported: the
    # correlation is given pooled over all windows (descriptive) and within each
    # animal, for all windows and within each true state.
    names = display_map(df["mouse_id"])
    df["animal"] = df["mouse_id"].map(names)
    rows = []
    rho = float(spearmanr(df["motion_diff"], df["p_baseline"])[0])
    rows.append({"scope": "all", "animal": "", "n": len(df), "rho": rho})
    lines += ["", "(a) Spearman correlation: motion index vs predicted baseline-class probability",
              f"  pooled over all windows (descriptive): n={len(df)}  rho={rho:+.3f}"]
    for scope, frame in [("all", df)] + [(state, df[df["state"] == state.capitalize()]) for state in states]:
        values = []
        for animal, sub in frame.groupby("animal"):
            if len(sub) > 2 and sub["motion_diff"].nunique() > 1 and sub["p_baseline"].nunique() > 1:
                value = float(spearmanr(sub["motion_diff"], sub["p_baseline"])[0])
                values.append(value)
                rows.append({"scope": scope, "animal": animal, "n": len(sub), "rho": value})
        if values:
            lines.append(f"  within {scope:9s}: rho per animal {min(values):+.3f} to {max(values):+.3f}, "
                         f"median {float(np.median(values)):+.3f} "
                         f"({sum(value < 0 for value in values)} of {len(values)} negative)")
    pd.DataFrame(rows).to_csv(out / "motion_vs_baseline_probability.csv", index=False)

    # ---------- (b) motion of correctly and incorrectly decoded windows ----------
    rows = []
    lines += ["", "(b) Motion by true state and correctness"]
    for state in states:
        for correct in (1, 0):
            sub = df[(df["state"] == state.capitalize()) & (df["correct"] == correct)]
            rows.append({"state": state.capitalize(), "decoded": "correct" if correct else "incorrect", "n": len(sub),
                         "median": float(sub["motion_diff"].median()) if len(sub) else np.nan,
                         "q25": float(sub["motion_diff"].quantile(0.25)) if len(sub) else np.nan,
                         "q75": float(sub["motion_diff"].quantile(0.75)) if len(sub) else np.nan})
            if len(sub):
                lines.append(f"  {state.capitalize():9s} {'correct  ' if correct else 'incorrect'} n={len(sub):4d}"
                             f"  median={sub['motion_diff'].median():6.1f}"
                             f"  IQR=({sub['motion_diff'].quantile(0.25):.0f}, {sub['motion_diff'].quantile(0.75):.0f})")
    pd.DataFrame(rows).to_csv(out / "error_motion_decomposition.csv", index=False)
    errors = df[df["correct"] == 0].groupby("state")["motion_diff"].agg(["count", "median"])
    lines.append("  incorrectly decoded windows by true state: " + "; ".join(
        f"{state}: n={int(row['count'])}, median={row['median']:.0f}" for state, row in errors.iterrows()))

    # ---------- (c) baseline-vs-recovery accuracy by motion quintile ----------
    subset = df[df["label"].isin([baseline, recovery])].copy()
    subset["true_class"] = np.where(subset["label"] == baseline, "Baseline", "Recovery")
    subset["quintile"] = pd.qcut(subset["motion_diff"], 5, labels=[f"Q{index}" for index in range(1, 6)])
    rows = []
    lines += ["", "(c) Baseline-vs-recovery accuracy by motion quintile (Q1 = lowest motion)"]
    for quintile, sub in subset.groupby("quintile", observed=True):
        accuracy = {name: float(sub.loc[sub["true_class"] == name, "correct"].mean()) for name in ("Baseline", "Recovery")}
        rows.append({"quintile": str(quintile), "n": len(sub), "motion_median": float(sub["motion_diff"].median()),
                     "accuracy_baseline": accuracy["Baseline"], "accuracy_recovery": accuracy["Recovery"],
                     "accuracy_pooled": float(sub["correct"].mean())})
        lines.append(f"  {quintile}  n={len(sub):4d}  motion-median={sub['motion_diff'].median():5.1f}"
                     f"  accuracy(B)={accuracy['Baseline']:.3f}  accuracy(R)={accuracy['Recovery']:.3f}"
                     f"  pooled={sub['correct'].mean():.3f}")
    quintiles = pd.DataFrame(rows)
    quintiles.to_csv(out / "quintile_accuracy.csv", index=False)
    stratified = float(np.mean(quintiles[["accuracy_baseline", "accuracy_recovery"]].to_numpy()))
    lines.append(f"  motion-stratified balanced accuracy: {stratified:.3f}; pooled baseline-versus-recovery accuracy: "
                 f"{subset['correct'].mean():.3f}")

    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    x = np.arange(len(quintiles))
    ax.plot(x, quintiles["accuracy_baseline"], "o-", color="#548235", label="Baseline windows")
    ax.plot(x, quintiles["accuracy_recovery"], "s-", color="#E6A817", label="Recovery windows")
    lowest = float(min(quintiles["accuracy_baseline"].min(), quintiles["accuracy_recovery"].min()))
    ax.set_ylim(min(0.5, lowest - 0.05), 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{label}\n(lowest motion)" if label == "Q1" else label for label in quintiles["quintile"]], fontsize=9)
    ax.set_ylabel("Decoding accuracy (baseline vs recovery)")
    ax.set_xlabel("Motion-index quintile")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "quintile_accuracy.png", dpi=300)
    fig.savefig(out / "quintile_accuracy.pdf")

    print("\n".join(lines))
    print(f"\nsaved -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
