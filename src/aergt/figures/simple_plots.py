"""Figures of the manuscript that are drawn from the outputs of a run.

- Fig. 5: mean predicted-probability trajectories of the held-out recordings
  after stage-wise alignment.
- Fig. 6 (b-d): decoding of the pharmacological-control cohort.
- Fig. 7: readout ranking, ROI-masking sensitivity, readout contrasts of the
  Top-8 ROIs, and the readout-weighted interaction contrast map.
- Fig. 8: recording-quality and motion metrics.

Every function takes the output directory of a run and writes the figure and
the numbers behind it under ``<run>/figures/``.
"""
from __future__ import annotations

import csv
import re
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

STATE_NAMES = ["Baseline", "Seizure", "Recovery"]
STATE_COLORS = {"Baseline": "#548235", "Seizure": "#C00000", "Recovery": "#E6A817"}
STATE_FILLS = {"Baseline": "#dcebd2", "Seizure": "#f6d6d6", "Recovery": "#f9ecc8"}


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _state_name(label: int) -> str:
    return STATE_NAMES[label] if 0 <= label < len(STATE_NAMES) else f"State {label}"


def _save(fig, out_dir: Path, stem: str) -> Path:
    path = out_dir / f"{stem}.png"
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Fig. 5
# --------------------------------------------------------------------------- #

def plot_state_probability_trajectories(run_dir: str | Path, *, points: int = 100) -> Path:
    """Fig. 5: stage-wise aligned probability trajectories of the held-out recordings.

    Stages follow the session-level reference labels, which are defined relative
    to EEG-confirmed seizure onset and termination (Sec. 2.1): the baseline stage
    is the set of baseline-labelled recordings, the seizure stage the
    seizure-labelled recordings, and the recovery stage the recovery-labelled
    recordings of the held-out animals. The window-level probability trajectory
    of each recording is interpolated to ``points`` positions across the
    recording (normalized to stage duration). Lines are the means over the
    recordings of a stage and bands the standard error of the mean. Panel (a)
    shows the recovery stage, panel (b) the three aligned stages in sequence.
    """
    run_dir = Path(run_dir)
    eval_dir = run_dir / "evaluation"
    out_dir = run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = eval_dir / "oof_predictions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    prob_cols = [col for col in df.columns if col.startswith("prob_")]
    if not prob_cols:
        raise ValueError(f"No probability columns found in {csv_path}")
    classes = [col[len("prob_"):] for col in prob_cols]

    grid = np.linspace(0.0, 1.0, points)
    trajectories: dict[int, list[np.ndarray]] = {label: [] for label in range(len(classes))}
    for (_mouse, _recording), group in df.groupby(["mouse_id", "original_file"]):
        group = group.sort_values("window_index")
        if group["label"].nunique() != 1 or len(group) < 2:
            continue
        x = np.linspace(0.0, 1.0, len(group))
        trajectories[int(group["label"].iloc[0])].append(
            np.vstack([np.interp(grid, x, group[col].to_numpy(dtype=float)) for col in prob_cols]).T)

    stage_data = {}
    for stage, runs in trajectories.items():
        if not runs:
            continue
        stack = np.stack(runs)                                     # recordings x points x classes
        stage_data[stage] = (stack.mean(axis=0), stack.std(axis=0, ddof=1) / np.sqrt(stack.shape[0]), stack.shape[0])

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=True)
    table_rows = []

    def draw(ax, stage: int, offset: float) -> None:
        mean, sem, n = stage_data[stage]
        for k, name in enumerate(classes):
            color = STATE_COLORS.get(name.capitalize())
            ax.plot(grid + offset, mean[:, k], color=color, lw=1.6, label=name.capitalize())
            ax.fill_between(grid + offset, mean[:, k] - sem[:, k], mean[:, k] + sem[:, k], color=color, alpha=0.2, lw=0)
            table_rows.extend([[_state_name(stage), round(float(value + offset), 6), name,
                                round(float(mean[index, k]), 6), round(float(sem[index, k]), 6)]
                               for index, value in enumerate(grid)])

    recovery = len(classes) - 1
    if recovery in stage_data:
        draw(ax_a, recovery, 0.0)
        ax_a.set_title(f"(a) Recovery period ({stage_data[recovery][2]} recordings)", fontsize=10)
    for stage, (mean, _sem, n) in stage_data.items():
        ax_b.axvspan(stage, stage + 1, color=STATE_FILLS.get(_state_name(stage), "#eeeeee"), lw=0, zorder=0)
        ax_b.text(stage + 0.5, 1.03, _state_name(stage), ha="center", va="bottom", fontsize=9)
        draw(ax_b, stage, float(stage))
    for boundary in sorted(stage_data)[:-1]:
        ax_b.axvline(boundary + 1, color="#555555", ls="--", lw=1.0)
    ax_b.set_title(f"(b) Aligned stages ({sum(value[2] for value in stage_data.values())} recordings)", fontsize=10)

    ax_a.set_xlabel("Normalized recovery-stage duration")
    ax_b.set_xlabel("Aligned stage (normalized duration within each stage)")
    ax_b.set_xlim(0.0, float(max(stage_data) + 1) if stage_data else 1.0)
    ax_a.set_xlim(0.0, 1.0)
    ax_a.set_ylim(0.0, 1.0)
    ax_a.set_ylabel("Predicted probability")
    for ax in (ax_a, ax_b):
        ax.grid(alpha=0.25)
    ax_a.legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    _write_csv(out_dir / "state_probability_trajectories.csv", ["stage", "position", "class", "mean", "sem"], table_rows)
    return _save(fig, out_dir, "state_probability_trajectories")


# --------------------------------------------------------------------------- #
# Fig. 6 (b-d)
# --------------------------------------------------------------------------- #

def plot_control_evaluation(run_dir: str | Path, *, recording: str = "") -> Path:
    """Fig. 6 (b-d): decoding of the pharmacological-control cohort.

    (b) ensemble class-probability trajectory of one control recording (mean
    softmax over the fold models); (c) class-assignment proportions per animal
    and overall, with the seizure-class share marked; (d) distribution of the
    ensemble class probabilities over all control windows with the mean
    probability and the assignment rate of each class. Panel (a) of the
    manuscript (a raw IOS trace) is drawn from the raw recording and is not
    produced here.

    ``recording`` selects the recording of panel (b) as ``"<MouseID>|<OriginalFile>"``;
    by default the first recording (in order of the identifiers) is used.
    """
    run_dir = Path(run_dir)
    control_dir = run_dir / "control_inference"
    out_dir = run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path, summary_path = control_dir / "control_window_predictions.csv", control_dir / "control_decoding_summary.json"
    for path in (predictions_path, summary_path):
        if not path.exists():
            raise FileNotFoundError(f"{path}\nRun scripts/run_control_inference.py first.")
    windows = pd.read_csv(predictions_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    classes = [column[len("p_"):] for column in windows.columns if column.startswith("p_")]

    def order_of(identifiers) -> list[str]:
        return sorted(set(identifiers), key=lambda value: [int(token) if token.isdigit() else token for token in
                                                           re.split(r"(\d+)", str(value))])

    animals = order_of(windows["mouse_id"])
    if recording:
        mouse_id, original_file = recording.split("|", 1)
    else:
        first_animal = windows[windows["mouse_id"] == animals[0]].sort_values(["original_file", "window_index"])
        mouse_id, original_file = str(first_animal["mouse_id"].iloc[0]), str(first_animal["original_file"].iloc[0])
    selected = windows[(windows["mouse_id"] == mouse_id) & (windows["original_file"] == original_file)].sort_values("window_index")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    ax_b, ax_c, ax_d = axes

    # (b) class-probability trajectory of one recording
    x = (selected["window_index"].to_numpy(dtype=float) - 1) * 4.0          # window onset in seconds (4-s stride)
    for name in classes:
        ax_b.plot(x, selected[f"p_{name}"], color=STATE_COLORS.get(name.capitalize()), lw=1.6, marker="o", ms=2.8,
                  label=name.capitalize())
    ax_b.set_xlabel("Time within the recording (s)")
    ax_b.set_ylabel("Ensemble class probability")
    ax_b.set_title(f"(b) {mouse_id}, {original_file}", fontsize=10)
    ax_b.set_ylim(0.0, 1.02)
    ax_b.legend(fontsize=8, frameon=False)

    # (c) class-assignment proportions per animal and overall
    overall = summary["overall"]
    labels, shares = animals + ["All"], []
    for animal in animals:
        shares.append([summary["per_animal"][animal][f"{name}_proportion"] for name in classes])
    shares.append([overall[f"{name}_proportion"] for name in classes])
    shares = np.asarray(shares)
    bottom = np.zeros(len(labels))
    for k, name in enumerate(classes):
        ax_c.bar(np.arange(len(labels)), shares[:, k], bottom=bottom, color=STATE_COLORS.get(name.capitalize()),
                 alpha=0.85, label=name.capitalize())
        bottom += shares[:, k]
    if "seizure" in classes:
        k = classes.index("seizure")
        for index in range(len(labels)):
            ax_c.text(index, 1.012, f"S {shares[index, k]:.1%}", ha="center", va="bottom", fontsize=7.5)
    ax_c.set_xticks(np.arange(len(labels)))
    ax_c.set_xticklabels(labels, rotation=20, ha="right")
    ax_c.set_ylim(0.0, 1.08)
    ax_c.set_ylabel("Share of control windows")
    ax_c.set_title("(c) Class-assignment proportions", fontsize=10)
    ax_c.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.94))

    # (d) ensemble class-probability distributions
    data = [windows[f"p_{name}"].to_numpy(dtype=float) for name in classes]
    parts = ax_d.violinplot(data, positions=np.arange(len(classes)), widths=0.8, showextrema=False)
    for body, name in zip(parts["bodies"], classes):
        body.set_facecolor(STATE_COLORS.get(name.capitalize()))
        body.set_alpha(0.6)
    for index, (name, values) in enumerate(zip(classes, data)):
        rate = overall[f"{name}_proportion"]
        ax_d.text(index, 0.5, f"mean {values.mean():.3f}\nassigned {rate:.1%}", ha="center", va="center", fontsize=8)
    ax_d.set_xticks(np.arange(len(classes)))
    ax_d.set_xticklabels([name.capitalize() for name in classes])
    ax_d.set_ylim(0.0, 1.0)
    ax_d.set_ylabel("Ensemble class probability")
    ax_d.set_title("(d) Class probabilities over all control windows", fontsize=10)
    for ax in axes:
        ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    _write_csv(out_dir / "control_assignment_proportions.csv", ["animal", *classes],
               [[label, *[round(float(value), 6) for value in row]] for label, row in zip(labels, shares)])
    return _save(fig, out_dir, "control_evaluation")


# --------------------------------------------------------------------------- #
# Fig. 7
# --------------------------------------------------------------------------- #

def _roi_table(roi_labels: Path | None) -> pd.DataFrame:
    default = Path(__file__).resolve().parent / "roi_table.csv"
    return pd.read_csv(roi_labels if roi_labels and Path(roi_labels).exists() else default)


def plot_roi_attention_summary(run_dir: str | Path, *, top_k: int = 8, roi_labels: Path | None = None) -> Path:
    """Fig. 7 (a-d) from the interpretability outputs of ``scripts/run_interpretability.py``.

    (a) recovery-class readout weight of every ROI (fold-normalized): lollipops to the
        cross-fold mean with SEM whiskers, gray dots for the fold-wise values, blue
        rings for the Top-k ROIs; (b) Recovery-F1 drop when an ROI is masked: bars for
        the cross-fold mean, SEM error bars, dots for the fold-wise values; (c) readout
        contrasts of the Top-k ROIs (left recovery minus baseline, right recovery minus
        seizure) in the same lollipop style; (d) recovery-minus-baseline interaction
        contrast with the top-10% readout-weighted edges (edge color: signed contrast,
        edge width: score; node size: cross-fold mean readout weight).
    ROI names and node positions come from ``roi_table.csv`` (index, label, x, y).
    """
    run_dir = Path(run_dir)
    interp = run_dir / "interpretability"
    out_dir = run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = interp / "interpretability_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path}\nRun scripts/run_interpretability.py first.")
    result = json.loads(summary_path.read_text(encoding="utf-8"))
    per_fold = result["per_fold"]

    readout = np.asarray(result["mean_recovery_readout"], dtype=float)
    readout_folds = np.asarray([fold for fold in per_fold["readout_by_state"]["recovery"] if fold is not None], dtype=float)
    masking = np.asarray(result["mask_recovery_f1_drop_pp_mean"], dtype=float)
    masking_folds = np.asarray(per_fold["mask_recovery_f1_drop_pp"], dtype=float)
    n_rois = readout.size
    top = np.asarray(result["top_roi_indices"], dtype=int)
    contrast_fold = per_fold["readout_contrasts"]
    interaction = np.asarray(result["interaction_contrast"], dtype=float)
    table = _roi_table(roi_labels).iloc[:n_rois].reset_index(drop=True)
    names = [str(value) for value in table["label"]]
    xy = table[["x", "y"]].to_numpy(dtype=float)

    def mean_sem(folds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return folds.mean(axis=0), folds.std(axis=0, ddof=1) / np.sqrt(folds.shape[0])

    fig = plt.figure(figsize=(13.4, 10.4))
    grid = fig.add_gridspec(2, 2)
    ax_a, ax_b = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    grid_c = grid[1, 0].subgridspec(1, len(contrast_fold), wspace=0.40)
    ax_c_all = []
    for cell in grid_c:
        ax_c_all.append(fig.add_subplot(cell, sharey=ax_c_all[0] if ax_c_all else None))
    ax_d = fig.add_subplot(grid[1, 1])

    def lollipop(ax, x, folds, cmap_limit, ring=None):
        """Stems to the cross-fold mean, fold-wise gray dots, SEM whiskers, colored endpoint, black mean marker."""
        mean, sem = mean_sem(folds)
        ax.axhline(0.0, color="black", lw=0.6)
        ax.vlines(x, 0.0, mean, color="#9a9a9a", lw=1.0, zorder=1)
        for position, column in zip(x, folds.T):
            ax.scatter(np.full(column.size, position), column, s=13, color="#c4c4c4", edgecolor="#8f8f8f", lw=0.3, zorder=2)
        ax.errorbar(x, mean, yerr=sem, fmt="none", ecolor="black", capsize=2, lw=0.8, zorder=3)
        ax.scatter(x, mean, s=52, c=mean, cmap="coolwarm", vmin=-cmap_limit, vmax=cmap_limit,
                   edgecolor="#555555", lw=0.5, zorder=4)
        ax.scatter(x, mean, s=9, color="black", zorder=5)
        if ring is not None and len(ring):
            ax.scatter(x[ring], mean[ring], s=120, facecolors="none", edgecolors="#1f4fd8", lw=1.6, zorder=6)
        return mean, sem

    # (a) readout contribution of every ROI
    x = np.arange(n_rois)
    limit = float(np.abs(readout_folds).max()) or 1.0
    lollipop(ax_a, x, readout_folds, limit, ring=top)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(names, rotation=90, fontsize=7)
    ax_a.set_ylabel("Recovery-class readout weight\n(fold-normalized)")
    ax_a.set_title(f"(a) Recovery-class ROI readout contribution (blue rings: Top-{top_k})", fontsize=9.5)

    # (b) masking sensitivity, same ROI order
    masking_mean, masking_sem = mean_sem(masking_folds)
    ax_b.bar(x, masking_mean, color="#C00000", alpha=0.75, width=0.7, zorder=2)
    ax_b.errorbar(x, masking_mean, yerr=masking_sem, fmt="none", ecolor="black", capsize=2, lw=0.8, zorder=3)
    for position, column in zip(x, masking_folds.T):
        ax_b.scatter(np.full(column.size, position), column, s=13, color="#555555", zorder=4)
    ax_b.axhline(0.0, color="black", lw=0.6)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(names, rotation=90, fontsize=7)
    ax_b.set_ylabel("Recovery-F1 drop when masked (pp)")
    ax_b.set_title("(b) ROI masking sensitivity (mean +/- SEM, fold-wise dots)", fontsize=9.5)

    # (c) readout contrasts of the Top-k ROIs: left recovery minus baseline, right recovery minus seizure
    contrast_names = list(contrast_fold)
    contrast_folds = {name: np.asarray([fold for fold in contrast_fold[name] if fold is not None], dtype=float)   # folds x ROIs
                      for name in contrast_names}
    contrast_limit = max([float(np.abs(folds[:, top]).max()) for folds in contrast_folds.values() if folds.size] or [1.0])
    x_top = np.arange(len(top))
    for offset, (ax_c, name) in enumerate(zip(ax_c_all, contrast_names)):
        folds = contrast_folds[name]
        if folds.size == 0:
            ax_c.set_visible(False)
            continue
        lollipop(ax_c, x_top, folds[:, top], contrast_limit)
        ax_c.set_xticks(x_top)
        ax_c.set_xticklabels([names[i] for i in top], rotation=60, ha="right", fontsize=7)
        ax_c.set_ylabel("Readout contrast (fold-normalized)")
        ax_c.set_title(f"(c {'left' if offset == 0 else 'right'}) {name.replace('_minus_', ' - ').capitalize()}, Top-{top_k} ROIs", fontsize=9.5)
    for ax in (ax_a, ax_b, *ax_c_all):
        ax.grid(alpha=0.25, axis="y")

    # (d) interaction contrast map with the top-10% readout-weighted edges
    score = np.zeros_like(interaction)
    for i in range(n_rois):
        for j in range(i + 1, n_rois):
            score[i, j] = score[j, i] = abs(interaction[i, j] * 0.5 * (readout[i] + readout[j]))
    upper = np.triu_indices(n_rois, k=1)
    keep = max(1, int(round(0.10 * len(upper[0]))))
    best = np.argsort(score[upper])[::-1][:keep]
    edge_limit = max(float(np.abs(interaction[upper][best]).max()), 1e-12)
    norm = matplotlib.colors.Normalize(vmin=-edge_limit, vmax=edge_limit)
    cmap = plt.get_cmap("coolwarm")
    width_max = max(float(score[upper][best].max()), 1e-12)
    for k in best:
        i, j = int(upper[0][k]), int(upper[1][k])
        ax_d.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], color=cmap(norm(interaction[i, j])),
                  lw=0.8 + 3.2 * score[i, j] / width_max, alpha=0.9, zorder=1)
    # node size: cross-fold mean normalized readout weight
    span = float(readout.max() - readout.min()) or 1.0
    node_size = 70.0 + 330.0 * (readout - readout.min()) / span
    ax_d.scatter(xy[:, 0], xy[:, 1], s=node_size, c="#e8e8e8", edgecolor="black", lw=0.7, zorder=2)
    for index in range(n_rois):
        ax_d.text(xy[index, 0], xy[index, 1], names[index], fontsize=5.8, ha="center", va="center", zorder=3)
    ax_d.set_title("(d) Recovery - baseline interaction contrast, top 10% readout-weighted edges\n"
                   "(node size: cross-fold mean readout weight)", fontsize=9.5)
    ax_d.set_xticks([])
    ax_d.set_yticks([])
    ax_d.set_aspect("equal")
    pad = 0.08 * float(max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])))
    ax_d.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ax_d.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax_d, fraction=0.05, pad=0.02, label="Interaction contrast")

    fig.tight_layout()
    return _save(fig, out_dir, "roi_attention_summary")


# --------------------------------------------------------------------------- #
# Fig. 8
# --------------------------------------------------------------------------- #

def plot_motion_quality(run_dir: str | Path, *, recording_times: Path | None = None) -> Path:
    """Fig. 8: recording-quality and motion metrics.

    (a) window-level frame-difference motion index by state (boxes: quartiles,
    dots: per-animal medians); (b) per-animal median motion across states;
    (c) mean motion per recording against post-injection time (needs the
    recording-times table with mouse_id, recording, t_post_inj_min);
    (d) consecutive-frame correlation by state (bars: medians; the minimum of
    each state is annotated).
    """
    run_dir = Path(run_dir)
    motion_dir = run_dir / "motion_quality"
    out_dir = run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("per_window_motion.csv", "windows_joined_predictions.csv"):
        if not (motion_dir / name).exists():
            raise FileNotFoundError(f"{motion_dir / name}\nRun scripts/compute_motion_quality.py first.")
    windows = pd.read_csv(motion_dir / "windows_joined_predictions.csv")
    windows["state"] = [_state_name(int(value)) for value in windows["label"]]
    animals = sorted(windows["mouse_id"].unique(), key=lambda value: [int(t) if t.isdigit() else t for t in
                                                                     re.split(r"(\d+)", str(value))])
    states = [state for state in STATE_NAMES if state in set(windows["state"])]

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.9))
    ax_a, ax_b, ax_c, ax_d = axes

    # (a) motion index by state
    data = [windows.loc[windows["state"] == state, "motion_diff"].to_numpy(dtype=float) for state in states]
    ax_a.boxplot(data, tick_labels=states, showfliers=False, widths=0.55,
                 medianprops=dict(color="black"), whiskerprops=dict(color="#555555"), capprops=dict(color="#555555"))
    offsets = np.linspace(-0.14, 0.14, len(animals))
    medians = windows.groupby(["mouse_id", "state"])["motion_diff"].median().unstack()
    for index, animal in enumerate(animals):
        for position, state in enumerate(states):
            if state in medians.columns and animal in medians.index:
                ax_a.scatter(position + 1 + offsets[index], medians.loc[animal, state], s=16, zorder=3,
                             color=STATE_COLORS[state], edgecolor="black", lw=0.4)
    ax_a.set_ylabel("Frame-difference motion index (A/D levels)")
    ax_a.set_title("(a) Motion index by state", fontsize=10)

    # (b) per-animal median motion across states
    for animal in medians.index:
        values = [medians.loc[animal, state] for state in states if state in medians.columns]
        ax_b.plot(np.arange(len(values)), values, "o-", lw=1.0, ms=4, alpha=0.8)
    ax_b.set_xticks(np.arange(len(states)))
    ax_b.set_xticklabels(states)
    ax_b.set_ylabel("Median motion index")
    ax_b.set_title(f"(b) Per-animal medians (n = {len(medians)})", fontsize=10)

    # (c) mean motion per recording against post-injection time
    per_recording = (windows.groupby(["mouse_id", "original_file", "state"], as_index=False)
                     .agg(mean_motion=("motion_diff", "mean")))
    if recording_times and Path(recording_times).exists():
        times = pd.read_csv(recording_times)
        times["mouse_id"] = times["mouse_id"].astype(str).str.strip()
        times["recording"] = times["recording"].astype(str).str.strip()
        lookup = times.set_index(["mouse_id", "recording"])["t_post_inj_min"]
        per_recording["minutes"] = [lookup.get((str(mouse), str(rec)), np.nan)
                                    for mouse, rec in zip(per_recording["mouse_id"], per_recording["original_file"])]
        for state in states:
            sub = per_recording[(per_recording["state"] == state) & per_recording["minutes"].notna()]
            ax_c.scatter(sub["minutes"], sub["mean_motion"], s=28, color=STATE_COLORS[state], edgecolor="black", lw=0.4,
                         label=state)
        ax_c.legend(fontsize=7.5, frameon=False)
        ax_c.set_xlabel("Time after injection (min)")
    else:
        ax_c.text(0.5, 0.5, "recording-times table not given", ha="center", va="center", transform=ax_c.transAxes, fontsize=9)
    ax_c.set_ylabel("Mean motion index per recording")
    ax_c.set_title("(c) Motion against post-injection time", fontsize=10)

    # (d) consecutive-frame correlation by state
    medians_corr = [float(windows.loc[windows["state"] == state, "frame_corr"].median()) for state in states]
    minima_corr = [float(windows.loc[windows["state"] == state, "frame_corr_min"].min()) for state in states]
    ax_d.bar(np.arange(len(states)), medians_corr, color=[STATE_COLORS[state] for state in states], alpha=0.85, width=0.6)
    for index, (median, minimum) in enumerate(zip(medians_corr, minima_corr)):
        ax_d.text(index, median, f"median {median:.5f}\nmin {minimum:.5f}", ha="center", va="bottom", fontsize=7.5)
    low = min(medians_corr + minima_corr)
    ax_d.set_ylim(max(0.0, low - 0.02), 1.005)
    ax_d.set_xticks(np.arange(len(states)))
    ax_d.set_xticklabels(states)
    ax_d.set_ylabel("Consecutive-frame correlation")
    ax_d.set_title("(d) Consecutive-frame correlation", fontsize=10)
    for ax in axes:
        ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    return _save(fig, out_dir, "motion_quality")
