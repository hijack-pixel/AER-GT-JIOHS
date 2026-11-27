"""Decomposition of the pharmacological-control decoding.

Two analyses of the seizure-free pilocarpine control cohort:

(a) Time decomposition. Every control window is assigned a class by the trained
    fold models (``scripts/run_control_inference.py``). Windows are grouped by
    animal and recording, and the recovery-assignment rate and the mean
    recovery-class probability are computed per recording, per animal, and for
    the pooled cohort.

(b) Deviation analysis. The network deviation of a window is 1 - r between the
    upper-triangle profiles of its ROI correlation matrix and the animal's
    baseline connectivity template. For the ictal-transition animals the
    template is the mean correlation matrix of the animal's baseline-labelled
    windows; a control animal has no pre-seizure recording, so its template is
    the mean correlation matrix of its earliest recording. Control windows are
    split by their assigned class, and four groups (ictal recovery, ictal
    baseline, control windows assigned to recovery, control windows assigned to
    baseline) are summarised by their median and quartiles. Windows of a
    recording overlap and the control cohort has few animals, so the groups
    are compared descriptively.

Usage:
    python scripts/supplementary_analyses/control_decomposition_analysis.py \
        --config configs/full_model.yaml \
        --control-dir data/private/control_pharmacological \
        --predictions outputs/<run>/control_inference/control_window_predictions.csv \
        --recording-times recording_times.csv

Inputs:
    --control-dir       control window samples (Label = -1) written by the MATLAB
                        preprocessing and read by ``run_control_inference.py``
    --predictions       control_window_predictions.csv of that run
    --recording-times   CSV with mouse_id, recording, t_post_inj_min (one row per
                        control recording; ``recording`` is the OriginalFile field
                        of the samples)
    labeled window samples of the run configuration (deviation reference)

Output:
    outputs/supplementary_analyses/control_decomposition/
        decomposition_table.csv     counts, rates, and mean probability per recording, animal, and pooled
        deviation_groups.csv        n, median, and quartiles per group
        time_decomposition.png / .pdf
        deviation_boxplot.png / .pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _common import ROOT, display_map, load_run_config, read_recording_times, resolve_path  # noqa: E402


def recording_name(sample) -> str:
    return sample.original_file or Path(sample.source_file).stem


def load(cfg: dict, folder: Path, *, labeled: bool):
    from aergt.data.samples import load_samples

    return load_samples(folder, labeled=labeled, num_nodes=int(cfg["model"]["num_nodes"]),
                        in_channels=int(cfg["model"]["in_channels"]), use_abs_adj=False)


def deviations(frame: pd.DataFrame, templates: dict, upper: tuple, class_index: dict) -> dict[str, np.ndarray]:
    """Network deviation (1 - r against the animal's template) of the windows of each class."""
    out = {}
    for name, index in class_index.items():
        select = (frame["label"] == index).to_numpy() if "label" in frame.columns else (frame["pred_class"] == index).to_numpy()
        profiles, mice = frame.loc[select, "upper"].tolist(), frame.loc[select, "mouse_id"].tolist()
        out[name] = np.array([1.0 - float(np.corrcoef(profile, templates[mouse])[0, 1]) for profile, mouse in zip(profiles, mice)])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/full_model.yaml")
    parser.add_argument("--control-dir", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--recording-times", required=True)
    parser.add_argument("--out", default="outputs/supplementary_analyses/control_decomposition")
    args = parser.parse_args()

    cfg = load_run_config(args.config)
    classes = [str(name).lower() for name in cfg["model"]["class_names"]]
    class_index = {name: index for index, name in enumerate(classes)}
    count_columns = [f"n_{name}" for name in classes]
    upper = np.triu_indices(int(cfg["model"]["num_nodes"]), k=1)
    times = read_recording_times(args.recording_times)
    tmin = {(str(row.mouse_id), str(row.recording)): float(row.t_post_inj_min) for row in times.itertuples(index=False)}

    labeled = load(cfg, resolve_path(cfg["data"]["labeled_dir"]), labeled=True)
    control = load(cfg, resolve_path(args.control_dir), labeled=False)
    names = display_map([s.mouse_id for s in labeled] + [s.mouse_id for s in control])
    label_of = lambda value: names.get(str(value), str(value))  # noqa: E731

    # ---------- control windows and their assigned classes ----------
    windows = pd.DataFrame([{"mouse_id": s.mouse_id, "recording": recording_name(s), "t_post_inj_min": tmin[(s.mouse_id, recording_name(s))],
                             "file": Path(s.source_file).name, "window_index": s.window_index, "upper": s.adj[upper]}
                            for s in control])
    predictions = pd.read_csv(resolve_path(args.predictions))
    required = {"mouse_id", "source_file", "window_index", "pred_class", "p_recovery"}
    if not required.issubset(predictions.columns):
        raise SystemExit(f"predictions CSV lacks columns {sorted(required - set(predictions.columns))}")
    predictions["mouse_id"] = predictions["mouse_id"].astype(str).str.strip()
    predictions["file"] = predictions["source_file"].astype(str).map(lambda value: Path(value).name)
    merged = windows.merge(predictions[["mouse_id", "file", "window_index", "pred_class", "p_recovery"]],
                           on=["mouse_id", "file", "window_index"], how="inner", validate="one_to_one")
    if merged.empty:
        raise SystemExit("no control window matched a prediction row")
    if len(merged) < len(windows):
        print(f"NOTE: {len(windows) - len(merged)} control windows have no prediction row and are excluded")

    # ---------- (a) time decomposition ----------
    assigned = merged.assign(**{column: (merged["pred_class"] == index).astype(int) for index, column in enumerate(count_columns)})
    per_recording = (assigned.groupby(["mouse_id", "recording", "t_post_inj_min"], as_index=False)
                     .agg(n_windows=("file", "size"), **{column: (column, "sum") for column in count_columns},
                          mean_p_recovery=("p_recovery", "mean")))
    per_animal = (per_recording.groupby("mouse_id", as_index=False)
                  .agg(n_windows=("n_windows", "sum"), **{column: (column, "sum") for column in count_columns},
                       mean_p_recovery=("mean_p_recovery", "mean")))
    per_animal["recording"], per_animal["level"] = "all", "animal"
    per_recording["level"] = "recording"
    pooled = {"mouse_id": "Pooled", "recording": "all", "level": "pooled", "t_post_inj_min": np.nan,
              "n_windows": int(len(merged)), **{column: int(assigned[column].sum()) for column in count_columns},
              "mean_p_recovery": float(merged["p_recovery"].mean())}

    recovery_column = f"n_{classes[class_index['recovery']]}"
    table = pd.concat([per_recording, per_animal, pd.DataFrame([pooled])], ignore_index=True)
    table["pct_recovery"] = 100.0 * table[recovery_column] / table["n_windows"]
    table.insert(0, "animal", table["mouse_id"].map(label_of))
    table = (table[["animal", "level", "recording", "t_post_inj_min", "n_windows", *count_columns, "pct_recovery", "mean_p_recovery"]]
             .round({"pct_recovery": 1, "mean_p_recovery": 3}))

    # recovery-assigned share at the first and last time point of each animal
    time_points = {}
    for animal_id, group in per_recording.sort_values("t_post_inj_min").groupby("mouse_id"):
        share = 100.0 * group[recovery_column] / group["n_windows"]
        time_points[animal_id] = (float(group["t_post_inj_min"].iloc[0]), float(share.iloc[0]),
                                  float(group["t_post_inj_min"].iloc[-1]), float(share.iloc[-1]))

    # ---------- (b) deviation analysis ----------
    templates = {}
    for animal_id in sorted({s.mouse_id for s in labeled}):
        baseline = [s.adj[upper] for s in labeled if s.mouse_id == animal_id and s.label == class_index["baseline"]]
        if baseline:
            templates[animal_id] = np.mean(baseline, axis=0)
    for animal_id, group in windows.groupby("mouse_id"):
        if animal_id not in templates:
            earliest = group.loc[group["t_post_inj_min"] == group["t_post_inj_min"].min()]
            templates[animal_id] = np.mean(np.vstack(earliest["upper"]), axis=0)

    ictal_frame = pd.DataFrame({"mouse_id": [s.mouse_id for s in labeled], "label": [s.label for s in labeled],
                                "upper": [s.adj[upper] for s in labeled]})
    ictal = deviations(ictal_frame, templates, upper, class_index)
    control_assigned = deviations(merged, templates, upper, class_index)

    groups = [("Ictal recovery", ictal["recovery"]), ("Ictal baseline", ictal["baseline"]),
              ("Control to recovery", control_assigned["recovery"]), ("Control to baseline", control_assigned["baseline"])]
    deviation_table = pd.DataFrame([
        {"group": name, "n": int(values.size),
         "median": float(np.median(values)) if values.size else np.nan,
         "q25": float(np.percentile(values, 25)) if values.size else np.nan,
         "q75": float(np.percentile(values, 75)) if values.size else np.nan}
        for name, values in groups])

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "decomposition_table.csv", index=False)
    deviation_table.to_csv(out / "deviation_groups.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for animal_id, group in per_recording.groupby("mouse_id"):
        group = group.sort_values("t_post_inj_min")
        axes[0].plot(group["t_post_inj_min"], 100.0 * group[recovery_column] / group["n_windows"], "o-", label=label_of(animal_id))
        axes[1].plot(group["t_post_inj_min"], group["mean_p_recovery"], "o-", label=label_of(animal_id))
    axes[1].axhline(float(merged["p_recovery"].mean()), color="#555555", ls="--", lw=1.0)
    axes[0].set_xlabel("Time after injection (min)")
    axes[0].set_ylabel("Recovery-assigned windows (%)")
    axes[1].set_xlabel("Time after injection (min)")
    axes[1].set_ylabel("Mean recovery-class probability")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "time_decomposition.png", dpi=300)
    fig.savefig(out / "time_decomposition.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.boxplot([values for _, values in groups], showfliers=False,
               tick_labels=["Ictal\nrecovery", "Ictal\nbaseline", "Control to\nrecovery", "Control to\nbaseline"])
    ax.set_ylabel("Network deviation (1 - r)")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out / "deviation_boxplot.png", dpi=300)
    fig.savefig(out / "deviation_boxplot.pdf")
    plt.close(fig)

    pd.set_option("display.width", 200)
    print(table.to_string(index=False))
    print("\nRecovery-assigned share at the first and last time point of each animal:")
    for animal_id, (t_first, share_first, t_last, share_last) in time_points.items():
        print(f"  {label_of(animal_id):8s} {t_first:g} min: {share_first:5.1f}%   {t_last:g} min: {share_last:5.1f}%")
    print()
    print(deviation_table.to_string(index=False))
    print(f"\nsaved -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
