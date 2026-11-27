"""Time-based negative control: early/late pseudo-class decoding.

In the three-state task, baseline windows come from the early post-injection
period and recovery windows from the post-seizure period, so the state labels
are collinear with elapsed time. This control asks whether the pipeline can
decode elapsed time itself. Windows are split by time into two pseudo-classes
and decoded with the identical two-stage pipeline, hyperparameters, and
subject-level protocol as the three-state task (``train_from_config`` with the
run configuration; only the pseudo-labels, the class names, the number of
classes, the number of folds, and the output location differ).

Two cohorts can be analysed (``--cohort``):

- ``baseline`` (default): the baseline windows of the ictal-transition
  recordings. The pseudo-class follows the post-injection time of the
  recording: early (windows of recordings up to ``--early-max-min`` minutes
  after injection, default 15, i.e. the ~10-min recordings) and late (windows
  of the later baseline recordings, ~20-30 min). Folds are the subject-level
  folds of the main task. An animal whose baseline recordings all fall into one
  pseudo-class has a held-out fold without both classes; its AUC is undefined
  and it is excluded from the pseudo-class summary.
- ``nb``: the windows of the normative-baseline (saline) recordings. Within
  every recording the windows are ordered in time; the first half form the
  early pseudo-class and the remaining windows the late pseudo-class. The
  folds follow ``--fold-assignment`` (CSV with mouse_id and fold): the fold
  label of an animal becomes the identifier of its pseudo-samples, so that the
  subject-level protocol holds out one fold (batch) at a time.

The fold-level AUCs are tested against chance (0.5) with an exact sign-flip
test over the fold-level deviations.

Usage:
    python scripts/supplementary_analyses/time_negative_control_decode.py \
        --config configs/full_model.yaml --recording-times recording_times.csv

    python scripts/supplementary_analyses/time_negative_control_decode.py \
        --config configs/full_model.yaml --cohort nb --fold-assignment nb_batches.csv \
        --out outputs/time_negative_control_nb

    # summarize existing runs without retraining
    python scripts/supplementary_analyses/time_negative_control_decode.py --skip-training
    python scripts/supplementary_analyses/time_negative_control_decode.py --skip-training \
        --cohort nb --out outputs/time_negative_control_nb

Inputs:
    --recording-times CSV   columns mouse_id, recording, t_post_inj_min (one row per
                            recording; ``recording`` is the OriginalFile field of
                            the window samples); cohort ``baseline``
    --fold-assignment CSV   columns mouse_id, fold; cohort ``nb``
    window samples named by the run configuration (``data.labeled_dir`` or
    ``data.pretrain_dir``)

Output:
    outputs/time_negative_control[_nb]/
        pseudo_dataset/            pseudo-labeled window samples (no raw data)
        checkpoints/, scalers/, evaluation/    training and evaluation outputs
        recording_assignment.csv   pseudo-class of every recording
        per_fold.csv               per fold: class counts, AUC, Macro-F1, accuracy
        summary.json               mean +/- SD over the folds with both classes, exact test
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from _common import (ROOT, display_map, exact_sign_flip_test, load_run_config, mean_sd,  # noqa: E402
                     read_recording_times, resolve_path)

GROUP = {"baseline": "Ictal baseline windows", "nb": "NB saline windows"}


def recording_name(sample) -> str:
    return sample.original_file or Path(sample.source_file).stem


def load(cfg: dict, folder: Path, *, labeled: bool):
    from aergt.data.samples import load_samples

    return load_samples(folder, labeled=labeled, num_nodes=int(cfg["model"]["num_nodes"]),
                        in_channels=int(cfg["model"]["in_channels"]), use_abs_adj=bool(cfg["data"]["use_abs_adjacency"]))


def write_pseudo_samples(groups: dict, pseudo_of, out_dir: Path, mouse_id_of=lambda key: key[0]) -> tuple[int, pd.DataFrame]:
    """Write one pseudo-labeled sample per window.

    ``groups`` maps a recording key (animal, recording) to its windows sorted in
    time; ``pseudo_of(key, position, size)`` returns the pseudo-class (0 = early,
    1 = late) of the window at ``position``; ``mouse_id_of`` gives the identifier
    written to the samples (the animal, or the fold of the animal).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, written, first = [], 0, None
    for key, windows in sorted(groups.items()):
        animal, recording = key
        mouse = mouse_id_of(key)
        classes = [pseudo_of(key, position, len(windows)) for position in range(len(windows))]
        for sample, pseudo in zip(windows, classes):
            path = out_dir / f"{Path(sample.source_file).stem}_pseudo{pseudo}.mat"
            scipy.io.savemat(str(path), {"AERT_Sample": {
                "NodeFeats": sample.x, "Adjacency": sample.adj, "Label": pseudo,
                "MouseID": mouse, "OriginalFile": recording, "WindowIndex": sample.window_index}})
            first = first or (path, sample, pseudo)
            written += 1
        rows.append({"mouse_id": animal, "fold": mouse, "recording": recording, "n_windows": len(windows),
                     "n_early": classes.count(0), "n_late": classes.count(1)})
    if first is None:
        raise SystemExit("no windows found for the pseudo-class dataset")

    # self-check: the first written file must reproduce the sample it was built from
    path, sample, pseudo = first
    check = scipy.io.loadmat(str(path))["AERT_Sample"][0, 0]
    if int(check["Label"][0, 0]) != pseudo or not np.allclose(check["NodeFeats"], sample.x):
        raise RuntimeError(f"pseudo-sample self-check failed for {path.name}")
    return written, pd.DataFrame(rows)


def build_baseline_pseudo_dataset(cfg: dict, times: pd.DataFrame, early_max_min: float, out_dir: Path) -> pd.DataFrame:
    """Baseline windows of the ictal-transition recordings; pseudo-class by recording time."""
    samples = load(cfg, resolve_path(cfg["data"]["labeled_dir"]), labeled=True)
    baseline_class = [str(name).lower() for name in cfg["model"]["class_names"]].index("baseline")
    tmin = {(str(row.mouse_id), str(row.recording)): float(row.t_post_inj_min) for row in times.itertuples(index=False)}
    groups = defaultdict(list)
    for sample in samples:
        if sample.label == baseline_class:
            groups[(sample.mouse_id, recording_name(sample))].append(sample)
    for key in groups:
        if key not in tmin:
            raise SystemExit(f"no post-injection time for baseline recording {key}")
        groups[key].sort(key=lambda s: s.window_index)

    def pseudo_of(key, position, size):
        return 0 if tmin[key] <= early_max_min else 1

    n_written, frame = write_pseudo_samples(groups, pseudo_of, out_dir)
    frame.insert(2, "t_post_inj_min", [tmin[(mouse, recording)] for mouse, recording in zip(frame["mouse_id"], frame["recording"])])
    print(f"Pseudo dataset: {n_written} baseline windows from {len(frame)} recordings "
          f"({int(frame['n_early'].sum())} early / {int(frame['n_late'].sum())} late) -> {out_dir}")
    return frame


def build_nb_pseudo_dataset(cfg: dict, assignment: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Windows of the normative-baseline recordings.

    Within each recording the first half of the windows (in time) form the early
    pseudo-class and the rest the late pseudo-class. The fold (batch) label of an
    animal becomes the identifier of its samples.
    """
    samples = load(cfg, resolve_path(cfg["data"]["pretrain_dir"]), labeled=False)
    fold_of = {str(row.mouse_id).strip(): str(row.fold).strip() for row in assignment.itertuples(index=False)}
    groups = defaultdict(list)
    for sample in samples:
        if sample.mouse_id not in fold_of:
            raise SystemExit(f"no fold (batch) assigned to animal {sample.mouse_id!r}")
        groups[(sample.mouse_id, recording_name(sample))].append(sample)
    for key in groups:
        groups[key].sort(key=lambda s: s.window_index)
    n_written, frame = write_pseudo_samples(groups, lambda key, position, size: 0 if position < (size + 1) // 2 else 1,
                                            out_dir, mouse_id_of=lambda key: fold_of[key[0]])
    print(f"Pseudo dataset: {n_written} windows from {len(frame)} recordings "
          f"({int(frame['n_early'].sum())} early / {int(frame['n_late'].sum())} late) -> {out_dir}")
    return frame


def summarize(out_root: Path, cohort: str) -> None:
    oof = pd.read_csv(out_root / "evaluation" / "oof_predictions.csv")
    oof["mouse_id"] = oof["mouse_id"].astype(str).str.strip()
    names = display_map(oof["mouse_id"]) if cohort == "baseline" else {value: value for value in oof["mouse_id"].unique()}
    rows = []
    for animal_id, group in oof.groupby("mouse_id"):
        y = group["label"].to_numpy()
        n_early, n_late = int(np.sum(y == 0)), int(np.sum(y == 1))
        both = n_early > 0 and n_late > 0
        rows.append({
            "held_out": names[str(animal_id)],
            "n_early": n_early, "n_late": n_late,
            "auc": float(roc_auc_score(y, group["prob_late"])) if both else float("nan"),
            "macro_f1": float(f1_score(y, group["pred"], average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y, group["pred"])),
            "majority_class_rate": float(max(n_early, n_late) / len(y)),
        })
    per_fold = pd.DataFrame(rows).sort_values("held_out")
    per_fold.to_csv(out_root / "per_fold.csv", index=False)

    valid = per_fold.dropna(subset=["auc"])
    deviations = valid["auc"].to_numpy() - 0.5
    exceed, assignments = exact_sign_flip_test(deviations, alternative="two-sided")
    auc_mean, auc_sd = mean_sd(valid["auc"])
    f1_mean, f1_sd = mean_sd(valid["macro_f1"])
    acc_mean, acc_sd = mean_sd(valid["accuracy"])
    summary = {
        "cohort": cohort,
        "n_windows": int(len(oof)),
        "n_early": int(np.sum(oof["label"] == 0)), "n_late": int(np.sum(oof["label"] == 1)),
        "folds_with_both_classes": int(len(valid)),
        "folds_without_both_classes": per_fold.loc[per_fold["auc"].isna(), "held_out"].tolist(),
        "auc_mean": round(auc_mean, 4), "auc_sd": round(auc_sd, 4),
        "macro_f1_mean": round(f1_mean, 4), "macro_f1_sd": round(f1_sd, 4),
        "accuracy_mean": round(acc_mean, 4), "accuracy_sd": round(acc_sd, 4),
        "folds_above_chance": int(np.sum(deviations > 0)), "folds_below_chance": int(np.sum(deviations < 0)),
        "exact_sign_flip_test_two_sided": {"exceedances": exceed, "assignments": assignments,
                                           "p": round(exceed / assignments, 4)},
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(per_fold.to_string(index=False))
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/full_model.yaml")
    parser.add_argument("--cohort", choices=["baseline", "nb"], default="baseline")
    parser.add_argument("--recording-times", help="CSV with mouse_id, recording, t_post_inj_min (cohort baseline)")
    parser.add_argument("--early-max-min", type=float, default=15.0,
                        help="baseline recordings up to this post-injection time form the early pseudo-class")
    parser.add_argument("--fold-assignment", help="CSV with mouse_id, fold (cohort nb)")
    parser.add_argument("--out", default="", help="default: outputs/time_negative_control (or ..._nb)")
    parser.add_argument("--skip-training", action="store_true", help="only summarize an existing run")
    args = parser.parse_args()

    out_root = resolve_path(args.out or f"outputs/time_negative_control{'' if args.cohort == 'baseline' else '_nb'}")
    if out_root == ROOT or ROOT not in out_root.parents:
        raise SystemExit("the output directory must lie inside the repository, e.g. outputs/time_negative_control")
    if args.skip_training:
        summarize(out_root, args.cohort)
        return
    if args.cohort == "baseline" and not args.recording_times:
        parser.error("--recording-times is required for the baseline cohort")
    if args.cohort == "nb" and not args.fold_assignment:
        parser.error("--fold-assignment is required for the nb cohort")

    cfg = load_run_config(args.config)
    pseudo_dir = out_root / "pseudo_dataset"
    if args.cohort == "baseline":
        assignment = build_baseline_pseudo_dataset(cfg, read_recording_times(args.recording_times), args.early_max_min, pseudo_dir)
        num_folds = int(cfg["training"]["num_folds"])
    else:
        table = pd.read_csv(resolve_path(args.fold_assignment))
        if not {"mouse_id", "fold"}.issubset(table.columns):
            raise SystemExit("the fold-assignment CSV needs the columns mouse_id and fold")
        assignment = build_nb_pseudo_dataset(cfg, table, pseudo_dir)
        num_folds = int(assignment["fold"].nunique())
    out_root.mkdir(parents=True, exist_ok=True)
    assignment.to_csv(out_root / "recording_assignment.csv", index=False)

    derived = copy.deepcopy(cfg)
    derived["project"]["output_dir"] = out_root.relative_to(ROOT).as_posix()
    derived["data"]["labeled_dir"] = pseudo_dir.relative_to(ROOT).as_posix()
    derived["model"]["num_classes"] = 2
    derived["model"]["class_names"] = ["Early", "Late"]
    derived["training"]["num_folds"] = num_folds

    from aergt.evaluation.inference import evaluate_from_config
    from aergt.training.runner import train_from_config

    train_from_config(derived, repo_root=ROOT)
    evaluate_from_config(derived, repo_root=ROOT)
    summarize(out_root, args.cohort)
    print(f"\nsaved -> {out_root.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
