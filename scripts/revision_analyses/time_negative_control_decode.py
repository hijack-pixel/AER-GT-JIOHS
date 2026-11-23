"""Time negative control: early/late pseudo-class decoding (Reviewer 3, Comment 4).

Implements the control described in revised Sec. 2.9: baseline windows are
split into early and late pseudo-classes (the first and second halves of the
consecutive baseline windows within each recording) and decoded with the
identical two-stage pipeline, hyperparameters, and subject-level five-fold
protocol as the main three-state task.

What this script does:

1. Loads the labeled window samples from the configured private directory
   and keeps only baseline-labeled windows (Label == 0).
2. Groups the windows by (MouseID, OriginalFile), orders them by
   WindowIndex, and assigns pseudo-labels: first half -> 0 (early),
   second half -> 1 (late). Odd window counts are split so that the early
   half takes the extra window.
3. Writes the pseudo-labeled dataset as AERT_Sample .mat files under
   ``outputs/time_negative_control/pseudo_dataset/`` (no raw data is copied;
   only NodeFeats/Adjacency of baseline windows already present in the
   processed dataset, with pseudo-labels).
4. Derives a training config identical to ``configs/full_model.yaml``
   except for the data/output paths and runs the repo's standard
   ``train_from_config`` entry, i.e. the exact paper pipeline
   (subject-level five-fold, two-stage, fixed hyperparameters, seed 42).
5. Reads the per-fold metrics, reports the mean +/- SD pseudo-class AUC,
   and computes a permutation p value by shuffling the early/late
   pseudo-labels of the out-of-fold windows and recomputing the AUC
   (1,000 permutations; label-level ROC null).

Usage:
    python scripts/revision_analyses/time_negative_control_decode.py \
        --config configs/full_model.yaml

Inputs (from the config; point at your private data):
    data/private/labeled_epilepsy/   labeled window-level .mat samples

Outputs:
    outputs/time_negative_control/
        pseudo_dataset/          pseudo-labeled .mat samples
        per_fold_auc.csv         fold-level pseudo-class AUC
        summary.json             mean +/- SD AUC and permutation p
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.io

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aergt.config import load_config  # noqa: E402
from aergt.data.samples import load_samples, read_aert_sample  # noqa: E402


def build_pseudo_dataset(labeled_dir: Path, out_dir: Path) -> list[str]:
    """Split baseline windows into early/late pseudo-classes per recording."""
    samples = load_samples(
        labeled_dir, labeled=True, num_nodes=22, in_channels=6, use_abs_adj=True
    )
    baseline = [s for s in samples if s.label == 0]
    if not baseline:
        raise RuntimeError("No baseline-labeled windows were found.")

    groups: dict[tuple[str, str], list] = defaultdict(list)
    for s in baseline:
        groups[(s.mouse_id, s.original_file or s.source_file)].append(s)

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    n_early = n_late = 0
    for (mouse, recording), wins in sorted(groups.items()):
        wins.sort(key=lambda s: s.window_index)
        half = (len(wins) + 1) // 2
        for i, s in enumerate(wins):
            pseudo = 0 if i < half else 1  # 0 = early, 1 = late
            if pseudo == 0:
                n_early += 1
            else:
                n_late += 1
            stem = Path(s.source_file).stem
            out_path = out_dir / f"{stem}_pseudo{pseudo}.mat"
            scipy.io.savemat(
                str(out_path),
                {"AERT_Sample": {
                    "NodeFeats": s.x,
                    "Adjacency": s.adj,
                    "Label": pseudo,
                    "MouseID": mouse,
                    "OriginalFile": recording,
                    "WindowIndex": s.window_index,
                }},
            )
            written.append(str(out_path))

    # self-check: the repo loader must read back what we wrote
    probe = read_aert_sample(written[0], labeled=True)
    if probe is None or probe.label not in (0, 1):
        raise RuntimeError("Pseudo-sample self-check failed; loader could not re-read the file.")
    print(f"Pseudo dataset: {len(written)} windows ({n_early} early / {n_late} late) "
          f"from {len(groups)} recordings -> {out_dir}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/full_model.yaml")
    parser.add_argument("--permutations", type=int, default=1000)
    args = parser.parse_args()

    repo_root = ROOT
    cfg = load_config(repo_root / args.config)

    out_root = repo_root / "outputs" / "time_negative_control"
    pseudo_dir = out_root / "pseudo_dataset"
    build_pseudo_dataset(repo_root / cfg["data"]["labeled_dir"], pseudo_dir)

    derived = copy.deepcopy(cfg)
    derived["project"]["output_dir"] = "outputs/time_negative_control"
    derived["data"]["labeled_dir"] = str(pseudo_dir.relative_to(repo_root))

    from aergt.training.runner import train_from_config
    train_from_config(derived, repo_root)

    # Collect per-fold AUC from the training outputs.
    rows = []
    for fold in range(1, int(derived["training"]["num_folds"]) + 1):
        hist = list(csv.DictReader(
            open(out_root / "checkpoints" / f"fold_{fold}_history.csv")
        )) if (out_root / "checkpoints" / f"fold_{fold}_history.csv").exists() else []
        if hist:
            rows.append(hist[-1])
    if not rows:
        print("NOTE: read the per-fold metrics written by train_from_config under "
              f"{out_root} and compute the pseudo-class AUC from the out-of-fold "
              "prediction files (oof_predictions.csv) with the helper below.")

    rng = np.random.default_rng(42)
    aucs = []
    perm_p = None
    oof = out_root / "oof_predictions.csv"
    if oof.exists():
        data = list(csv.DictReader(open(oof)))
        y = np.array([int(r["label"]) for r in data])
        p = np.array([float(r["p_class1"]) if "p_class1" in r else float(r["prob_1"])
                      for r in data])
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y, p)
        aucs = [auc]
        null = [roc_auc_score(rng.permutation(y), p) for _ in range(args.permutations)]
        perm_p = float((np.sum(np.array(null) >= auc) + 1) / (args.permutations + 1))

    summary = {
        "task": "early/late pseudo-class time negative control",
        "auc_mean_sd": (f"{np.mean(aucs):.3f} +/- {np.std(aucs, ddof=1):.3f}"
                        if len(aucs) > 1 else (f"{aucs[0]:.3f}" if aucs else None)),
        "permutation_p": perm_p,
        "n_permutations": args.permutations,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
