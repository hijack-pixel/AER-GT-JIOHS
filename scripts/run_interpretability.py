"""Run the model-based interpretability analyses of Sec. 2.8 (Fig. 7).

For each subject-level fold, the fold's trained model is applied to its
held-out animal only (no additional training or tuning) and the following
quantities are produced:

- the recovery-class readout weight of every ROI (mean over the held-out
  recovery windows, centered and scaled across ROIs within the fold);
- the Recovery-F1 drop (percentage points) when the node features of one ROI
  are masked, for every ROI;
- the recovery-minus-baseline and recovery-minus-seizure readout contrasts;
- the readout-weighted recovery-minus-baseline ROI interaction contrast map
  with its top-10% edges (head-averaged, symmetrized attention matrices).

Usage:
    python scripts/run_interpretability.py --config configs/full_model.yaml

Without ``--artifacts``, the checkpoints and scalers of the run are used: the
directory named in the configuration (``artifacts.artifact_dir``) or, if none is
named, the output directory of the run.

Output (``outputs/<run>/interpretability/``):
    interpretability_summary.json            cross-fold results
    readout_weights_by_fold.csv              fold-normalized mean readout per ROI, state, and fold
    mask_recovery_f1_drop_by_fold.csv        Recovery-F1 drop per ROI and fold
    readout_contrasts_per_roi.csv            cross-fold mean contrasts per ROI
    interaction_contrast_matrix.csv          cross-fold mean interaction contrast (ROI x ROI)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aergt.config import load_config  # noqa: E402
from aergt.data.samples import load_samples  # noqa: E402
from aergt.interpretability.roi_analysis import run_all  # noqa: E402
from aergt.training.runner import _model_kwargs  # noqa: E402
from aergt.utils.runtime import resolve_device  # noqa: E402


def artifact_dir_of(cfg: dict, repo_root: Path) -> Path:
    named = cfg.get("artifacts", {}).get("artifact_dir", cfg["project"]["output_dir"])
    return (repo_root / named).resolve()


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sec. 2.8 interpretability analyses.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "full_model.yaml"))
    parser.add_argument("--artifacts", default="", help="checkpoint and scaler directory (default: the run's own)")
    parser.add_argument("--out-dir", default="", help="default: outputs/<run>/interpretability")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    cfg = load_config(args.config)
    samples = load_samples(ROOT / cfg["data"]["labeled_dir"], labeled=True, num_nodes=int(cfg["model"]["num_nodes"]),
                           in_channels=int(cfg["model"]["in_channels"]), use_abs_adj=bool(cfg["data"]["use_abs_adjacency"]))
    device = resolve_device(str(cfg["hardware"]["device"]))
    result = run_all(samples, args.artifacts or artifact_dir_of(cfg, ROOT), _model_kwargs(cfg),
                     device=device, top_k=args.top_k)

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / cfg["project"]["output_dir"] / "interpretability"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "interpretability_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    per_fold = result["per_fold"]
    n_folds = len(per_fold["held_out_subjects"])
    states = list(per_fold["readout_by_state"])
    rows = []
    for roi in range(len(result["mean_recovery_readout"])):
        row = [roi]
        for state in states:
            row += ["" if per_fold["readout_by_state"][state][fold] is None
                    else round(per_fold["readout_by_state"][state][fold][roi], 6) for fold in range(n_folds)]
        rows.append(row)
    write_csv(out_dir / "readout_weights_by_fold.csv",
              ["roi_index", *[f"{state}_fold{fold + 1}" for state in states for fold in range(n_folds)]], rows)
    rows = [[roi, *[round(per_fold["mask_recovery_f1_drop_pp"][fold][roi], 4) for fold in range(n_folds)],
             round(result["mask_recovery_f1_drop_pp_mean"][roi], 4)]
            for roi in range(len(result["mask_recovery_f1_drop_pp_mean"]))]
    write_csv(out_dir / "mask_recovery_f1_drop_by_fold.csv",
              ["roi_index", *[f"fold{fold + 1}" for fold in range(n_folds)], "mean"], rows)
    rows = [[roi, *[round(result["readout_contrasts_per_roi"][name][roi], 6) for name in result["readout_contrasts_per_roi"]]]
            for roi in range(len(result["mean_recovery_readout"]))]
    write_csv(out_dir / "readout_contrasts_per_roi.csv",
              ["roi_index", *result["readout_contrasts_per_roi"]], rows)
    if result["interaction_contrast"] is not None:
        write_csv(out_dir / "interaction_contrast_matrix.csv",
                  ["roi_index", *[f"roi{roi}" for roi in range(len(result["interaction_contrast"]))]],
                  [[roi, *[round(value, 6) for value in row]] for roi, row in enumerate(result["interaction_contrast"])])

    print("top ROIs:", result["top_roi_indices"])
    print("mean Top-8 readout contrast (recovery - baseline):",
          result["readout_contrasts_topk_mean"]["recovery_minus_baseline"])
    print(f"outputs -> {out_dir}")


if __name__ == "__main__":
    main()
