"""Run the pharmacological-control cohort evaluation of Sec. 2.9 / Sec. 3.3.

Decodes all preprocessed control windows with the trained fold models
(ensemble = mean softmax, per-fold scalers) and reports the class-assignment
proportions per animal and overall.

Usage:
    python scripts/run_control_inference.py \
        --config configs/paper_trained_inference.yaml \
        --control-dir data/private/control_pharmacological

Without ``--artifacts``, the checkpoints and scalers of the run are used: the
directory named in the configuration (``artifacts.artifact_dir``) or, if none is
named, the output directory of the run.

Numeric outputs are written under ``outputs/<run>/control_inference/`` and are
regenerated at run time; no result tables are versioned in the repository.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aergt.config import load_config  # noqa: E402
from aergt.evaluation.control_inference import (  # noqa: E402
    decode_control_cohort,
    load_control_samples,
    load_fold_models,
    write_control_outputs,
)
from aergt.training.runner import _model_kwargs  # noqa: E402
from aergt.utils.runtime import resolve_device  # noqa: E402


def artifact_dir_of(cfg: dict, repo_root: Path) -> Path:
    named = cfg.get("artifacts", {}).get("artifact_dir", cfg["project"]["output_dir"])
    return (repo_root / named).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Pharmacological-control decoding (Sec. 2.9).")
    parser.add_argument("--config", default=str(ROOT / "configs" / "paper_trained_inference.yaml"))
    parser.add_argument("--control-dir", default=str(ROOT / "data" / "private" / "control_pharmacological"))
    parser.add_argument("--artifacts", default="", help="checkpoint and scaler directory (default: the run's own)")
    parser.add_argument("--out-dir", default="", help="default: outputs/<run>/control_inference")
    args = parser.parse_args()

    cfg = load_config(args.config)
    class_names = [str(name).lower() for name in cfg["model"]["class_names"]]
    samples = load_control_samples(args.control_dir, num_nodes=int(cfg["model"]["num_nodes"]),
                                   in_channels=int(cfg["model"]["in_channels"]),
                                   use_abs_adj=bool(cfg["data"]["use_abs_adjacency"]))
    fold_entries = load_fold_models(args.artifacts or artifact_dir_of(cfg, ROOT), model_kwargs=_model_kwargs(cfg),
                                    num_folds=int(cfg["training"]["num_folds"]))
    device = resolve_device(str(cfg["hardware"]["device"]))
    result = decode_control_cohort(samples, fold_entries, class_names=class_names, device=device)

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / cfg["project"]["output_dir"] / "control_inference"
    write_control_outputs(result, samples, class_names, out_dir)
    summary = result["summary"]
    print("overall:", {k: round(v, 4) for k, v in summary["overall"].items()})
    print("mean class probability:", {k: round(v, 4) for k, v in summary["mean_class_probability"].items()})
    print(f"outputs -> {out_dir}")


if __name__ == "__main__":
    main()
