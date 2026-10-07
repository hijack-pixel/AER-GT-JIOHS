"""Run the pharmacological-control cohort evaluation of Sec. 2.9 / Sec. 3.3.

Decodes all preprocessed control windows with the five paper-trained fold
models (ensemble = mean softmax, per-fold scalers) and reports the
class-assignment proportions per animal and overall.

Usage:
    python scripts/run_control_inference.py \
        --control-dir data/private/control_epilepsy \
        --artifacts model_artifacts/paper_trained \
        --config configs/paper_trained_inference.yaml

All numeric outputs are written under ``outputs/control_inference/`` and are
regenerated at run time; no result tables are versioned in the repository.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aergt.config import load_config
from aergt.evaluation.control_inference import (
    decode_control_cohort,
    load_control_samples,
    load_fold_models,
    write_control_outputs,
)
from aergt.training.runner import _model_kwargs
from aergt.utils.runtime import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Pharmacological-control decoding (Sec. 2.9).")
    parser.add_argument("--config", default=str(ROOT / "configs" / "paper_trained_inference.yaml"))
    parser.add_argument("--control-dir", default=str(ROOT / "data" / "private" / "control_epilepsy"))
    parser.add_argument("--artifacts", default=str(ROOT / "model_artifacts" / "paper_trained"))
    parser.add_argument("--out-dir", default=str(ROOT / "outputs" / "control_inference"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    samples = load_control_samples(args.control_dir)
    print(f"loaded {len(samples)} control windows")

    # reuse the exact model construction used at training time so the
    # paper-trained checkpoints load without key mismatches
    model_kwargs = _model_kwargs(cfg)
    fold_entries = load_fold_models(args.artifacts, model_kwargs=model_kwargs)
    device = resolve_device(cfg.get("device", "auto"))
    result = decode_control_cohort(samples, fold_entries, device=device)

    write_control_outputs(result, samples, args.out_dir)
    s = result["summary"]
    print("overall:", {k: round(v, 4) for k, v in s["overall"].items()})
    print("mean baseline probability:", round(s["mean_baseline_probability"], 4))
    print(f"outputs -> {args.out_dir}")


if __name__ == "__main__":
    main()
