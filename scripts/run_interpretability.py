"""Run the full model-based interpretability pipeline of Sec. 2.8 (Fig. 7).

For each subject-level fold, the fold-specific trained model is applied to
its held-out subjects only - no additional training or fold-specific
tuning - and the following quantities are produced:

* cross-fold normalized recovery-class AER readout weights and the Top-8
  ROI ranking;
* per-ROI Recovery-F1 drop under single-ROI feature masking (percentage
  points);
* recovery-minus-baseline and recovery-minus-seizure readout contrasts for
  the Top-8 ROIs;
* the readout-weighted recovery-minus-baseline ROI interaction contrast map
  with the top-10% edges (head-averaged, symmetrized global-attention
  matrices).

Usage:
    python scripts/run_interpretability.py \
        --config configs/full_model.yaml \
        --artifacts model_artifacts/paper_trained \
        --out-dir outputs/interpretability

Numeric outputs are written under ``outputs/`` and are not versioned.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aergt.config import load_config
from aergt.data.samples import load_samples
from aergt.interpretability.roi_analysis import run_all
from aergt.training.runner import _model_kwargs
from aergt.utils.runtime import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Sec. 2.8 interpretability analyses.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "full_model.yaml"))
    parser.add_argument("--artifacts", default=str(ROOT / "model_artifacts" / "paper_trained"))
    parser.add_argument("--out-dir", default=str(ROOT / "outputs" / "interpretability"))
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    cfg = load_config(args.config)
    samples = load_samples(cfg["data"]["labeled_dir"])
    print(f"loaded {len(samples)} labelled samples")

    device = resolve_device(cfg.get("device", "auto"))
    result = run_all(samples, args.artifacts, _model_kwargs(cfg),
                     device=device, top_k=args.top_k)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "interpretability_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print("top ROIs:", result["top_roi_indices"])
    print("mean Top-8 readout contrast (recovery - baseline):",
          result["readout_contrasts_topk_mean"]["recovery_minus_baseline"])
    print(f"outputs -> {out_dir}")


if __name__ == "__main__":
    main()
