"""Recovery-trajectory quantification (Sec. 3.2, Fig. 5).

Implements the quantification described in the manuscript: during the
recovery period, the predicted **baseline-class probability** is tested for
a monotonic increase with post-injection time **within each held-out
animal**:

* within-animal Spearman rank correlation between the predicted
  baseline-class probability and post-injection time (one rho per animal);
* a one-sided sign test across animals (all rhos positive gives
  p = 1/2^n animals), reported together with the mean rho.

Input: the out-of-fold predictions CSV produced by ``scripts/run_inference.py``
(columns ``mouse_id``, ``source_file``, ``true_label_name``,
``p_baseline``/class probabilities). Post-injection time is parsed from the
recording filename, exactly as in the preprocessing pipeline.

Usage:
    python scripts/recovery_trajectory_stats.py \
        --oof-predictions outputs/full_model/evaluation/oof_predictions.csv \
        --recovery-label recovery \
        --out-dir outputs/recovery_trajectory

Outputs are written under ``outputs/`` and are not versioned.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def minutes_from_source(path: str) -> float:
    """Post-injection minutes parsed from a sample filename such as
    ``Mouse_5_10min_tif_Win3.mat`` or ``Mouse_3_1_5h_seizure_tif_Win7.mat``."""
    base = Path(str(path)).name
    m = re.search(r"Mouse_\d+_(.+?)_Win\d+", base)
    token = m.group(1) if m else base
    token = token.replace("_tif", "")
    m = re.match(r"(\d+(?:\.\d+)?)\s*min", token, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.match(r"(\d+(?:\.\d+)?)\s*_?\s*h", token, flags=re.IGNORECASE)
    if m:
        return float(m.group(1)) * 60.0
    m = re.match(r"(\d+)", token)
    return float(m.group(1)) if m else float("nan")


def probability_column(oof: pd.DataFrame, class_label: str) -> str:
    """Locate the predicted-probability column for one class."""
    candidates = [f"p_{class_label}", f"prob_{class_label}", class_label]
    for name in candidates:
        if name in oof.columns:
            return name
    numeric = [c for c in oof.columns if c.startswith(("p_", "prob_"))]
    if len(numeric) == 3:
        order = ["baseline", "seizure", "recovery"]
        return numeric[order.index(class_label)]
    raise KeyError(f"no probability column for class '{class_label}'; "
                   f"available columns: {list(oof.columns)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recovery-trajectory statistics (Fig. 5).")
    parser.add_argument("--oof-predictions", default=str(
        ROOT / "outputs" / "full_model" / "evaluation" / "oof_predictions.csv"))
    parser.add_argument("--recovery-label", default="recovery",
                        help="name of the recovery class in the CSV")
    parser.add_argument("--trajectory-class", default="baseline",
                        help="class whose predicted probability is tested for increase")
    parser.add_argument("--out-dir", default=str(ROOT / "outputs" / "recovery_trajectory"))
    args = parser.parse_args()

    oof = pd.read_csv(args.oof_predictions)
    label_col = "true_label_name" if "true_label_name" in oof.columns else "true_label_id"
    rec = oof[oof[label_col].astype(str).str.lower() == args.recovery_label.lower()].copy()
    if rec.empty:
        raise SystemExit(f"no {args.recovery_label}-labelled windows in {args.oof_predictions}")

    rec["minutes"] = rec["source_file"].map(minutes_from_source)
    rec = rec.dropna(subset=["minutes"])

    prob_col = probability_column(oof, args.trajectory_class)
    animal_ids = sorted(rec["mouse_id"].unique())

    rows = []
    for animal in animal_ids:
        sub = rec[rec["mouse_id"] == animal]
        if len(sub) < 3:
            continue
        rho, _ = spearmanr(sub["minutes"].to_numpy(), sub[prob_col].to_numpy())
        rows.append({"mouse_id": animal, "n_windows": int(len(sub)),
                     "spearman_rho": float(rho)})

    rhos = np.asarray([r["spearman_rho"] for r in rows], dtype=float)
    n_pos = int((rhos > 0).sum())
    # one-sided sign test: probability of >= n_pos positive rhos under H0
    from scipy.stats import binomtest
    sign_p = float(binomtest(n_pos, len(rhos), 0.5, alternative="greater").pvalue) \
        if len(rhos) else float("nan")

    summary = {
        "n_animals": len(rows),
        "n_positive": n_pos,
        "mean_rho": float(rhos.mean()) if rhos.size else float("nan"),
        "sign_test_p_one_sided": sign_p,
        "per_animal": rows,
        "trajectory_class": args.trajectory_class,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "per_animal_spearman.csv", index=False)
    (out_dir / "recovery_trajectory_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
