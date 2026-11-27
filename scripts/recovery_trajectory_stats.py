"""Recovery-trajectory quantification (Sec. 3.2, Fig. 5).

Tests whether, within the recovery period, the predicted baseline-class
probability rises with post-injection time:

- the within-animal Spearman rank correlation between the predicted
  baseline-class probability of the recovery windows and their post-injection
  time (one rho per held-out animal);
- a one-sided sign test across animals (the probability of at least the observed
  number of positive rhos under p = 0.5), reported with the mean rho.

The post-injection time of a window is the time of its recording as given in
the recording-times table plus the offset of the window within the recording
(window index - 1, in strides of 40 frames at 10 Hz, i.e. 4 s).

Usage:
    python scripts/recovery_trajectory_stats.py \
        --config configs/full_model.yaml --recording-times recording_times.csv

Inputs:
    --recording-times CSV   columns mouse_id, recording, t_post_inj_min (one row per
                            recording; ``recording`` is the OriginalFile field of
                            the window samples)
    outputs/<run>/evaluation/oof_predictions.csv   written by scripts/run_inference.py

Output:
    outputs/<run>/recovery_trajectory/
        per_animal_spearman.csv
        recovery_trajectory_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aergt.config import load_config  # noqa: E402

STRIDE_MIN = 40 / 10.0 / 60.0          # 40-frame stride at 10 Hz, in minutes


def main() -> None:
    parser = argparse.ArgumentParser(description="Recovery-trajectory statistics (Fig. 5).")
    parser.add_argument("--config", default=str(ROOT / "configs" / "full_model.yaml"))
    parser.add_argument("--recording-times", required=True, help="CSV with mouse_id, recording, t_post_inj_min")
    parser.add_argument("--out-dir", default="", help="default: outputs/<run>/recovery_trajectory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    classes = [str(name).lower() for name in cfg["model"]["class_names"]]
    recovery, baseline = classes.index("recovery"), classes.index("baseline")
    times = pd.read_csv(args.recording_times)
    for column in ("mouse_id", "recording"):
        if column not in times.columns:
            raise SystemExit(f"recording-times CSV lacks the column {column!r}")
        times[column] = times[column].astype(str).str.strip()
    tmin = {(row.mouse_id, row.recording): float(row.t_post_inj_min) for row in times.itertuples(index=False)}

    oof = pd.read_csv(ROOT / cfg["project"]["output_dir"] / "evaluation" / "oof_predictions.csv")
    oof["mouse_id"] = oof["mouse_id"].astype(str).str.strip()
    recovery_windows = oof[oof["label"] == recovery].copy()
    if recovery_windows.empty:
        raise SystemExit("no recovery-labelled windows in the out-of-fold predictions")
    keys = list(zip(recovery_windows["mouse_id"], recovery_windows["original_file"].astype(str)))
    missing = sorted({key for key in keys if key not in tmin})
    if missing:
        raise SystemExit(f"no post-injection time for recordings: {missing}")
    recovery_windows["minutes"] = [tmin[key] + (window_index - 1) * STRIDE_MIN
                                   for key, window_index in zip(keys, recovery_windows["window_index"])]

    animal_ids = sorted(recovery_windows["mouse_id"].unique())
    rows = []
    for animal in animal_ids:
        sub = recovery_windows[recovery_windows["mouse_id"] == animal]
        if len(sub) < 3 or sub["minutes"].nunique() < 2:
            continue
        rho, p = spearmanr(sub["minutes"].to_numpy(), sub[f"prob_{classes[baseline]}"].to_numpy())
        rows.append({"animal": f"Mouse {animal_ids.index(animal) + 1}", "n_windows": int(len(sub)),
                     "n_recordings": int(sub["original_file"].nunique()), "spearman_rho": float(rho), "p": float(p)})

    rhos = np.asarray([row["spearman_rho"] for row in rows], dtype=float)
    n_pos = int((rhos > 0).sum())
    sign_p = float(binomtest(n_pos, len(rhos), 0.5, alternative="greater").pvalue) if len(rhos) else float("nan")
    summary = {
        "n_animals": len(rows),
        "n_positive": n_pos,
        "mean_rho": float(rhos.mean()) if rhos.size else float("nan"),
        "sign_test_p_one_sided": sign_p,
        "per_animal": rows,
    }

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / cfg["project"]["output_dir"] / "recovery_trajectory"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "per_animal_spearman.csv", index=False)
    (out_dir / "recovery_trajectory_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"outputs -> {out_dir}")


if __name__ == "__main__":
    main()
