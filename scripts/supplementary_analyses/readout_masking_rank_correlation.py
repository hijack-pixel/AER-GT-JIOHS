"""Readout-masking rank concordance.

The manuscript ranks ROIs in two ways: by their recovery-class readout weight
(Fig. 7a) and by the Recovery-F1 drop when their node features are masked
(Fig. 7b). This script quantifies how well the two rankings agree:

- Spearman rank correlation between the cross-fold mean readout weight and the
  cross-fold mean Recovery-F1 drop over all ROIs, with a permutation null
  (``--permutations`` random re-pairings of the ROI labels). The reported p
  values are the shares of permutations whose null correlation is at least as
  large as the observed one (one-sided) or at least as large in absolute value
  (two-sided);
- the same correlation within the Top-k ROIs of the readout ranking;
- the fold-to-fold stability of each ranking (mean pairwise Spearman
  correlation between the fold-level rankings) against a permutation null that
  re-pairs ROI labels within every fold.

Usage:
    python scripts/supplementary_analyses/readout_masking_rank_correlation.py \
        --config configs/full_model.yaml

Input (written by ``scripts/run_interpretability.py``):
    outputs/<run>/interpretability/readout_weights_by_fold.csv
    outputs/<run>/interpretability/mask_recovery_f1_drop_by_fold.csv

Output:
    outputs/supplementary_analyses/readout_masking/
        roi_ranking_table.csv     readout weight, masking drop, and both ranks per ROI
        summary.txt
        scatter.png / .pdf
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from _common import ROOT, evaluation_dir, load_run_config  # noqa: E402


def permutation_test(x: np.ndarray, y: np.ndarray, *, permutations: int, rng: np.random.Generator) -> dict[str, float]:
    """Spearman correlation of x and y against a null that re-pairs the ROI labels.

    p values are shares of permutations: ``p_one_sided_greater`` counts null
    correlations at least as large as the observed one, ``p_two_sided`` counts
    null correlations at least as large in absolute value.
    """
    rho = float(stats.spearmanr(x, y)[0])
    null = np.array([float(stats.spearmanr(x, rng.permutation(y))[0]) for _ in range(permutations)])
    return {"rho": rho,
            "p_one_sided_greater": float(np.mean(null >= rho)),
            "p_two_sided": float(np.mean(np.abs(null) >= abs(rho)))}


def fold_stability(values: np.ndarray, *, permutations: int, rng: np.random.Generator) -> dict[str, float]:
    """Fold-to-fold stability of a ranking.

    ``values`` holds one row per ROI and one column per fold. The statistic is
    the mean pairwise Spearman correlation between the fold rankings; the null
    re-pairs the ROI labels within every fold (rows permuted independently).
    """
    n_folds = values.shape[1]

    def mean_pairwise(matrix: np.ndarray) -> float:
        return float(np.mean([stats.spearmanr(matrix[:, a], matrix[:, b])[0]
                              for a in range(n_folds) for b in range(a + 1, n_folds)]))

    observed = mean_pairwise(values)
    null = np.array([mean_pairwise(rng.permuted(values, axis=0)) for _ in range(permutations)])
    return {"mean_pairwise_rho": observed, "p_one_sided_greater": float(np.mean(null >= observed))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/full_model.yaml")
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--stability-permutations", type=int, default=2000)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="outputs/supplementary_analyses/readout_masking")
    args = parser.parse_args()

    run_dir = evaluation_dir(load_run_config(args.config)).parent
    interp = run_dir / "interpretability"
    readout_csv, mask_csv = interp / "readout_weights_by_fold.csv", interp / "mask_recovery_f1_drop_by_fold.csv"
    for path in (readout_csv, mask_csv):
        if not path.exists():
            raise SystemExit(f"Missing input file: {path}\nRun scripts/run_interpretability.py first.")
    readout_wide = pd.read_csv(readout_csv)
    mask_wide = pd.read_csv(mask_csv)
    readout_cols = [column for column in readout_wide.columns if column.startswith("recovery_fold")]
    fold_cols = [column for column in mask_wide.columns if column.startswith("fold")]
    if len(readout_cols) != len(fold_cols):
        raise SystemExit("readout and masking tables cover different numbers of folds")

    roi = readout_wide["roi_index"].to_numpy()
    readout = readout_wide[readout_cols].to_numpy(dtype=float)          # ROIs x folds
    masking = mask_wide[fold_cols].to_numpy(dtype=float)                # ROIs x folds
    valid = ~np.isnan(readout).any(axis=1) & ~np.isnan(masking).any(axis=1)
    readout, masking, roi = readout[valid], masking[valid], roi[valid]

    readout_mean, masking_mean = readout.mean(axis=1), masking.mean(axis=1)
    order = np.argsort(-readout_mean)                                   # ROIs by readout weight
    rng = np.random.default_rng(args.seed)
    overall = permutation_test(readout_mean[order], masking_mean[order], permutations=args.permutations, rng=rng)
    top = order[:args.top_k]
    top_k = permutation_test(readout_mean[top], masking_mean[top], permutations=args.permutations, rng=rng)
    stability = {"masking": fold_stability(masking[order], permutations=args.stability_permutations, rng=rng),
                 "readout": fold_stability(readout[order], permutations=args.stability_permutations, rng=rng)}

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame({
        "roi_index": roi,
        "readout_weight": readout_mean, "mask_recovery_f1_drop_pp": masking_mean,
        "readout_rank": stats.rankdata(-readout_mean).astype(int),
        "masking_rank": stats.rankdata(-masking_mean).astype(int),
    }).sort_values("readout_rank").reset_index(drop=True)
    table.to_csv(out / "roi_ranking_table.csv", index=False)

    lines = [
        f"Readout ranking vs masking-drop ranking ({len(roi)} ROIs, {len(fold_cols)} folds)",
        f"  Spearman rho = {overall['rho']:.3f}   permutation p, one-sided (greater) = {overall['p_one_sided_greater']:.4f}"
        f"   two-sided = {overall['p_two_sided']:.4f}   ({args.permutations} permutations)",
        f"Top-{args.top_k} ROIs of the readout ranking:",
        f"  Spearman rho = {top_k['rho']:.3f}   permutation p, one-sided (greater) = {top_k['p_one_sided_greater']:.4f}"
        f"   two-sided = {top_k['p_two_sided']:.4f}",
        "Fold-to-fold stability of the rankings (mean pairwise Spearman correlation between folds):",
        *(f"  {name:8s} {values['mean_pairwise_rho']:.3f}   permutation p, one-sided = {values['p_one_sided_greater']:.4f}"
          for name, values in stability.items()),
    ]
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(readout_mean, masking_mean, s=42, color="#2ca02c", zorder=3,
               label=f"{len(roi)} ROIs, rho = {overall['rho']:.2f}")
    for index in np.union1d(top, np.argsort(-masking_mean)[:args.top_k]):
        ax.annotate(str(roi[index]), (readout_mean[index], masking_mean[index]), fontsize=8,
                    xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("Recovery-class readout weight")
    ax.set_ylabel("Recovery-F1 drop when masked (pp)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "scatter.png", dpi=300)
    fig.savefig(out / "scatter.pdf")

    print("\n".join(lines))
    print(table.to_string(index=False))
    print(f"\nsaved -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
