"""Lambda-sensitivity table and figure.

For runs that differ only in the AER regularization coefficient lambda
(``configs/lambda_0p002.yaml``, ``configs/full_model.yaml``, ``configs/lambda_0p2.yaml``)
this script tabulates decoding performance and the entropy of the attention
(readout) distribution:

- per run, the mean +/- SD over the held-out animals of Macro-F1, Recovery-F1,
  accuracy, and Macro-AUC (``fold_metrics.csv``);
- per window, the normalized attention entropy H(alpha) / ln(N_R) of the ROI
  readout weights (``fold_k_interpretability.npz``); reported as mean +/- SD over
  the out-of-fold windows, together with the entropy of the mean readout and the
  ROI with the largest mean weight.

Usage:
    python scripts/supplementary_analyses/lambda_entropy_table.py \
        --runs outputs/lambda_0p002 outputs/full_model outputs/lambda_0p2 \
        --lambdas 0.002 0.02 0.2

The run without the AER term (``configs/ablation_no_aer_regularization.yaml``) can be
included as lambda = 0:
    --runs outputs/ablation_no_aer_regularization outputs/lambda_0p002 outputs/full_model \
           outputs/lambda_0p2 --lambdas 0 0.002 0.02 0.2

Inputs (per run, written by ``scripts/run_inference.py``):
    outputs/<run>/evaluation/fold_metrics.csv
    outputs/<run>/evaluation/fold_k_interpretability.npz

Output:
    outputs/supplementary_analyses/lambda_sensitivity/
        lambda_sensitivity_table.csv      performance, entropy, and the top-ROI share of the readout
        lambda_sensitivity_figure.png / .pdf
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _common import ROOT, mean_sd, read_fold_metrics, resolve_path  # noqa: E402

METRICS = ["macro_f1", "recovery_f1", "accuracy", "macro_auc"]


def attention_entropy(run: Path) -> dict[str, object]:
    files = sorted((run / "evaluation").glob("fold_*_interpretability.npz"))
    if not files:
        raise SystemExit(f"no fold_*_interpretability.npz under {run / 'evaluation'}")
    alpha = np.vstack([np.load(path)["attention"] for path in files]).astype(float)
    if alpha.ndim != 2:
        raise SystemExit(f"unexpected attention shape {alpha.shape}")
    alpha = alpha / alpha.sum(axis=1, keepdims=True)
    entropy = -(alpha * np.log(alpha + 1e-12)).sum(axis=1)
    mean_alpha = alpha.mean(axis=0)
    top = int(np.argmax(mean_alpha))
    uniform_share = 1.0 / alpha.shape[1]
    return {"n_windows": int(alpha.shape[0]), "num_rois": int(alpha.shape[1]),
            "normalized_entropy_mean": float(entropy.mean() / np.log(alpha.shape[1])),
            "normalized_entropy_sd": float(entropy.std(ddof=1) / np.log(alpha.shape[1])),
            "normalized_entropy_of_mean_readout": float(-(mean_alpha * np.log(mean_alpha + 1e-12)).sum() / np.log(alpha.shape[1])),
            "top_roi_index": top, "top_roi_mean_weight": float(mean_alpha[top]),
            "uniform_share": uniform_share,
            "top_roi_share_ratio": float(mean_alpha[top] / uniform_share)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, help="run output directories, one per lambda")
    parser.add_argument("--lambdas", nargs="+", type=float, required=True, help="the lambda of each run, same order")
    parser.add_argument("--out", default="outputs/supplementary_analyses/lambda_sensitivity")
    args = parser.parse_args()
    if len(args.runs) != len(args.lambdas):
        parser.error("--runs and --lambdas must have the same length")

    rows = []
    for run_path, lam in zip(args.runs, args.lambdas):
        run = resolve_path(run_path)
        fold = read_fold_metrics(run / "evaluation")
        row = {"lambda": lam, **{f"{metric}_{part}": round(value, 4) for metric in METRICS
                                  for part, value in zip(("mean", "sd"), mean_sd(fold[metric]))},
               **{key: (round(value, 4) if isinstance(value, float) else value) for key, value in attention_entropy(run).items()}}
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("lambda").reset_index(drop=True)

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "lambda_sensitivity_table.csv", index=False)
    (out / "lambda_sensitivity_table.json").write_text(json.dumps(json.loads(table.to_json(orient="records")), indent=2) + "\n",
                                                        encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    x = np.arange(len(table))
    axes[0].plot(x, table["macro_f1_mean"], "o-", color="#C00000", label="Macro-F1")
    axes[0].plot(x, table["recovery_f1_mean"], "s-", color="#E6A817", label="Recovery-F1")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{value:g}" for value in table["lambda"]])
    axes[0].set_xlabel("AER regularization coefficient \u03bb")
    axes[0].set_ylabel("Score (mean over held-out animals)")
    low = float(min(table["macro_f1_mean"].min(), table["recovery_f1_mean"].min()))
    axes[0].set_ylim(min(0.8, low - 0.05), 1.02)
    axes[0].legend(fontsize=9, frameon=False)
    axes[0].grid(alpha=0.25)
    axes[0].set_title("Decoding performance", fontsize=10.5)
    axes[1].errorbar(x, table["normalized_entropy_mean"], yerr=table["normalized_entropy_sd"], fmt="o-", color="#1f77b4",
                     capsize=4, lw=1.2)
    axes[1].axhline(1.0, color="#555555", ls="--", lw=1.0)
    axes[1].text(x[-1], 1.0, "uniform", fontsize=8.5, color="#555555", ha="right", va="bottom")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{value:g}" for value in table["lambda"]])
    axes[1].set_xlabel("AER regularization coefficient \u03bb")
    axes[1].set_ylabel("Normalized attention entropy")
    axes[1].set_ylim(0.0, 1.08)
    axes[1].grid(alpha=0.25)
    axes[1].set_title("Readout entropy", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out / "lambda_sensitivity_figure.png", dpi=300)
    fig.savefig(out / "lambda_sensitivity_figure.pdf")

    pd.set_option("display.width", 250)
    print(table.to_string(index=False))
    print(f"\nsaved -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
