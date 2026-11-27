# Supplementary Analyses

Self-contained analysis scripts for the statistics and figures of the manuscript that go beyond the main training and evaluation pipeline. Each script documents its inputs, outputs, and usage.

| Script | Manuscript item | Analysis |
|---|---|---|
| `per_animal_performance.py` | Table 4; Sec. 3.2 | Per-animal performance table (six metrics), animal-level 95% t-intervals, and the decomposition of the Baseline-F1 SD into precision and recall |
| `paired_baseline_comparison.py` | Sec. 2.5, Sec. 3.2 | Paired fold-wise comparison of AER-GT with a baseline on identical folds: per-animal differences, mean paired difference with 95% confidence interval, sign test, Wilcoxon signed-rank test, exact sign-flip permutation test |
| `motion_proxy_analysis.py` | Sec. 3.5 | Motion index versus decoding outputs: Spearman correlations, motion of correctly and incorrectly decoded windows, baseline-versus-recovery accuracy by motion quintile |
| `segment_position_sensitivity.py` | Sec. 3.1 | Dependence of the Fig. 4 statistics on which part of each recording is analysed, evaluated as blocks of five consecutive windows (full recordings, random block positions, every block start position); `--block-windows 1` evaluates single 200-frame windows |
| `time_negative_control_decode.py`, `time_negative_control_figures.py` | Sec. 2.9, Sec. 3.3 | Early/late pseudo-class decoding with the identical pipeline (`train_from_config`), per-fold AUC, and an exact sign-flip test against chance, for the baseline windows of the ictal-transition recordings (pseudo-class by post-injection time of the recording) and, optionally, for the normative-baseline (NB saline) windows (first and second half of each recording; one fold per batch) |
| `control_decomposition_analysis.py` | Sec. 3.3 | Recovery assignments of the pharmacological controls by animal and post-injection recording; network deviation of four window groups with Mann-Whitney tests |
| `readout_masking_rank_correlation.py` | Sec. 3.4 | Spearman correlation between the readout ranking and the ROI-masking ranking over all ROIs, with a permutation null; fold-to-fold stability of both rankings |
| `lambda_entropy_table.py` with `configs/lambda_0p002.yaml`, `configs/lambda_0p2.yaml` (and the no-AER ablation as lambda = 0) | Sec. 2.6, Sec. 3.4 | Decoding performance, normalized attention entropy, and the top-ROI share of the readout as a function of the AER coefficient lambda |
| `seed_stability_analysis.py` with `configs/seed_123.yaml`, `configs/seed_2024.yaml` | Sec. 2.6 | Per-seed metrics, between-animal and between-seed variability |
| `state_timeline_figure.py` | Table 5 | Reference state and post-injection time of every training recording, with per-animal window counts |

## Notes

- **Data**: no animal recording data and no result tables or figures are included in this repository (see `docs/PRIVATE_DATA_NOTE.md` and `CODE_AND_DATA_AVAILABILITY.md`). The scripts consume the outputs of the main pipeline (`outputs/<run>/evaluation/`, `interpretability/`, `control_inference/`, `motion_quality/`) and write their results under `outputs/`.
- **Reruns**: the lambda and seed reruns use the standard training and inference entry points with a configuration derived from `configs/full_model.yaml` (`base_config`), so that only one setting differs from the main run.
- **Recording times**: analyses that need the post-injection time of a recording (`state_timeline_figure.py`, `time_negative_control_decode.py`, `control_decomposition_analysis.py`) read a CSV with the columns `mouse_id`, `recording`, `t_post_inj_min`; `recording` is the `OriginalFile` field of the window samples.
- **Batches of the normative-baseline cohort**: the NB variant of the time negative control (`--cohort nb`) reads a CSV with the columns `mouse_id` and `fold` that names the batch (NB1 or NB2) of every animal; each batch is held out as one fold.
- **Animal labels**: tables number the animals `Mouse 1`, `Mouse 2`, ... in ascending order of their identifiers in the dataset, the rule used by the manuscript tables. The identifiers of the private dataset are not part of the repository.
- **Paths**: all inputs and outputs are repository-relative and are declared in each script's usage text; no absolute paths are versioned.
