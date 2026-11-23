# Revision Analyses

Analysis scripts added during the Major Revision of JIOHS-D-26-00096.
Each script is self-documented and maps to the reviewer comment it answers.

| Script | Reviewer comment | Analysis |
|---|---|---|
| `per_animal_performance.py` | R2.2(a) / R3 minor 5 | Per-held-out-animal performance table (six metrics), animal-level 95% t-CIs, Baseline-F1 SD decomposition |
| `motion_proxy_analysis.py` | R2.3 / R3.7 | Motion-index vs decoding outputs: Spearman correlations, error-window motion decomposition, motion-quintile-stratified and quiescent-only accuracy |
| `segment_position_sensitivity.py` | R3.2 | Segment-position sensitivity of the Fig. 4 statistics (all 29 positions + 1,000 random draws) |
| `time_negative_control_figures.py` | R3.4 | Figures for the early/late pseudo-class time negative control (per-fold AUC vs chance) |
| `readout_masking_rank_correlation.py` | R3.5(a) | Readout-ranking vs ROI-masking-drop ranking across all 22 ROIs, Spearman with permutation null |
| `lambda_train_0p002.py`, `lambda_infer_0p002.py`, `lambda_train_0p2.py`, `lambda_infer_0p2.py`, `lambda_entropy_table.py` | R3.5(b) | Lambda-sensitivity reruns (lambda = 0.002, 0.2) and the attention-entropy table |
| `seed_train_123.py`, `seed_infer_123.py`, `seed_train_2024.py`, `seed_infer_2024.py`, `seed_stability_analysis.py` | R3.8 | Seed-stability reruns (seeds 123, 2024) and the between-animal vs between-seed variability analysis |
| `state_timeline_figure.py` | R2 minor 1 / R3.9(b) | Per-animal reference-state timeline of the 45 training recordings (session-level labels) |

## Notes

- **Data**: no animal recording data is included in this repository
  (see `docs/PRIVATE_DATA_NOTE.md` and `CODE_AND_DATA_AVAILABILITY.md`).
  The scripts consume the processed window-level outputs of the main
  pipeline (`train_full_model.py` / `run_inference.py`) and aggregate
  result CSVs produced by the corresponding runs.
- **Paths**: scripts were executed in the authors' analysis environment and
  reference absolute paths of that environment (documented in each header).
  Adjust the input roots to your local layout to reproduce the analyses.
- **Time negative control**: the early/late pseudo-class decoding itself uses
  the identical two-stage pipeline with early/late baseline pseudo-labels
  (revised manuscript Sec. 2.9); `time_negative_control_figures.py` renders
  the per-fold AUC summary of that run.
