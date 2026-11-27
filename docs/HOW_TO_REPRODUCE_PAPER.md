# Full Paper Reproduction

Every command is run from the repository root after `conda activate aergt` (or the pip installation). The quantitative results require the private dataset; the synthetic demo verifies the code path only.

## 1. Preprocess the raw recordings

Convert the raw TIFF recordings into graph samples with the MATLAB pipeline (`preprocess/matlab/README.md`). It writes

```text
data/private/pretrain_normative_baseline/
data/private/labeled_epilepsy/
data/private/control_pharmacological/
```

## 2. Evaluate the bundled paper-trained artifacts (no retraining)

```bash
python scripts/run_inference.py --config configs/paper_trained_inference.yaml
python scripts/run_control_inference.py --config configs/paper_trained_inference.yaml
```

## 3. Train and evaluate the full model

```bash
python scripts/train_full_model.py --config configs/full_model.yaml
python scripts/run_inference.py --config configs/full_model.yaml
```

`outputs/full_model/evaluation/fold_metrics.csv` holds the per-animal metrics behind Table 2 (row "Proposed model").

## 4. Baselines (Table 2)

```bash
python scripts/run_baselines.py --config configs/full_model.yaml --all
```

## 5. Ablations (Table 3, Sec. 3.2)

```bash
python scripts/run_ablation.py no_pretraining
python scripts/run_ablation.py no_aer_regularization
python scripts/run_ablation.py no_topology
python scripts/run_inference.py --config configs/ablation_no_pretraining.yaml
python scripts/run_inference.py --config configs/ablation_no_aer_regularization.yaml
python scripts/run_inference.py --config configs/ablation_no_topology.yaml
python scripts/compare_ablations.py
```

## 6. Pharmacological controls (Sec. 3.3, Fig. 6)

```bash
python scripts/run_control_inference.py --config configs/full_model.yaml
python scripts/reproduce_fig6.py --run-dir outputs/full_model
```

## 7. Recording quality and motion (Sec. 2.10, Sec. 3.5, Fig. 8)

```bash
python scripts/compute_motion_quality.py --config configs/full_model.yaml --recordings-dir <root with one TIFF folder per animal>
python scripts/reproduce_fig8.py --run-dir outputs/full_model --recording-times recording_times.csv
```

## 8. Recovery trajectory (Sec. 3.2, Fig. 5)

`recording_times.csv` lists the post-injection time of every recording (columns `mouse_id`, `recording`, `t_post_inj_min`).

```bash
python scripts/recovery_trajectory_stats.py --config configs/full_model.yaml --recording-times recording_times.csv
python scripts/reproduce_fig5.py --run-dir outputs/full_model
```

## 9. Interpretability (Sec. 2.8, Fig. 7)

```bash
python scripts/run_interpretability.py --config configs/full_model.yaml
python scripts/reproduce_fig7.py --run-dir outputs/full_model
```

## 10. State-wise characterization (Sec. 2.3, Fig. 4)

```matlab
addpath('preprocess/matlab')
addpath('preprocess/fig4')
run_fig4_ios_state_analysis(cfg)
```

## 11. Time-based negative control (Sec. 2.9, Sec. 3.3)

```bash
# baseline windows of the ictal-transition recordings (pseudo-class by post-injection time)
python scripts/supplementary_analyses/time_negative_control_decode.py --config configs/full_model.yaml --recording-times recording_times.csv
# windows of the normative-baseline (NB1, NB2) recordings (first and second half of each recording; one fold per batch)
python scripts/supplementary_analyses/time_negative_control_decode.py --config configs/full_model.yaml --cohort nb --fold-assignment nb_batches.csv --out outputs/time_negative_control_nb
python scripts/supplementary_analyses/time_negative_control_figures.py --config configs/full_model.yaml --run outputs/time_negative_control --nb-run outputs/time_negative_control_nb
```

`nb_batches.csv` lists the batch (`NB1` or `NB2`) of every normative-baseline animal (columns `mouse_id`, `fold`).

## 12. Supplementary analyses

The per-animal performance table (Table 4), the recording timeline behind Table 5, the seed and lambda reruns, the segment-position sensitivity analysis, the readout-masking concordance, and further supplementary analyses are described in `scripts/supplementary_analyses/README.md`.
