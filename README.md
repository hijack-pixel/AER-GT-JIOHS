# AER-GT Reproducibility Package

This repository contains code for the manuscript on explainable graph network decoding of widefield cortical intrinsic optical signals (IOS) for baseline, seizure, and post-ictal recovery states.

The private animal IOS data are not included. The repository is structured so that the full paper pipeline can run when the private `.mat` graph samples are placed under `data/private/`, while synthetic demo data can be generated to verify the code path.

## What This Code Reproduces

Manuscript-section-to-code mapping (revised manuscript):

| Manuscript item | Code |
|---|---|
| Sec. 2.2 preprocessing (registration, 22-ROI graphs, 6-dim features, adjacency) | `preprocess/matlab/` |
| Sec. 2.3 state-wise characterization + animal-level Friedman/Wilcoxon-Holm statistics | `preprocess/fig4/` |
| Sec. 2.4-2.6 AER-GT model, two-stage training, subject-level 5-fold CV | `src/aergt/`, `scripts/train_full_model.py`, `scripts/run_inference.py` |
| Sec. 2.7 feature-based and graph-based baselines (Table 2) | `src/aergt/models/baselines.py`, `scripts/run_baselines.py` |
| Sec. 2.8 interpretability (readout weights, ROI masking, contrasts, interaction map) | `src/aergt/interpretability/`, `scripts/run_interpretability.py` |
| Sec. 2.9 pharmacological-control decoding (per-fold scalers, mean-softmax ensemble) | `src/aergt/evaluation/control_inference.py`, `scripts/run_control_inference.py` |
| Sec. 2.10 motion and stability quantification (frame-difference index, consecutive-frame correlation, error-motion join) | `scripts/compute_motion_quality.py` |
| Sec. 3.2 recovery-trajectory quantification (within-animal Spearman rho, sign test) | `scripts/recovery_trajectory_stats.py` |
| Fig. 5 probability-trajectory figure | `scripts/reproduce_fig5.py` |
| Fig. 7 interpretability figure | `scripts/reproduce_fig7.py` |
| Table 3 ablations + Holm-corrected statistics | `scripts/run_ablation.py`, `scripts/compare_ablations.py`, `src/aergt/evaluation/ablation_stats.py` |

- Three-state IOS decoding: baseline, seizure, and post-ictal recovery.
- ROI-level graph samples with 22 cortical ROIs and six node features per ROI.
- Subject-level 5-fold cross-validation across Mouse 1–Mouse 5.
- AER-GT with topology-guided aggregation, global Transformer encoding, attention readout, self-supervised pretraining, and AER regularization.
- Ablations without pretraining, without AER regularization, and without topology-guided aggregation.
- Baseline models (Logistic Regression, XGBoost, MLP; supervised GCN/GAT; self-supervised GAT/BNT) under the same folds.
- Pharmacological-control cohort decoding, motion/recording-quality metrics, recovery-trajectory statistics, and the Sec. 2.8 interpretability pipeline.

## Repository Layout

```text
configs/                  YAML configs for full model, demo, and ablations
src/aergt/                Python package for data loading, model, training, and evaluation
scripts/                  Command-line entry points
preprocess/matlab/        MATLAB raw TIFF-to-graph preprocessing pipeline
preprocess/fig4/          MATLAB Fig.4 IOS state-analysis metric pipeline
data/demo/                Synthetic demo data location
data/private/             Expected location for private paper data; ignored by git
model_artifacts/          Paper-trained checkpoints and scalers (no result tables; run the code to regenerate)
docs/                     Data format, journal compliance, and reproducibility notes
outputs/                  Generated training/evaluation outputs; ignored by git
tests/                    Lightweight smoke tests for configs, data loading, and model forward pass
```

## What Each Directory Does

- `configs/`: Stores all run configurations. `demo.yaml` runs the synthetic minimum demo, `full_model.yaml` runs the private-data paper pipeline, `paper_trained_inference.yaml` evaluates bundled paper-trained artifacts, and `ablation_*.yaml` controls the ablation settings.
- `src/aergt/`: Core Python package. It contains `.mat` graph-sample loading, subject-level folds, AER-GT model definition, training loops, inference, metrics, and simple figure helpers.
- `scripts/`: User-facing command-line entry points. Use these instead of importing internal modules directly.
- `preprocess/matlab/`: MATLAB raw TIFF-to-graph preprocessing pipeline. With authorized private raw IOS recordings, it exports `AERT_Sample` `.mat` files for `data/private/pretrain_con1_con2/` and `data/private/labeled_epilepsy/`.
- `preprocess/fig4/`: MATLAB code for Fig.4-style IOS state-analysis metrics. It writes numerical outputs under `outputs/` and does not version generated figures.
- `data/`: Data placeholder. `data/demo/` is created by the synthetic demo generator and is ignored after cleanup. `data/private/` is where authorized private processed samples should be placed and is ignored by git.
- `model_artifacts/paper_trained/`: Bundled fold-wise trained checkpoints, pretrained checkpoints, and scalers. These are intentionally versioned so private-data holders can run inference without retraining. No numerical result tables are bundled; all metrics are regenerated under `outputs/` by running the code.
- `docs/`: Supporting documentation for data format, full reproduction, private-data policy, RTX 4090 server setup, and release boundaries.
- `outputs/`: Runtime outputs from training, inference, ablation comparison, generated figures, and preprocessing quality reports. Everything here is ignored except `outputs/.gitkeep`.
- `tests/`: Lightweight tests that do not require private data.

## Minimal Demo Path

The minimal demo does not require private animal data. It creates synthetic graph samples, runs short training, and runs inference:

```bash
python scripts/smoke_test_demo.py
```

This should create:

```text
data/demo/pretrain_con1_con2/
data/demo/labeled_epilepsy/
outputs/demo/
```

These files are generated artifacts and should not be committed. Clean them with:

```bash
python scripts/clean_generated_outputs.py
```

The demo verifies the software path only. It cannot reproduce manuscript-level performance.

## Full Private-Data Path

For authorized users with the private IOS recordings, first run the MATLAB preprocessing pipeline:

```matlab
addpath('preprocess/matlab')
cfg = aergt_preprocess_config_template();
cfg.raw.epilepsy_root = 'PATH_TO_PRIVATE_EPILEPSY_MOUSE_DATA';
cfg.raw.control_con1_root = 'PATH_TO_PRIVATE_CON1';
cfg.raw.control_con2_root = 'PATH_TO_PRIVATE_CON2';
run_full_preprocess(cfg)
```

This produces:

```text
data/private/pretrain_con1_con2/
data/private/labeled_epilepsy/
```

Then run the full Python training and inference pipeline using `configs/full_model.yaml`.

## Install

For a CUDA server with an RTX 4090, install a CUDA-enabled PyTorch build. One conda route is:

```bash
conda env create -f environment.yml
conda activate aergt
```

Or with pip after installing the correct PyTorch wheel for the server:

```bash
pip install -r requirements.txt
```

See `docs/RTX4090_SERVER_SETUP.md` for server-specific notes.

Optional development/test tools can be installed with:

```bash
pip install -e ".[dev]"
```

## Run Tests

Run lightweight tests:

```bash
python -m pytest tests
```

## Run the Full Paper Pipeline

After generating private graph samples with the MATLAB preprocessing pipeline, or after placing existing authorized processed samples, confirm the folders are:

```text
data/private/pretrain_con1_con2/
data/private/labeled_epilepsy/
```

Then run:

```bash
python scripts/train_full_model.py --config configs/full_model.yaml
python scripts/run_inference.py --config configs/full_model.yaml
```

To evaluate the included paper-trained artifacts without retraining:

```bash
python scripts/run_inference.py --config configs/paper_trained_inference.yaml
```

Run ablations:

```bash
python scripts/run_ablation.py no_pretraining
python scripts/run_ablation.py no_aer_regularization
python scripts/run_ablation.py no_topology
```

After training and evaluating all runs, compare ablations:

```bash
python scripts/compare_ablations.py
```

Generate simple figure outputs from evaluation files:

```bash
python scripts/reproduce_fig5.py --run-dir outputs/full_model
python scripts/reproduce_fig7.py --run-dir outputs/full_model
```

## Data Availability

The complete raw and processed IOS datasets are private animal experimental data and are not committed to this repository. See `CODE_AND_DATA_AVAILABILITY.md`, `docs/DATA_FORMAT.md`, and `docs/PRIVATE_DATA_NOTE.md`.

Generated outputs are not versioned. This includes figures, intermediate `.npz` files, out-of-fold prediction tables, and training outputs under `outputs/`.

See `docs/RELEASE_BOUNDARIES.md` for the exact inclusion/exclusion policy.
