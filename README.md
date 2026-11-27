# AER-GT

Code for the manuscript "Explainable Graph Network Decoding of Post-Ictal Recovery from Intrinsic Optical Signals in a Mouse Epilepsy Model".

AER-GT (Attention-Entropy-Regularized Graph Transformer) decodes baseline, seizure, and post-ictal recovery states from widefield intrinsic optical signal (IOS) recordings of awake, head-fixed mice. Each 20-s window of a recording is represented as a graph of 22 cortical ROIs with six IOS features per ROI; the model combines topology-guided aggregation with a global self-attention encoder and an attention-entropy readout, and is trained in two stages (self-supervised node-masking pretraining on unlabeled normative-baseline recordings, then supervised fine-tuning) under subject-level five-fold cross-validation.

The private animal data are not included. The repository is organized so that the complete pipeline runs when the processed `.mat` graph samples are placed under `data/private/`.

## Manuscript-to-code map

| Manuscript item | Code |
|---|---|
| Sec. 2.1 Animals, recordings, and reference labels (Table 1, Table 5) | specification tables under `preprocess/spec/`; `scripts/supplementary_analyses/state_timeline_figure.py` |
| Sec. 2.2 ROI-wise graph construction (Fig. 2) | `preprocess/matlab/` |
| Sec. 2.3 State-wise characterization (Fig. 4) | `preprocess/fig4/` |
| Sec. 2.4-2.6 AER-GT, two-stage training, subject-level five-fold cross-validation | `src/aergt/`, `scripts/train_full_model.py`, `scripts/run_inference.py`, `configs/full_model.yaml` |
| Sec. 2.7 Feature-based and graph-based baselines (Table 2) | `src/aergt/models/baselines.py`, `scripts/run_baselines.py` |
| Sec. 2.8 Model-based interpretability (Fig. 7) | `scripts/run_interpretability.py`, `scripts/reproduce_fig7.py` |
| Sec. 2.9 Pharmacological-control evaluation (Sec. 3.3, Fig. 6) | `scripts/run_control_inference.py`, `scripts/reproduce_fig6.py` |
| Sec. 2.10 Motion and stability quantification (Sec. 3.5, Fig. 8) | `scripts/compute_motion_quality.py`, `scripts/reproduce_fig8.py` |
| Sec. 3.2 Recovery-trajectory quantification (Fig. 5) | `scripts/recovery_trajectory_stats.py`, `scripts/reproduce_fig5.py` |
| Table 3 Ablations and their statistics | `scripts/run_ablation.py`, `scripts/compare_ablations.py`, `src/aergt/evaluation/ablation_stats.py` |
| Table 4 Per-animal performance | `scripts/supplementary_analyses/per_animal_performance.py` |
| Supplementary analyses: per-animal statistics, sensitivity and stability analyses, control decomposition, figures | `scripts/supplementary_analyses/` |

## Repository layout

```text
configs/                    YAML configs for the full model, the demo, the ablations, and the lambda/seed reruns
src/aergt/                  Python package: data loading, model, training, evaluation, interpretability, figures
scripts/                    Command-line entry points
scripts/supplementary_analyses/  Supplementary analyses beyond the main pipeline
preprocess/matlab/          MATLAB raw TIFF-to-graph preprocessing
preprocess/fig4/            MATLAB analysis behind Fig. 4
preprocess/spec/            Placeholder tables of animals, recordings, and Fig. 4 recordings
data/private/               Expected location of the private processed samples; ignored by git
model_artifacts/paper_trained/  Checkpoints and scalers of the trained model
docs/                       Data format, reproduction steps, server setup, and release boundaries
outputs/                    Generated results; ignored by git
tests/                      Smoke tests that need no private data
```

## Install

```bash
conda env create -f environment.yml
conda activate aergt
```

or with pip after installing a CUDA-enabled PyTorch build where needed:

```bash
pip install -r requirements.txt
```

The bundled scaler files were written with NumPy 2; use the versions in the dependency files to load them. Server notes are in `docs/CUDA_SERVER_SETUP.md`.

## Tests and demo

```bash
python -m pytest tests
python scripts/smoke_test_demo.py
```

The demo creates synthetic, non-biological graph samples, runs a short training and inference pass, and writes `outputs/demo/`. It validates the software path only and cannot reproduce the performance of the manuscript. Remove its outputs with `python scripts/clean_generated_outputs.py`.

## Full pipeline

1. Preprocess the raw recordings with MATLAB (`preprocess/matlab/README.md`) or place authorized processed samples under `data/private/` (`docs/DATA_FORMAT.md`).
2. Train and evaluate:

```bash
python scripts/train_full_model.py --config configs/full_model.yaml
python scripts/run_inference.py --config configs/full_model.yaml
```

3. Baselines, ablations, controls, motion, trajectories, interpretability, and figures: see `docs/HOW_TO_REPRODUCE_PAPER.md` for the complete list of commands.

To evaluate the bundled paper-trained checkpoints and scalers without retraining:

```bash
python scripts/run_inference.py --config configs/paper_trained_inference.yaml
```

## Data availability

The raw and processed IOS datasets are private animal experimental data and are not committed to this repository. See `CODE_AND_DATA_AVAILABILITY.md`, `docs/DATA_FORMAT.md`, and `docs/PRIVATE_DATA_NOTE.md`. `docs/RELEASE_BOUNDARIES.md` lists what is and is not versioned.
