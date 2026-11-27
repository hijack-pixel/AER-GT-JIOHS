# Processed Data Format

The Python pipeline expects one MATLAB `.mat` file per graph sample. Each file contains a variable named `AERT_Sample`.

## Fields read by the pipeline

- `NodeFeats`: `22 x 6` numeric matrix. Rows are ROIs and columns are the six IOS features.
- `Adjacency`: `22 x 22` numeric functional-connectivity matrix (Pearson correlations; the loader takes absolute values when `data.use_abs_adjacency` is true).
- `Label`: state label. `0` for baseline, `1` for seizure, `2` for post-ictal recovery, and `-1` for unlabeled samples (normative-baseline and pharmacological-control cohorts).
- `MouseID`: subject identifier. Subject-level cross-validation groups the samples by this field.
- `OriginalFile`, `WindowIndex`: recording identifier and window index (1-based). They link the out-of-fold predictions to recordings and are used by every analysis that groups windows by recording.

## Additional fields written by the exporters

`WindowStartFrame`, `WindowEndFrame`, `FPS`, `WinSec`, `StepSec`, `ROIMap`, and `ROI_Names` document each window; they are not used for training.

## Folder layout

```text
data/private/pretrain_normative_baseline/*.mat   unlabeled normative-baseline (NB1, NB2) samples
data/private/labeled_epilepsy/*.mat               labeled samples of the ictal-transition recordings
data/private/control_pharmacological/*.mat        unlabeled samples of the pharmacological controls (Sec. 2.9)
```

## Generating the folders

The MATLAB exporters convert raw TIFF recordings into these samples. Animals, recordings, and labels are listed in three specification tables that you create for your dataset (columns in `preprocess/spec/README.md`):

```matlab
addpath('preprocess/matlab')
cfg = aergt_preprocess_config_template();
cfg.raw.epilepsy_root = 'PATH_TO_RAW_ICTAL_TRANSITION_RECORDINGS';
cfg.raw.pretrain_root  = 'PATH_TO_RAW_NORMATIVE_BASELINE_RECORDINGS';
cfg.raw.control_root   = 'PATH_TO_RAW_PHARMACOLOGICAL_CONTROL_RECORDINGS';
cfg.spec.animals = 'PATH_TO/animals.csv';
cfg.spec.recordings = 'PATH_TO/recordings.csv';
run_full_preprocess(cfg)
```

Validate the folders before running the Python pipeline:

```matlab
validate_aert_sample_folder('data/private/pretrain_normative_baseline', false)
validate_aert_sample_folder('data/private/control_pharmacological', false)
validate_aert_sample_folder('data/private/labeled_epilepsy', true)
```

## Cohort sizes

The cohort sizes and window counts are those listed in Table 1 of the manuscript; the exporters report the number of windows written for every recording, and `subject_metrics_for_stats.csv` lists the windows of each held-out animal.

## Configuration assumptions

- `num_nodes = 22`
- `in_channels = 6`
- `class_names = [Baseline, Seizure, Recovery]`
- `num_folds = 5`
