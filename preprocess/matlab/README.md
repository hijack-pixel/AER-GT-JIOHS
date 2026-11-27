# MATLAB Preprocessing

This folder converts raw widefield IOS TIFF recordings into the AER-GT graph samples of Sec. 2.2 (ROI-wise graph construction). The raw recordings are not part of the repository.

## Specification tables

Animals and recordings are listed in three CSV tables that describe the dataset (column definitions in [spec/README.md](../spec/README.md)). Create them for your data, for example under `data/private/spec/`:

| Table | Columns | Content |
|---|---|---|
| `animals.csv` | `cohort, animal, folder, landmark1_x, landmark1_y, landmark2_x, landmark2_y` | One row per animal: its folder inside the cohort root and its two registration landmarks (pixel coordinates) |
| `recordings.csv` | `cohort, animal, recording, file, order, label` | One row per TIFF file; labeled recordings carry the session-level reference label (`baseline`, `seizure`, `recovery`), chunks of one unlabeled recording share `recording` and are concatenated in `order` |
| `fig4_state_recordings.csv` | `animal, baseline_file, seizure_file, recovery_file` | The baseline, seizure, and recovery recording of each animal used for Fig. 4 |

`cohort` is `labeled` (ictal-transition recordings), `pretrain` (normative-baseline cohort: the NB1 and NB2 saline recordings), or `control` (pharmacological controls). See [spec/README.md](../spec/README.md) for details.

## Run

Create the tables, set the raw-data roots, and run:

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

The three stages can also be run separately: `export_pretrain_samples(cfg)`, `export_control_samples(cfg)`, and `export_labeled_aergt_samples(cfg)`.

Required toolboxes: Image Processing (`imfinfo`, `imread`, `cp2tform`), Signal Processing (`bandpower`; `medfilt1` and `pwelch` for Fig. 4 and the optional quality reports), and Statistics and Machine Learning (`skewness`, `kurtosis`, `corr`; `friedman`, `signrank`, `tiedrank` for Fig. 4).

## Outputs

The default outputs are under `data/private/`, which is excluded by `.gitignore`:

- `data/private/pretrain_normative_baseline/`: unlabeled samples of the normative-baseline cohort (`Label = -1`), used for self-supervised pretraining and for the feature standardization.
- `data/private/control_pharmacological/`: unlabeled samples of the pharmacological controls (`Label = -1`), used only for evaluation (Sec. 2.9).
- `data/private/labeled_epilepsy/`: labeled samples of the ictal-transition recordings (`Label = 0` baseline, `1` seizure, `2` post-ictal recovery).

Each output `.mat` file contains one `AERT_Sample` structure (`NodeFeats`, `Adjacency`, `Label`, `MouseID`, `OriginalFile`, `WindowIndex`, and frame information; see `docs/DATA_FORMAT.md`). The `MouseID` is the animal identifier of the table and is the grouping key of the subject-level cross-validation.

## Processing steps

1. Registration: a similarity transform maps the two landmarks of the common cortical template onto the animal's landmarks and places the 22 ROI centers (11 x 11 pixel boxes) in the recording. The unlabeled cohorts use the equivalent affine transform (`aergt_map_rois_affine.m`).
2. ROI time series: mean intensity of each box per frame.
3. Normalization: percent intensity change against the mean of the first 100 frames of the recording.
4. Windows: 20 s (200 frames at 10 Hz) with a 4-s stride (40 frames); a recording of N frames yields `floor((N - 200) / 40) + 1` windows, each inheriting the reference label of its recording.
5. Node features per ROI and window: mean, standard deviation, skewness, kurtosis, bandpower 0.01-0.1 Hz, bandpower 0.1-0.5 Hz.
6. Adjacency: Pearson correlation between the 22 ROI time series of the window (absolute value is taken by the Python loader).

ROI order is fixed by `aergt_default_roi_labels.m` and must not be changed without retraining.

## Quality reports

Optional per-animal quality figures are disabled by default (`cfg.output.write_quality_reports = false`). When enabled they are written to `outputs/preprocess_quality/`, which is ignored by git.
