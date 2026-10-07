# MATLAB Preprocessing

This folder contains the private-data preprocessing code used to convert raw widefield IOS TIFF recordings into AER-GT graph samples.

The raw recordings are not included in this repository. Configure local private paths in `aergt_preprocess_config_template.m` before running the exporters.

## Main Entries

```matlab
addpath('preprocess/matlab')
cfg = aergt_preprocess_config_template();
cfg.raw.epilepsy_root = 'PATH_TO_PRIVATE_EPILEPSY_MOUSE_DATA';
cfg.raw.control_con1_root = 'PATH_TO_PRIVATE_CON1';
cfg.raw.control_con2_root = 'PATH_TO_PRIVATE_CON2';
run_full_preprocess(cfg)
```

You can also run the two stages separately:

```matlab
export_control_pretrain_samples(cfg)
export_labeled_aergt_samples(cfg)
```

## Outputs

The default outputs are under `data/private/`, which is excluded by `.gitignore`:

- `data/private/pretrain_con1_con2/`: unlabeled NB1+NB2 (normative baseline) saline-control samples for self-supervised pretraining.
- `data/private/labeled_epilepsy/`: labeled epilepsy samples from Mouse 1–Mouse 5.

Each output `.mat` file contains one `AERT_Sample` structure:

- `NodeFeats`: `22 x 6` matrix.
- `Adjacency`: `22 x 22` ROI functional-connectivity matrix.
- `Label`: `0` baseline, `1` seizure, `2` post-ictal recovery, `-1` unlabeled saline-control pretraining sample.
- `MouseID`: subject identifier used by subject-level cross-validation.

## Paper Parameters

- Sampling rate: 10 Hz.
- Sliding window: 20 s / 200 frames.
- Window step: 4 s / 40 frames.
- ROI box: 11 x 11 pixels.
- Node features: mean, standard deviation, skewness, kurtosis, bandpower 0.01-0.1 Hz, bandpower 0.1-0.5 Hz.
- ROI order is fixed by `aergt_default_roi_labels.m` and must not be changed without retraining.

## Quality Reports

Control quality PNGs are disabled by default:

```matlab
cfg.output.write_quality_reports = false;
```

If enabled, they are written to `outputs/preprocess_quality/`, which is not intended for version control.
