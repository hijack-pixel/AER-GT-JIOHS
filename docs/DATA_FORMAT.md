# Processed Data Format

The Python model expects one MATLAB `.mat` file per graph sample. Each file must contain a variable named `AERT_Sample`.

Required fields:

- `NodeFeats`: `22 x 6` numeric matrix. Rows are ROIs and columns are the six IOS features.
- `Adjacency`: `22 x 22` numeric functional-connectivity matrix.
- `Label`: integer state label. Use `0` for baseline, `1` for seizure, `2` for post-ictal recovery, and `-1` for unlabeled control/pretraining samples.
- `MouseID`: subject identifier. Subject-level cross-validation groups samples by this field.

Recommended fields:

- `OriginalFile`: original recording filename.
- `WindowIndex`: sliding-window index.
- `WindowStartFrame`: first frame in the sample window.
- `WindowEndFrame`: last frame in the sample window.
- `FPS`: imaging frame rate.
- `WinSec`: window duration in seconds.
- `StepSec`: window step in seconds.

Expected folder layout for the full paper pipeline:

```text
data/private/pretrain_con1_con2/*.mat
data/private/labeled_epilepsy/*.mat
```

Generate the full private-data processed folders from raw TIFF recordings:

```matlab
addpath('preprocess/matlab')
cfg = aergt_preprocess_config_template();
cfg.raw.epilepsy_root = 'PATH_TO_PRIVATE_EPILEPSY_MOUSE_DATA';
cfg.raw.control_con1_root = 'PATH_TO_PRIVATE_CON1';
cfg.raw.control_con2_root = 'PATH_TO_PRIVATE_CON2';
run_full_preprocess(cfg)
```

Validate processed folders before running the Python model:

```matlab
validate_aert_sample_folder('data/private/pretrain_con1_con2', false)
validate_aert_sample_folder('data/private/labeled_epilepsy', true)
```

The paper configuration assumes:

- `num_nodes = 22`
- `in_channels = 6`
- `class_names = [Baseline, Seizure, Recovery]`
- `num_folds = 5`
