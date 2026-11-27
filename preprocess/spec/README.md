# Specification Tables

The preprocessing reads the animals and recordings of a dataset from three CSV tables with a header row. The tables describe a particular dataset and are not part of the repository: create them for your data (for example under `data/private/spec/`, the default location) and point `cfg.spec.*` at them (see `preprocess/matlab/README.md`).

## `animals.csv`

One row per animal, with the columns:

- `cohort`: `labeled`, `pretrain`, or `control`.
- `animal`: animal identifier; it becomes the `MouseID` of the samples and the grouping key of the subject-level cross-validation.
- `folder`: folder with the animal's TIFF files, relative to the root of the cohort (or absolute).
- `landmark1_x`, `landmark1_y`, `landmark2_x`, `landmark2_y`: the animal's two registration landmarks in pixel coordinates; they are mapped onto the landmarks of the cortical template.

## `recordings.csv`

One row per TIFF file, with the columns:

- `cohort`: `labeled`, `pretrain`, or `control`.
- `animal`: animal identifier as in `animals.csv`.
- `recording`: recording identifier; files that belong to one recording (chunks) share it.
- `file`: TIFF file name inside the animal's folder.
- `order`: concatenation order of the chunks of a recording (1, 2, ...; 1 for single files).
- `label`: reference label of a labeled recording (`baseline`, `seizure`, or `recovery`); empty for the unlabeled cohorts.

For the `labeled` cohort every recording is a single file and the label is the session-level reference state (Sec. 2.1). Recordings of the `pretrain` and `control` cohorts are exported with `Label = -1`.

## `fig4_state_recordings.csv`

One row per animal with the baseline, seizure, and recovery recording used for Fig. 4. Columns: `animal`, `baseline_file`, `seizure_file`, `recovery_file`. The animals must be listed in `animals.csv` with cohort `labeled`.
