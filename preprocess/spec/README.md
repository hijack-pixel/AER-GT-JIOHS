# Specification Tables

The preprocessing reads animals and recordings from CSV tables instead of hard-coded lists. The tables shipped here contain placeholder rows only; copy them, fill them in for your data, and point `cfg.spec.*` at the copies (see `preprocess/matlab/README.md`). Animal identifiers, file names, and registration landmarks are data-specific and are not part of the repository.

## `animals.csv`

One row per animal.

| Column | Content |
|---|---|
| `cohort` | `labeled`, `pretrain`, or `control` |
| `animal` | Animal identifier; it becomes the `MouseID` of the samples and the grouping key of the subject-level cross-validation |
| `folder` | Folder with the animal's TIFF files, relative to the root of the cohort (or absolute) |
| `landmark1_x`, `landmark1_y`, `landmark2_x`, `landmark2_y` | The animal's two registration landmarks in pixel coordinates; they are mapped onto the landmarks of the cortical template |

## `recordings.csv`

One row per TIFF file.

| Column | Content |
|---|---|
| `cohort` | `labeled`, `pretrain`, or `control` |
| `animal` | Animal identifier as in `animals.csv` |
| `recording` | Recording identifier; files that belong to one recording (chunks) share it |
| `file` | TIFF file name inside the animal's folder |
| `order` | Concatenation order of the chunks of a recording (1, 2, ...; 1 for single files) |
| `label` | Reference label of a labeled recording: `baseline`, `seizure`, or `recovery`; empty for the unlabeled cohorts |

For the `labeled` cohort every recording is a single file and the label is the session-level reference state (Sec. 2.1). Recordings of the `pretrain` and `control` cohorts are exported with `Label = -1`.

## `fig4_state_recordings.csv`

One row per animal with the baseline, seizure, and recovery recording used for Fig. 4: columns `animal`, `baseline_file`, `seizure_file`, `recovery_file`. The animals must be listed in `animals.csv` with cohort `labeled`.
