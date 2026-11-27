# Fig. 4: State-wise Characterization

This folder contains the numerical analysis behind Fig. 4 (Sec. 2.3): state-wise IOS signal features and inter-ROI network organization, with the animal-level statistics.

It stores no generated figures. Running the entry below writes the metrics to `outputs/fig4_ios_state_analysis/`.

```matlab
addpath('preprocess/matlab')
addpath('preprocess/fig4')
cfg = aergt_preprocess_config_template();
% set cfg.raw.epilepsy_root and cfg.spec.* as described in preprocess/matlab/README.md
run_fig4_ios_state_analysis(cfg)
```

## Analysis

- Animals, folders, and registration landmarks come from `cfg.spec.animals`; the three analyzed recordings of each animal from `cfg.spec.fig4_state_recordings` (one baseline-, one seizure-, and one recovery-labelled recording).
- Each recording is read for its first 300 frames and normalized against the first 100 frames. The analysis segment is the following 200 frames (frames 101-300, 20 s at 10 Hz) with a three-point median filter.
- Signal amplitude: mean normalized intensity over the 22 ROIs.
- Connectivity: Pearson correlation between the ROI time series of the segment, computed within each animal.
- Network deviation: `1 - r` between the upper-triangle connectivity profiles of a state segment and the animal's baseline segment.

With one segment per state, each animal contributes one observation per state, so all statistics are evaluated at the animal level (`compute_fig4_animal_level_stats.m`): Friedman omnibus tests with Kendall's W, followed by pairwise Wilcoxon signed-rank tests with Holm correction across the three state pairs.

The segment-position sensitivity analysis of these statistics is provided by `scripts/supplementary_analyses/segment_position_sensitivity.py`.
