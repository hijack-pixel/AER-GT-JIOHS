# Fig.4 IOS State Analysis

This folder contains the numerical preprocessing entry for the manuscript Fig.4-style IOS state analysis.

It does not store generated figures. Running the entry below writes ignored intermediate metrics to `outputs/fig4_ios_state_analysis/`.

```matlab
addpath('preprocess/matlab')
addpath('preprocess/fig4')
cfg = aergt_preprocess_config_template();
cfg.raw.epilepsy_root = 'PATH_TO_PRIVATE_EPILEPSY_MOUSE_DATA';
run_fig4_ios_state_analysis(cfg)
```

The analysis uses the five epilepsy mice and one representative baseline, seizure, and post-ictal recovery recording per mouse. It reads the first 300 frames of each selected recording, normalizes each file by its first 100 frames, analyzes frames 101-300, and computes ROI traces, functional-connectivity matrices, and network-deviation metrics.

**Correction (reviewer response):** window length is 200 frames (20 s at 10 Hz) as described in the manuscript; functional connectivity is computed per animal and then averaged across animals; network deviation and all reported statistics are evaluated at the animal level (`compute_fig4_animal_level_stats.m`, Friedman omnibus + pairwise Wilcoxon signed-rank with Holm correction, Kendall's W).
