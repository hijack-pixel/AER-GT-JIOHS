# Full Paper Reproduction

1. Generate processed graph samples matching `docs/DATA_FORMAT.md`.

```matlab
addpath('preprocess/matlab')
cfg = aergt_preprocess_config_template();
cfg.raw.epilepsy_root = 'PATH_TO_PRIVATE_EPILEPSY_MOUSE_DATA';
cfg.raw.control_con1_root = 'PATH_TO_PRIVATE_CON1';
cfg.raw.control_con2_root = 'PATH_TO_PRIVATE_CON2';
run_full_preprocess(cfg)
```

2. Confirm that NB1+NB2 (normative baseline) pretraining samples are in `data/private/pretrain_con1_con2/`.
3. Confirm that labeled epilepsy samples for Mouse 1–Mouse 5 are in `data/private/labeled_epilepsy/`.
4. Train the full model:

```bash
python scripts/train_full_model.py --config configs/full_model.yaml
```

5. Evaluate trained folds:

```bash
python scripts/run_inference.py --config configs/full_model.yaml
```

6. Alternatively, evaluate the included paper-trained fold artifacts:

```bash
python scripts/run_inference.py --config configs/paper_trained_inference.yaml
```

7. Train ablations:

```bash
python scripts/run_ablation.py no_pretraining
python scripts/run_ablation.py no_aer_regularization
python scripts/run_ablation.py no_topology
```

The full quantitative results require the private dataset. The synthetic demo verifies the code path but cannot reproduce the manuscript metrics.

8. Run the Table 2 baselines (Sec. 2.7):

```bash
python scripts/run_baselines.py --config configs/full_model.yaml --all
```

9. Pharmacological-control cohort decoding (Sec. 2.9), with the bundled paper-trained artifacts:

```bash
python scripts/run_control_inference.py --config configs/paper_trained_inference.yaml
```

10. Motion and recording-quality metrics (Sec. 2.10), from the authorized raw TIFF recordings:

```bash
python scripts/compute_motion_quality.py --recordings-dir <per-animal TIFF root>
```

11. Recovery-trajectory statistics (Sec. 3.2 / Fig. 5):

```bash
python scripts/recovery_trajectory_stats.py
```

12. Interpretability analyses (Sec. 2.8 / Fig. 7):

```bash
python scripts/run_interpretability.py --config configs/full_model.yaml
```
