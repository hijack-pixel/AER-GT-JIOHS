# CUDA Server Setup

The model pipeline runs on CPU or CUDA devices.

Recommended environment:

- Python 3.10
- PyTorch 2.4 or newer
- CUDA-enabled PyTorch build matching the server driver
- One CUDA GPU with 24 GB of memory or more

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate aergt
```

The full-model configuration (`configs/full_model.yaml`) uses the settings of the manuscript:

- `hardware.device: auto`
- `hardware.amp: false` (single-precision training; no mixed precision)
- `hardware.tf32: false`
- `hardware.deterministic: true`
- `training.pretrain_batch_size: 64`
- `training.finetune_batch_size: 32`

Data layout on the server:

```text
data/private/pretrain_normative_baseline/*.mat
data/private/labeled_epilepsy/*.mat
data/private/control_pharmacological/*.mat
```

Run:

```bash
python scripts/train_full_model.py --config configs/full_model.yaml
python scripts/run_inference.py --config configs/full_model.yaml
```

To evaluate the included paper-trained artifacts without retraining:

```bash
python scripts/run_inference.py --config configs/paper_trained_inference.yaml
```
