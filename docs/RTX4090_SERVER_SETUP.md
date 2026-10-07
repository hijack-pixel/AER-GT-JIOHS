# RTX 4090 Server Setup

The model pipeline is configured for CUDA servers and works without hard-coded local paths.

Recommended environment:

- Python 3.10
- PyTorch 2.1 or newer
- CUDA-enabled PyTorch build matching the server driver
- One RTX 4090 or equivalent CUDA GPU

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate aergt
```

The default full-model config uses:

- `hardware.device: auto`
- `hardware.amp: true`
- `hardware.tf32: true`
- `hardware.pin_memory: true`
- `hardware.num_workers: 4`
- `training.pretrain_batch_size: 256`
- `training.finetune_batch_size: 128`

If the GPU runs out of memory, reduce the two batch sizes in `configs/full_model.yaml`.

Expected private data layout on the server:

```text
data/private/pretrain_con1_con2/*.mat
data/private/labeled_epilepsy/*.mat
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
