# Demo Workflow

The demo creates synthetic non-biological `.mat` files and runs a short training/evaluation pass.

```bash
python scripts/smoke_test_demo.py
```

The demo uses `configs/demo.yaml`, which reduces epochs and folds so the pipeline runs quickly. Its outputs are written to:

```text
outputs/demo/
```

The demo verifies software correctness only. It is not intended to reproduce paper-level performance.

