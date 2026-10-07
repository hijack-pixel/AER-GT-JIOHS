from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)


def main() -> None:
    py = sys.executable
    run([py, "scripts/create_synthetic_demo_data.py", "--overwrite"])
    run([py, "scripts/train_full_model.py", "--config", "configs/demo.yaml"])
    run([py, "scripts/run_inference.py", "--config", "configs/demo.yaml"])


if __name__ == "__main__":
    main()

