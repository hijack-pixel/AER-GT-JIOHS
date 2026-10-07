from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aergt.config import load_config
from aergt.evaluation.inference import evaluate_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run subject-level inference for trained AER-GT folds.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "full_model.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    evaluate_from_config(cfg, repo_root=ROOT)


if __name__ == "__main__":
    main()

