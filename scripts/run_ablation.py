from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aergt.config import load_config
from aergt.training.runner import train_from_config


ABLATION_CONFIGS = {
    "no_pretraining": ROOT / "configs" / "ablation_no_pretraining.yaml",
    "no_aer_regularization": ROOT / "configs" / "ablation_no_aer_regularization.yaml",
    "no_topology": ROOT / "configs" / "ablation_no_topology.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one AER-GT ablation model.")
    parser.add_argument("ablation", choices=sorted(ABLATION_CONFIGS))
    args = parser.parse_args()
    cfg = load_config(ABLATION_CONFIGS[args.ablation])
    train_from_config(cfg, repo_root=ROOT)


if __name__ == "__main__":
    main()

