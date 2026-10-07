from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def remove_inside_root(path: Path) -> None:
    path = path.resolve()
    root = ROOT.resolve()
    if not str(path).startswith(str(root)):
        raise RuntimeError(f"Refusing to remove outside repository root: {path}")
    if path.exists():
        shutil.rmtree(path)
        print(f"Removed {path.relative_to(root)}")


def main() -> None:
    remove_inside_root(ROOT / "outputs" / "demo")
    remove_inside_root(ROOT / "outputs" / "full_model")
    remove_inside_root(ROOT / "outputs" / "paper_trained_inference")
    remove_inside_root(ROOT / "outputs" / "ablation_no_pretraining")
    remove_inside_root(ROOT / "outputs" / "ablation_no_aer_regularization")
    remove_inside_root(ROOT / "outputs" / "ablation_no_topology")
    remove_inside_root(ROOT / "outputs" / "ablation_statistics")
    remove_inside_root(ROOT / "data" / "demo" / "labeled_epilepsy")
    remove_inside_root(ROOT / "data" / "demo" / "pretrain_con1_con2")


if __name__ == "__main__":
    main()

