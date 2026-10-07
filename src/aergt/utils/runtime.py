from __future__ import annotations

import json
import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def configure_torch(tf32: bool = True) -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if tf32 else "highest")


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def cuda_summary() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    return {
        "available": True,
        "device_index": idx,
        "device_name": torch.cuda.get_device_name(idx),
        "capability": list(props.major_minor) if hasattr(props, "major_minor") else [props.major, props.minor],
        "total_memory_gb": round(props.total_memory / (1024**3), 3),
        "cuda_runtime": torch.version.cuda,
    }


def git_commit(root: str | Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return None


def runtime_manifest(root: str | Path, cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": cuda_summary(),
        "git_commit": git_commit(root),
        "working_directory": os.getcwd(),
        "config": cfg,
    }


def write_json(obj: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=True, indent=2)

