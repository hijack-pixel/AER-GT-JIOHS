from __future__ import annotations

from pathlib import Path

import joblib


def save_standardizer(scaler, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)


def load_standardizer(path: str | Path):
    path = Path(path)
    try:
        return joblib.load(path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load scaler artifact: {path}. "
            "Use the repository environment with joblib>=1.4 and scikit-learn>=1.3, "
            "or regenerate scalers by rerunning training with the current environment."
        ) from exc

