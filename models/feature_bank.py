from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def ensure_normalized(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return features / norms


def save_feature_bank(
    output_dir: str | Path,
    features: np.ndarray,
    metadata: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    features_path = root / "cloud_features.npy"
    metadata_path = root / "cloud_meta.json"
    config_path = root / "cloud_config.json"

    np.save(features_path, ensure_normalized(features))
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "features": str(features_path),
        "metadata": str(metadata_path),
        "config": str(config_path),
    }


def load_feature_bank(output_dir: str | Path) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    root = Path(output_dir)
    features = np.load(root / "cloud_features.npy").astype(np.float32)
    metadata = json.loads((root / "cloud_meta.json").read_text(encoding="utf-8"))
    config = json.loads((root / "cloud_config.json").read_text(encoding="utf-8"))
    return features, metadata, config
