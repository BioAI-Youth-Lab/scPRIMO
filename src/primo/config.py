from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, Union

from primo.data.sciplex_dataset import SciplexDatasetConfig
from primo.data.nips_dataset import NipsDatasetConfig
from primo.training_config import ModelConfig, TrainingConfig


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - clear error in runtime
        raise ImportError(
            "PyYAML is required to load configuration files. "
            "Please install it with `pip install pyyaml`."
        ) from exc

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level config structure must be a mapping, got {type(data)}")
    return data


def load_sciplex_config(config_path: str) -> SciplexDatasetConfig:
    """
    Load a SciplexDatasetConfig from a YAML file.
    """
    path = Path(config_path)
    raw = _load_yaml(path)
    dataset_cfg = raw.get("dataset", {})
    if not isinstance(dataset_cfg, dict):
        raise ValueError("`dataset` section in config must be a mapping.")
    return SciplexDatasetConfig(**dataset_cfg)


def load_full_sciplex_config(
    config_path: str,
) -> Tuple[SciplexDatasetConfig, ModelConfig, TrainingConfig, Dict[str, Any]]:
    """
    Load dataset, model, training configs and paths from a YAML file (sciplex).
    """
    path = Path(config_path)
    raw = _load_yaml(path)

    dataset_cfg = raw.get("dataset", {})
    if not isinstance(dataset_cfg, dict):
        raise ValueError("`dataset` section in config must be a mapping.")
    dataset_config = SciplexDatasetConfig(**dataset_cfg)

    model_cfg = raw.get("model", {})
    if not isinstance(model_cfg, dict):
        raise ValueError("`model` section in config must be a mapping.")
    model_config = ModelConfig(**model_cfg)

    training_cfg = raw.get("training", {})
    if not isinstance(training_cfg, dict):
        raise ValueError("`training` section in config must be a mapping.")
    training_config = TrainingConfig(**training_cfg)

    paths = raw.get("paths", {})
    if not isinstance(paths, dict):
        raise ValueError("`paths` section in config must be a mapping.")

    return dataset_config, model_config, training_config, paths


def load_full_nips_config(
    config_path: str,
) -> Tuple[NipsDatasetConfig, ModelConfig, TrainingConfig, Dict[str, Any]]:
    """
    Load dataset, model, training configs and paths from a YAML file (nips).
    """
    path = Path(config_path)
    raw = _load_yaml(path)

    dataset_cfg = raw.get("dataset", {})
    if not isinstance(dataset_cfg, dict):
        raise ValueError("`dataset` section in config must be a mapping.")
    dataset_config = NipsDatasetConfig(**dataset_cfg)

    model_cfg = raw.get("model", {})
    if not isinstance(model_cfg, dict):
        raise ValueError("`model` section in config must be a mapping.")
    model_config = ModelConfig(**model_cfg)

    training_cfg = raw.get("training", {})
    if not isinstance(training_cfg, dict):
        raise ValueError("`training` section in config must be a mapping.")
    training_config = TrainingConfig(**training_cfg)

    paths = raw.get("paths", {})
    if not isinstance(paths, dict):
        raise ValueError("`paths` section in config must be a mapping.")

    return dataset_config, model_config, training_config, paths


