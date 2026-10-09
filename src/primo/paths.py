from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
OUTPUT_DIR = REPO_ROOT / "outputs"
ARTIFACT_DIR = REPO_ROOT / "artifacts"
CONFIG_DIR = REPO_ROOT / "configs"

