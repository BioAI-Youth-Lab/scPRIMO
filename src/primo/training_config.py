from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    n_epochs: int = 100
    conditioning_dim: int = 200
    embedding_dim: int = 10
    conditioning_depth: int = 2
    n_flows: int = 20
    in_channels: int = 10
    hidden_dim: int = 1024
    hidden_depth: int = 2
    activation: str = "none"
    conditioner_use_bn: bool = True
    
    use_enhanced_cov: bool = False
    use_llm_similars: bool = False
    use_chemberta: bool = False
    
    similars_map_path: Optional[str] = None
    chemberta_global_path: Optional[str] = None
    chemberta_alpha: float = 2.0
    chemberta_pca_n: Optional[int] = None


@dataclass
class TrainingConfig:
    n_epochs: int = 10
    batch_size: int = 256
    lr: float = 9e-6
    train_ratio: float = 0.8
    seed: int = 42
    start_epoch: int = 1
    auto_save: int = 0
    device: Optional[str] = None
    use_scheduler: bool = True
    scheduler_type: str = "cosine"
    weight_decay: float = 0.0
    centroid_lambda: float = 0.0


