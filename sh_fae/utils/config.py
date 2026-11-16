"""Configuration definitions and YAML loading utilities for SH-FAE."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    latent_dim: int = 64
    point_embed_dim: int = 64
    encoder_layers: int = 3
    encoder_heads: int = 4
    encoder_mlp_ratio: int = 2
    dropout: float = 0.0
    decoder_hidden_dim: int = 128
    decoder_layers: int = 3
    density_eps: float = 1.0e-6
    x_min: float = -1.0
    x_max: float = 8.0
    x_int_points: int = 512
    u_int_points: int = 512
    y_min: float = -12.0


@dataclass
class BatchConfig:
    nc_swrc: int = 64
    nc_hcf: int = 64
    nq_swrc: int = 128
    nq_hcf: int = 128


@dataclass
class LossConfig:
    smooth_weight: float = 1.0e-5


@dataclass
class TrainingConfig:
    data_path: str = "data/swcc_hcf_50k.npz"
    output_dir: str = "outputs/ae_50k_lat64"
    batch_size: int = 256
    max_steps: int = 100_000
    lr_peak: float = 1.0e-3
    weight_decay: float = 1.0e-5
    grad_clip_norm: float = 1.0
    log_every: int = 100
    progress_every: int = 100
    eval_every: int = 1000
    save_every: int = 5000
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"


@dataclass
class AEConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _dataclass_from_dict(cls: type, values: dict[str, Any]) -> Any:
    names = {f.name for f in fields(cls)}
    filtered = {key: value for key, value in values.items() if key in names}
    return cls(**filtered)


def config_from_dict(values: dict[str, Any]) -> AEConfig:
    return AEConfig(
        model=_dataclass_from_dict(ModelConfig, values.get("model", {})),
        batch=_dataclass_from_dict(BatchConfig, values.get("batch", {})),
        loss=_dataclass_from_dict(LossConfig, values.get("loss", {})),
        training=_dataclass_from_dict(TrainingConfig, values.get("training", {})),
    )


def config_to_dict(config: AEConfig) -> dict[str, Any]:
    return asdict(config)


def load_config_yaml(path: str | Path) -> AEConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        values = yaml.safe_load(f) or {}
    if not isinstance(values, dict):
        raise ValueError(f"YAML config must contain a mapping: {path}")
    return config_from_dict(values)


def save_config_yaml(config: AEConfig, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config_to_dict(config), f, sort_keys=False)


def load_generic_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        values = yaml.safe_load(f) or {}
    if not isinstance(values, dict):
        raise ValueError(f"YAML config must contain a mapping: {path}")
    return values
