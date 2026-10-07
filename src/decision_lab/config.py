"""Configuration loading and validation."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class GridConfig:
    size: int = 7
    wall_density: float = 0.15
    min_path_length: int = 5
    num_train_layouts: int = 40
    num_indist_test_layouts: int = 10
    num_heldout_test_layouts: int = 10
    heldout_size_min: int = 8
    heldout_size_max: int = 10
    heldout_wall_extra: float = 0.05


@dataclass
class ModelConfig:
    gguf_path: str = "models/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"
    context_length: int = 2048
    server_port: int = 8080
    pooling: str = "last"


@dataclass
class HeadConfig:
    hidden_dim: int = 256
    num_actions: int = 5
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    max_epochs: int = 100
    patience: int = 10


@dataclass
class PromptLMConfig:
    temperature: float = 0.0
    max_tokens: int = 128
    system_prompt_zero_shot: str = ""
    system_prompt_cot: str = ""


@dataclass
class BenchmarkConfig:
    warmup_runs: int = 5
    timed_runs: int = 50


@dataclass
class WebConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    max_steps: int = 50
    num_layouts: int = 5
    random_seed: int = 42


@dataclass
class Config:
    grid: GridConfig = field(default_factory=GridConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    prompt_lm: PromptLMConfig = field(default_factory=PromptLMConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    web: WebConfig = field(default_factory=WebConfig)


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    """Load and validate config from YAML."""
    raw = yaml.safe_load(Path(path).read_text()) or {}

    cfg = Config(
        grid=GridConfig(**raw.get("grid", {})),
        model=ModelConfig(**raw.get("model", {})),
        head=HeadConfig(**raw.get("head", {})),
        prompt_lm=PromptLMConfig(**raw.get("prompt_lm", {})),
        benchmark=BenchmarkConfig(**raw.get("benchmark", {})),
        web=WebConfig(**raw.get("web", {})),
    )

    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    if cfg.grid.wall_density < 0 or cfg.grid.wall_density > 0.4:
        raise ValueError(f"wall_density {cfg.grid.wall_density} out of [0, 0.4]")
    if cfg.model.pooling not in ("last", "mean"):
        raise ValueError(f"pooling must be 'last' or 'mean', got {cfg.model.pooling}")
    if cfg.head.num_actions < 2:
        raise ValueError("num_actions must be >= 2")
    if cfg.head.dropout < 0 or cfg.head.dropout > 0.9:
        raise ValueError(f"dropout {cfg.head.dropout} out of range")