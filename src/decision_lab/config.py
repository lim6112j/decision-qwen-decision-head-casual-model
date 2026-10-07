"""Configuration loading and validation."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class GeneratorConfig:
    num_train: int = 5000
    num_test_indist: int = 1000
    num_test_heldout: int = 1000
    seed: int = 42
    heldout_templates: tuple[str, ...] = ("report", "log_entry", "config_file")


@dataclass
class QuestionsConfig:
    """Fixed question bank.

    choice: {qid: {"options": {key: description}}}
    score:  {qid: {"levels": [label0, label1, ...]}}   # level i = rank i
    noul:   {qid: "boolean question text"}
    """

    choice: dict = field(default_factory=dict)
    score: dict = field(default_factory=dict)
    noul: dict = field(default_factory=dict)


@dataclass
class ModelConfig:
    gguf_path: str = "models/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"
    context_length: int = 2048
    server_port: int = 8080
    pooling: str = "last"


@dataclass
class HeadConfig:
    hidden_dim: int = 256
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    max_epochs: int = 100
    patience: int = 10


@dataclass
class CalibrationConfig:
    holdout_fraction: float = 0.1
    lr: float = 1e-2
    max_iter: int = 1000


@dataclass
class PromptLMConfig:
    temperature: float = 0.0
    max_tokens: int = 256
    system_prompt_zero_shot: str = ""
    system_prompt_cot: str = ""


@dataclass
class BenchmarkConfig:
    warmup_runs: int = 5
    timed_runs: int = 50
    max_test_states: int = 0        # 0 = use the full test set
    latency_sample_size: int = 50   # states used for warmup/timed latency runs


@dataclass
class WebConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    num_states: int = 8          # pre-generated states offered in the UI
    random_seed: int = 42


@dataclass
class Config:
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    questions: QuestionsConfig = field(default_factory=QuestionsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    prompt_lm: PromptLMConfig = field(default_factory=PromptLMConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    web: WebConfig = field(default_factory=WebConfig)


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    """Load and validate config from YAML."""
    raw = yaml.safe_load(Path(path).read_text()) or {}

    cfg = Config(
        generator=GeneratorConfig(**raw.get("generator", {})),
        questions=QuestionsConfig(**raw.get("questions", {})),
        model=ModelConfig(**raw.get("model", {})),
        head=HeadConfig(**raw.get("head", {})),
        calibration=CalibrationConfig(**raw.get("calibration", {})),
        prompt_lm=PromptLMConfig(**raw.get("prompt_lm", {})),
        benchmark=BenchmarkConfig(**raw.get("benchmark", {})),
        web=WebConfig(**raw.get("web", {})),
    )

    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    q = cfg.questions
    num_questions = len(q.choice) + len(q.score) + len(q.noul)
    if num_questions == 0:
        raise ValueError("question bank must contain at least one question")
    for qid, spec in q.choice.items():
        if len(spec.get("options", {})) < 2:
            raise ValueError(f"choice question '{qid}' needs >= 2 options")
    for qid, spec in q.score.items():
        if len(spec.get("levels", [])) < 2:
            raise ValueError(f"score question '{qid}' needs >= 2 rubric levels")
    if cfg.model.pooling not in ("last", "mean"):
        raise ValueError(f"pooling must be 'last' or 'mean', got {cfg.model.pooling}")
    if cfg.head.dropout < 0 or cfg.head.dropout > 0.9:
        raise ValueError(f"dropout {cfg.head.dropout} out of range")
    if not 0 < cfg.calibration.holdout_fraction < 0.5:
        raise ValueError(f"calibration.holdout_fraction {cfg.calibration.holdout_fraction} out of (0, 0.5)")