"""Tests for config loading/validation, focused on the backbone settings.

The pooling knob was validated but never reached the server, and the
backbone/embed_dim identity fields are what feature-cache fingerprints and
checkpoints hang off — so their defaults and validation are load-bearing.
"""

import pytest

from decision_lab.config import BenchmarkConfig, Config, ModelConfig, load_config

MINIMAL = """
generator: {}
questions:
  choice:
    sentiment:
      options:
        positive: "Positive sentiment"
        negative: "Negative sentiment"
model: {}
head: {}
dynamic_head: {}
calibration: {}
prompt_lm: {}
benchmark: {}
web: {}
real: {}
"""


def _write(tmp_path, extra_model: str = "") -> str:
    body = MINIMAL.replace("model: {}", f"model:\n{extra_model}") \
        if extra_model else MINIMAL
    path = tmp_path / "cfg.yaml"
    path.write_text(body)
    return str(path)


class TestModelConfigDefaults:
    def test_backbone_defaults_to_qwen(self):
        m = ModelConfig()
        assert m.backbone == "qwen3.5-0.8b"
        assert m.embed_dim == 1024
        assert m.pooling == "last"

    def test_benchmark_skips_nothing_by_default(self):
        assert BenchmarkConfig().skip_agents == []


class TestPoolingValidation:
    @pytest.mark.parametrize("pooling", ["last", "mean", "cls"])
    def test_accepts_known_pooling(self, tmp_path, pooling):
        cfg = load_config(_write(tmp_path, f"  pooling: {pooling}"))
        assert cfg.model.pooling == pooling

    def test_rejects_unknown_pooling(self, tmp_path):
        with pytest.raises(ValueError, match="pooling must be"):
            load_config(_write(tmp_path, "  pooling: max"))

    def test_rejects_nonpositive_embed_dim(self, tmp_path):
        with pytest.raises(ValueError, match="embed_dim must be positive"):
            load_config(_write(tmp_path, "  embed_dim: 0"))


class TestBackboneConfig:
    def test_loads_backbone_identity(self, tmp_path):
        cfg = load_config(_write(
            tmp_path,
            "  backbone: bge-m3\n  pooling: cls\n  embed_dim: 1024",
        ))
        assert cfg.model.backbone == "bge-m3"
        assert cfg.model.pooling == "cls"

    def test_loads_skip_agents(self, tmp_path):
        body = MINIMAL.replace(
            "benchmark: {}",
            "benchmark:\n  skip_agents: [head_trained, prompt_lm_zero_shot]",
        )
        path = tmp_path / "cfg.yaml"
        path.write_text(body)
        cfg = load_config(str(path))
        assert cfg.benchmark.skip_agents == ["head_trained", "prompt_lm_zero_shot"]


class TestShippedConfigs:
    def test_default_config_is_unchanged_qwen(self):
        """Serving v4 must stay bit-identical: same model, same pooling."""
        cfg = Config()
        assert cfg.model.pooling == "last"

    def test_default_yaml_still_loads(self):
        cfg = load_config("configs/default.yaml")
        assert cfg.model.pooling == "last"
        assert cfg.model.backbone == "qwen3.5-0.8b"
