"""Tests for LlamaServer: launch command, backend identity, /props probing.

No llama-server is spawned — subprocess.Popen and the readiness poll are
stubbed so the launch command can be inspected directly.
"""

import subprocess
from pathlib import Path

import pytest
import requests

from decision_lab.backbone.llama_server import LlamaServer, server_from_config
from decision_lab.config import Config


class _FakeProc:
    returncode = None

    def poll(self):
        return None

    def terminate(self):
        pass

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


@pytest.fixture
def fake_launch(monkeypatch):
    """Capture the launch command instead of starting a real process."""
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(LlamaServer, "_health_ok", lambda self, timeout=2.0: False)
    monkeypatch.setattr(LlamaServer, "_wait_ready", lambda self, timeout=60.0: None)
    monkeypatch.setattr(LlamaServer, "_probe_served_model", lambda self: None)
    return captured


def _gguf(tmp_path: Path, name: str = "model.gguf") -> Path:
    path = tmp_path / name
    path.write_bytes(b"\x00" * 16)
    return path


class TestLaunchCommand:
    def test_pooling_comes_from_the_constructor(self, fake_launch, tmp_path):
        """Regression: --pooling was hardcoded to "last", so the config
        knob was decorative and an embedding model got the wrong pooling."""
        LlamaServer(_gguf(tmp_path), pooling="cls").start()
        cmd = fake_launch["cmd"]
        assert cmd[cmd.index("--pooling") + 1] == "cls"

    def test_default_pooling_is_last(self, fake_launch, tmp_path):
        LlamaServer(_gguf(tmp_path)).start()
        cmd = fake_launch["cmd"]
        assert cmd[cmd.index("--pooling") + 1] == "last"

    def test_still_requests_embeddings(self, fake_launch, tmp_path):
        LlamaServer(_gguf(tmp_path)).start()
        assert "--embeddings" in fake_launch["cmd"]


class TestBackendId:
    def test_stable_across_instances(self, tmp_path):
        gguf = _gguf(tmp_path)
        assert LlamaServer(gguf).backend_id == LlamaServer(gguf).backend_id

    def test_changes_with_pooling(self, tmp_path):
        gguf = _gguf(tmp_path)
        assert LlamaServer(gguf, pooling="last").backend_id != \
            LlamaServer(gguf, pooling="cls").backend_id

    def test_changes_with_model_file(self, tmp_path):
        assert LlamaServer(_gguf(tmp_path, "qwen.gguf")).backend_id != \
            LlamaServer(_gguf(tmp_path, "bge-m3.gguf")).backend_id

    def test_changes_with_embed_dim(self, tmp_path):
        gguf = _gguf(tmp_path)
        assert LlamaServer(gguf, embed_dim=1024).backend_id != \
            LlamaServer(gguf, embed_dim=768).backend_id

    def test_served_model_overrides_local_file(self, tmp_path):
        """start() adopts any healthy server on the port, so the local
        gguf_path is not authoritative — /props is."""
        server = LlamaServer(_gguf(tmp_path))
        before = server.backend_id
        server._served_model_path = "/remote/other.gguf"
        assert server.backend_id != before
        assert "/remote/other.gguf" in server.backend_id

    def test_missing_file_is_identifiable(self, tmp_path):
        server = LlamaServer(tmp_path / "nope.gguf")
        assert "missing" in server.backend_id


class TestProbeServedModel:
    def test_records_reported_model_path(self, monkeypatch, tmp_path):
        class _Resp:
            status_code = 200

            @staticmethod
            def json():
                return {"model_path": "/models/bge-m3.gguf"}

        monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp())
        server = LlamaServer(_gguf(tmp_path))
        server._probe_served_model()
        assert server._served_model_path == "/models/bge-m3.gguf"

    def test_connection_error_never_raises(self, monkeypatch, tmp_path):
        def boom(*args, **kwargs):
            raise requests.ConnectionError("refused")

        monkeypatch.setattr(requests, "get", boom)
        server = LlamaServer(_gguf(tmp_path))
        server._probe_served_model()   # must not raise
        assert server._served_model_path is None

    def test_missing_key_is_tolerated(self, monkeypatch, tmp_path):
        class _Resp:
            status_code = 200

            @staticmethod
            def json():
                return {}

        monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp())
        server = LlamaServer(_gguf(tmp_path))
        server._probe_served_model()
        assert server._served_model_path is None


class TestServerFromConfig:
    def test_carries_pooling_and_dim(self, tmp_path):
        """The factory exists because six call sites each passed only
        path/port/ctx and silently dropped the settings that decide whether
        a feature cache is still valid."""
        cfg = Config()
        cfg.model.gguf_path = str(_gguf(tmp_path))
        cfg.model.pooling = "cls"
        cfg.model.embed_dim = 768

        server = server_from_config(cfg)

        assert server.pooling == "cls"
        assert server.embed_dim == 768
        assert server.context_length == cfg.model.context_length
