"""The dynamic head must be loaded ONCE and shared between both entry points.

Two consumers need it: ``/api/decide-dynamic`` reads the ``dynamic_paths`` map
(keyed by sub-path), while the SSE endpoints (``/api/run``, ``/api/compare``)
read ``agents["head_dynamic"]`` and have no ``path`` field to route by. They
used to each call ``_load_dynamic_agent``, loading the same checkpoint twice
and printing "loaded dynamic head from ..." twice.

These tests pin the contract: the SSE agent IS the default sub-path's head
(same object, not a copy), and the default is resolved by one shared helper so
the two entry points cannot disagree.
"""

import pytest

from decision_lab.config import load_config
from decision_lab.webapp.agents import build_agents, resolve_default_path


class _FakeTypedHead:
    """Minimal stand-in for TypedDecisionHead (build_agents only calls eval/to)."""

    question_spec = {}
    temperatures = {}

    def eval(self):
        return self

    def to(self, device):
        return self


class _FakeServer:
    backend_id = "served=fake|pooling=last|ctx=2048|dim=1024"


@pytest.fixture
def cfg():
    return load_config("configs/default.yaml")


def _load_calls(monkeypatch):
    """Count how many dynamic heads get constructed."""
    import decision_lab.webapp.agents as agents_mod

    made: list[object] = []
    real = agents_mod._load_dynamic_agent

    def counting(*args, **kwargs):
        agent = real(*args, **kwargs)
        made.append(agent)
        return agent

    monkeypatch.setattr(agents_mod, "_load_dynamic_agent", counting)
    # The typed head needs a real checkpoint on disk; stub it out.
    monkeypatch.setattr(agents_mod, "_load_trained_head",
                        lambda cfg: (_FakeTypedHead(), {}))
    return made


def test_single_path_loads_head_exactly_once(cfg, monkeypatch):
    """No web.paths → one path; the checkpoint must be loaded once, not twice."""
    made = _load_calls(monkeypatch)
    agents = build_agents(cfg, _FakeServer())

    assert len(made) == 1, f"dynamic head loaded {len(made)}x, expected 1"
    assert agents["head_dynamic"] is made[0]


def test_sse_agent_is_the_default_subpath_object(cfg, monkeypatch):
    """With sub-paths, head_dynamic must BE the default path's head."""
    made = _load_calls(monkeypatch)
    paths = {"doc": object(), "typed": object()}
    cfg.web.default_path = "doc"

    agents = build_agents(cfg, _FakeServer(), paths)

    assert len(made) == 0, "pre-loaded paths must not trigger a second load"
    assert agents["head_dynamic"] is paths["doc"]


def test_no_default_with_several_paths_leaves_agent_absent(cfg, monkeypatch):
    """Ambiguous default is never guessed — head_dynamic is simply unavailable."""
    _load_calls(monkeypatch)
    cfg.web.default_path = ""
    paths = {"doc": object(), "typed": object()}

    agents = build_agents(cfg, _FakeServer(), paths)

    assert "head_dynamic" not in agents
    assert "head_trained" in agents     # the others still work


def test_single_path_resolves_default_to_it(cfg):
    assert resolve_default_path(cfg, {"": object()}) == ""
    cfg.web.default_path = "typed"
    assert resolve_default_path(cfg, {"doc": object(), "typed": object()}) == "typed"


def test_several_paths_without_default_asks_the_caller(cfg):
    cfg.web.default_path = ""
    with pytest.raises(ValueError, match="path` is required"):
        resolve_default_path(cfg, {"doc": object(), "typed": object()})


def test_typo_in_default_path_blames_the_config_not_the_caller(cfg):
    """A declared-but-missing default is a config error, not a missing request
    field — the message must point at the typo, not say `path` is required."""
    cfg.web.default_path = "doc"        # not in the map → misconfiguration
    with pytest.raises(ValueError, match="web.default_path 'doc'"):
        resolve_default_path(cfg, {"typed": object()})
