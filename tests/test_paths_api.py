"""Sub-path routing (``web.paths``) for /api/decide-dynamic.

Sub-paths give each problem its own dynamic head off one shared backbone. The
caller selects one with the request's ``path`` field — explicit routing, so
there is no router to misroute and an unknown name is an error, never a silent
fallback to a different head.
"""

import pytest

from decision_lab.config import load_config
from decision_lab.webapp.app import WebAppState


def _state(default: str = ""):
    cfg = load_config("configs/default.yaml")
    cfg.web.default_path = default
    return WebAppState(cfg)          # no server started — routing only


def test_backwards_compatible_single_path():
    st = _state()
    st.dynamic_paths = {"": object()}
    assert st.resolve_path(None) == ""


def test_default_path_is_used_when_omitted():
    st = _state(default="doc")
    st.dynamic_paths = {"doc": object(), "typed": object()}
    assert st.resolve_path(None) == "doc"
    assert st.resolve_path("typed") == "typed"


def test_multiple_paths_require_explicit_selection():
    st = _state()
    st.dynamic_paths = {"doc": object(), "typed": object()}
    with pytest.raises(ValueError, match="path"):
        st.resolve_path(None)


def test_unknown_path_is_an_error_not_a_fallback():
    st = _state()
    st.dynamic_paths = {"doc": object()}
    with pytest.raises(ValueError, match="unknown path"):
        st.resolve_path("nope")
