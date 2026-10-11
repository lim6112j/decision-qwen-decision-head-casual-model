"""OpenRouter labeler: schema, label mapping, error handling (no network)."""

import json

import pytest

from decision_lab.real.labels import LabelItem
from decision_lab.real import teacher
from decision_lab.real.teacher import (
    LabelingError,
    OpenRouterLabeler,
    allowed_labels,
    build_response_format,
)


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise teacher.requests.HTTPError(f"status {self.status_code}")

    def json(self):
        return self._payload


def _item(kind="choice", options=("left", "right", "stay")):
    return LabelItem(item_id="i", call_id="c", text="ball left", question="which way?",
                     kind=kind, options=list(options))


def _reply(answer):
    """A fake requests.post returning the given structured answer."""
    def fake_post(*args, **kwargs):
        payload = json.dumps({"answer": answer})
        return _Resp({"choices": [{"message": {"content": payload}}]})
    return fake_post


def test_allowed_labels_are_options():
    assert allowed_labels(_item()) == ["left", "right", "stay"]
    assert allowed_labels(_item("noul", ["false", "true"])) == ["false", "true"]


def test_response_format_enum_matches_labels():
    fmt = build_response_format(["a", "b"])
    schema = fmt["json_schema"]["schema"]
    assert fmt["type"] == "json_schema"
    assert schema["properties"]["answer"]["enum"] == ["a", "b"]


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(LabelingError, match="OPENROUTER_API_KEY"):
        OpenRouterLabeler(api_key=None)


def test_label_maps_answer_to_index(monkeypatch):
    monkeypatch.setattr(teacher.requests, "post", _reply("stay"))
    assert OpenRouterLabeler(api_key="key").label(_item()) == 2


def test_label_rejects_out_of_enum_answer(monkeypatch):
    monkeypatch.setattr(teacher.requests, "post", _reply("up"))
    with pytest.raises(LabelingError, match="not in allowed labels"):
        OpenRouterLabeler(api_key="key").label(_item())


def test_label_request_uses_configured_model_and_schema(monkeypatch):
    seen = {}

    def fake_post(*args, **kwargs):
        seen.update(kwargs["json"])
        return _Resp({"choices": [{"message": {"content": json.dumps({"answer": "left"})}}]})

    monkeypatch.setattr(teacher.requests, "post", fake_post)
    OpenRouterLabeler(api_key="key", model="some/model").label(_item())
    assert seen["model"] == "some/model"
    assert seen["response_format"]["type"] == "json_schema"