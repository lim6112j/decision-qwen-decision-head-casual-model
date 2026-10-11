"""Opt-in traffic capture: gate + JSONL rows + the endpoint hook."""

import json
from types import SimpleNamespace

from decision_lab.webapp import app as app_module
from decision_lab.webapp.traffic_log import TrafficLogger
from decision_lab.states.dataset import TextState


def _row(call_id: str) -> dict:
    return {"call_id": call_id, "text": "hello", "questions": [], "answers": []}


def test_disabled_never_logs(tmp_path):
    logger = TrafficLogger(tmp_path, enabled=False)
    assert logger.should_log({"x-decision-lab-log": "1"}) is False


def test_enabled_requires_header(tmp_path):
    logger = TrafficLogger(tmp_path, enabled=True)
    assert logger.should_log({"x-decision-lab-log": "1"}) is True
    assert logger.should_log({"x-decision-lab-log": "0"}) is False
    assert logger.should_log({}) is False


def test_record_writes_jsonl(tmp_path):
    logger = TrafficLogger(tmp_path, enabled=True)
    logger.record(_row("a"))
    logger.record(_row("b"))

    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    lines = [json.loads(line) for line in files[0].read_text().splitlines()]
    assert [r["call_id"] for r in lines] == ["a", "b"]


def _hook(tmp_path, monkeypatch, header):
    monkeypatch.setattr(
        app_module, "state",
        SimpleNamespace(traffic=TrafficLogger(tmp_path, enabled=True)),
    )
    req = SimpleNamespace(headers={"x-decision-lab-log": header} if header else {},
                          custom_fields=None,
                          questions=[{"type": "noul", "question": "?"}])
    s = TextState(doc_id=-1, state_type="custom", text="hi", labels={})
    app_module._maybe_log_traffic(req, s, req, [{"predicted": True}], 3.2)


def test_hook_writes_row_when_opted_in(tmp_path, monkeypatch):
    _hook(tmp_path, monkeypatch, "1")
    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    row = json.loads(files[0].read_text().splitlines()[0])
    assert row["text"] == "hi"
    assert row["answers"] == [{"predicted": True}]
    assert row["state_type"] == "custom"


def test_hook_silent_without_header(tmp_path, monkeypatch):
    _hook(tmp_path, monkeypatch, "")
    assert list(tmp_path.glob("*.jsonl")) == []