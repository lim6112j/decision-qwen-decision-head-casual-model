"""Labeling API endpoints: queue, manual submit, auto one, auto-all SSE."""

import json

import pytest
from fastapi.testclient import TestClient

from decision_lab.webapp import app as app_module


def _traffic_row(call_id="c1", text="ball is left"):
    return {
        "call_id": call_id,
        "text": text,
        "fields": None,
        "state_type": "custom",
        "questions": [{"type": "choice", "options": ["left", "right", "stay"],
                       "question": "Which way?"}],
        "answers": [{"predicted": "left"}],
    }


class FakeLabeler:
    """Stands in for OpenRouterLabeler — always picks option 0."""

    def __init__(self, model):
        self.model = model

    def label(self, item):
        return 0


@pytest.fixture
def client(tmp_path, monkeypatch):
    traffic = tmp_path / "traffic"
    traffic.mkdir()
    (traffic / "2026-01-01.jsonl").write_text(
        json.dumps(_traffic_row()) + "\n" + json.dumps(_traffic_row("c2", "another state"))
    )
    labels = tmp_path / "labels.jsonl"
    monkeypatch.setattr(app_module, "TRAFFIC_DIR", traffic)
    monkeypatch.setattr(app_module, "LABELS_PATH", labels)
    monkeypatch.setattr(app_module, "DISCARDED_PATH", tmp_path / "discarded.jsonl")
    # state is None (lifespan not run) → endpoints fall back to default config
    return TestClient(app_module.app)


def test_queue_returns_pending_with_prediction(client):
    body = client.get("/api/label/queue").json()
    assert len(body["items"]) == 2
    item = body["items"][0]
    assert item["predicted_idx"] == 0
    assert item["options"] == ["left", "right", "stay"]
    assert item["gold_idx"] is None


def test_manual_label_accept_and_override(client):
    item = client.get("/api/label/queue").json()["items"][0]

    acc = client.post("/api/label", json={"item_id": item["item_id"], "gold_idx": 0,
                                           "source": "accepted"}).json()
    assert acc["written"] is True
    assert acc["item"]["source"] == "accepted"

    # same item again → idempotent no-op
    again = client.post("/api/label", json={"item_id": item["item_id"], "gold_idx": 2,
                                            "source": "overridden"}).json()
    assert again["written"] is False

    # only the other item remains pending
    remaining = client.get("/api/label/queue").json()["items"]
    assert len(remaining) == 1


def test_manual_label_rejects_bad_source(client):
    item = client.get("/api/label/queue").json()["items"][0]
    res = client.post("/api/label", json={"item_id": item["item_id"], "gold_idx": 0,
                                          "source": "auto"})
    assert res.status_code == 400


def test_auto_one_labels_first_pending(client, monkeypatch):
    monkeypatch.setattr(app_module, "OpenRouterLabeler", FakeLabeler)
    res = client.post("/api/label/auto", json={}).json()
    assert res["item"]["source"] == "auto"
    assert res["item"]["gold_idx"] == 0
    assert res["item"]["labeled_by"].startswith("openrouter:")
    assert len(client.get("/api/label/queue").json()["items"]) == 1


def test_auto_one_without_key_is_clear_400(client, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    res = client.post("/api/label/auto", json={})
    assert res.status_code == 400
    assert "OPENROUTER_API_KEY" in res.json()["detail"]


def test_auto_all_streams_progress_and_done(client, monkeypatch):
    monkeypatch.setattr(app_module, "OpenRouterLabeler", FakeLabeler)
    text = client.post("/api/label/auto-all", json={}).text
    events = [line[6:] for line in text.splitlines() if line.startswith("data: ")]
    parsed = [json.loads(e) for e in events]
    kinds = [line[len("event: "):] for line in text.splitlines()
             if line.startswith("event: ")]
    assert kinds.count("progress") == 2
    assert kinds[-1] == "done"
    done = parsed[-1]
    assert done == {"labeled": 2, "failed": 0, "remaining": 0}
    assert client.get("/api/label/queue").json()["items"] == []


def test_stats_reflects_progress(client, monkeypatch):
    monkeypatch.setattr(app_module, "OpenRouterLabeler", FakeLabeler)
    assert client.get("/api/label/stats").json() == {
        "total": 2, "labeled": 0, "discarded": 0, "remaining": 2}
    client.post("/api/label/auto", json={})
    assert client.get("/api/label/stats").json() == {
        "total": 2, "labeled": 1, "discarded": 0, "remaining": 1}


def test_discard_removes_item_without_labeling(client):
    item = client.get("/api/label/queue").json()["items"][0]
    res = client.post("/api/label/discard", json={"item_id": item["item_id"]}).json()
    assert res["discarded"] is True
    assert client.post("/api/label/discard", json={"item_id": item["item_id"]}).json()["discarded"] is False
    assert len(client.get("/api/label/queue").json()["items"]) == 1
    assert client.get("/api/label/stats").json()["discarded"] == 1