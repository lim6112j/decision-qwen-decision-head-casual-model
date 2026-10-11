"""Label store: item explosion, dedupe, pending queue, idempotent append."""

import json

from decision_lab.real.labels import (
    SOURCE_ACCEPTED,
    SOURCE_OVERRIDDEN,
    LabelItem,
    append_label,
    build_queue,
    discard_item,
    items_from_call,
    make_item_id,
    option_texts,
    predicted_index,
    stats,
)


def _call(call_id="c1", text="the ball is left", questions=None, answers=None):
    return {
        "call_id": call_id,
        "text": text,
        "questions": questions or [{"type": "choice", "options": ["left", "right", "stay"],
                                     "question": "Which way?"}],
        "answers": answers or [{"predicted": "left"}],
    }


def test_option_texts_by_kind():
    assert option_texts({"type": "noul"}) == ["false", "true"]
    assert option_texts({"type": "choice", "options": ["a", "b"]}) == ["a", "b"]
    assert option_texts({"type": "score", "levels": ["lo", "hi"]}) == ["lo", "hi"]


def test_predicted_index_maps_each_kind():
    assert predicted_index({"type": "noul"}, {"predicted": True}) == 1
    assert predicted_index({"type": "noul"}, {"predicted": False}) == 0
    assert predicted_index({"type": "choice", "options": ["x", "y"]}, {"predicted": "y"}) == 1
    assert predicted_index({"type": "score", "levels": ["a", "b", "c"]}, {"predicted": 2}) == 2
    assert predicted_index({"type": "choice", "options": ["x"]}, {"predicted": "nope"}) is None


def test_items_from_call_explodes_and_skips_malformed():
    row = _call(questions=[
        {"type": "choice", "options": ["left", "right"], "question": "q1"},
        {"type": "bogus"},                                  # skipped
        {"type": "choice", "options": ["only-one"]},        # < 2 options → skipped
        {"type": "noul", "question": "q4"},
    ], answers=[{"predicted": "left"}, {}, {}, {"predicted": False}])
    items = items_from_call(row)
    assert [i.kind for i in items] == ["choice", "noul"]
    assert items[0].predicted_idx == 0
    assert items[1].predicted_idx == 0
    assert items[0].options == ["left", "right"]


def test_make_item_id_is_content_stable():
    a = make_item_id("t", None, "q", ["x", "y"], "choice")
    b = make_item_id("t", None, "q", ["x", "y"], "choice")
    c = make_item_id("t", None, "q", ["x", "z"], "choice")
    assert a == b and a != c


def test_build_queue_dedupes_and_excludes_labeled(tmp_path):
    traffic = tmp_path / "traffic"
    traffic.mkdir()
    # same call twice + a different one
    (traffic / "2026-01-01.jsonl").write_text(
        "\n".join(json.dumps(_call()) for _ in range(2))
        + "\n" + json.dumps(_call(call_id="c2", text="a different state"))
    )
    labels = tmp_path / "labels.jsonl"

    queue = build_queue(traffic, labels)
    assert len(queue) == 2                      # deduped by content

    one = queue[0]
    one.gold_idx, one.source = 0, SOURCE_ACCEPTED
    assert append_label(one, labels) is True

    queue2 = build_queue(traffic, labels)
    assert len(queue2) == 1
    assert queue2[0].item_id != one.item_id

    s = stats(traffic, labels)
    assert s == {"total": 2, "labeled": 1, "discarded": 0, "remaining": 1}


def test_discard_removes_without_labeling(tmp_path):
    traffic = tmp_path / "traffic"
    traffic.mkdir()
    (traffic / "2026-01-01.jsonl").write_text(
        json.dumps(_call()) + "\n" + json.dumps(_call(call_id="c2", text="a different state"))
    )
    labels = tmp_path / "labels.jsonl"
    discarded = tmp_path / "discarded.jsonl"

    queue = build_queue(traffic, labels, discarded)
    assert len(queue) == 2
    assert discard_item(queue[0].item_id, discarded) is True
    assert discard_item(queue[0].item_id, discarded) is False   # idempotent

    remaining = build_queue(traffic, labels, discarded)
    assert [i.item_id for i in remaining] == [queue[1].item_id]
    assert stats(traffic, labels, discarded) == {
        "total": 2, "labeled": 0, "discarded": 1, "remaining": 1,
    }


def test_append_label_idempotent(tmp_path):
    labels = tmp_path / "labels.jsonl"
    item = LabelItem(item_id="deadbeef", call_id="c", text="t", question="q",
                     kind="choice", options=["a", "b"], gold_idx=1, source=SOURCE_OVERRIDDEN)
    assert append_label(item, labels) is True
    assert append_label(item, labels) is False
    assert len(labels.read_text().splitlines()) == 1