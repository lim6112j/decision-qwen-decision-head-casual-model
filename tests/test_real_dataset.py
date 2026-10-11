"""Real-traffic converter: chunking contract, sample emission, holdout split."""

import numpy as np

from decision_lab.real.build_dataset import (
    build_samples,
    field_texts_for_item,
    split_holdout,
)
from decision_lab.real.labels import SOURCE_ACCEPTED, LabelItem

DIM = 8


class FakeServer:
    """Deterministic embeddings — no llama-server, no network."""

    def embed(self, texts):
        return [np.full(DIM, (len(t) % 5) + 1, dtype=np.float32) for t in texts]


def _item(gold_idx=0, fields=None, text="one. two. three."):
    return LabelItem(
        item_id="i", call_id="c", text=text, question="which?",
        kind="choice", options=["a", "b", "c"], predicted_idx=1,
        fields=fields, gold_idx=gold_idx, source=SOURCE_ACCEPTED,
    )


def test_caller_fields_used_verbatim():
    item = _item(fields=["f1", "f2"])
    assert field_texts_for_item(item, include_summary=True) == ["f1", "f2"]


def test_heuristic_split_applies_with_summary_field():
    texts = field_texts_for_item(_item(fields=None), include_summary=True)
    assert texts[0] == "one. two. three."     # summary field prepended
    assert len(texts) >= 2                     # split into fields


def test_build_samples_emits_stage2_sample():
    samples = build_samples([_item(gold_idx=2)], FakeServer(), include_summary_field=True)
    assert len(samples) == 1
    s = samples[0]
    assert s.gold_idx == 2
    assert s.is_variant is True                # real traffic → curriculum stage 2
    assert s.option_embs.shape == (3, DIM)
    assert s.question_emb is not None
    assert s.state_fields_emb.shape[1] == DIM


def test_build_samples_skips_unlabeled():
    unlabeled = LabelItem(item_id="u", call_id="c", text="t", question="q",
                          kind="choice", options=["a", "b"], gold_idx=None)
    assert build_samples([unlabeled], FakeServer()) == []


def test_split_holdout_is_seeded_and_bounded():
    items = [_item() for _ in range(10)]
    train, test = split_holdout(items, 0.2, seed=42)
    assert len(test) == 2 and len(train) == 8
    assert split_holdout(items, 0.2, seed=42) == (train, test)   # deterministic


def test_split_holdout_keeps_singleton_whole():
    only = [_item()]
    train, test = split_holdout(only, 0.2)
    assert train == only and test == []