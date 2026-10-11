"""Convert labeled real-traffic items into DynamicTrainingSample rows.

Chunking matches serving exactly: a caller-supplied ``fields`` list (HTTP
``custom_fields``) is used verbatim, otherwise the same
``states.fields.state_field_set`` heuristic used at inference. This is the
invariant that keeps real samples in the same field-set geometry the head
was trained on.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from decision_lab.head.dynamic_train import DynamicTrainingSample, embed_texts
from decision_lab.real.labels import SOURCE_AUTO, LabelItem, load_labels
from decision_lab.states.dataset import TextState
from decision_lab.states.fields import state_field_set


def field_texts_for_item(item: LabelItem, include_summary: bool) -> list[str]:
    """The field set for an item, identical to the one used when serving it."""
    if item.fields is not None:
        return list(item.fields)          # caller chunking — verbatim
    state = TextState(doc_id=-1, state_type="custom", text=item.text, labels={})
    return state_field_set(state, include_summary)


def split_holdout(
    items: list[LabelItem], fraction: float, seed: int = 42,
) -> tuple[list[LabelItem], list[LabelItem]]:
    """Seeded split → (train_items, test_items). A small set stays whole."""
    if fraction <= 0 or len(items) < 2:
        return list(items), []
    idx = list(range(len(items)))
    random.Random(seed).shuffle(idx)
    n_hold = max(1, int(len(items) * fraction))
    hold = set(idx[:n_hold])
    train = [it for i, it in enumerate(items) if i not in hold]
    test = [it for i, it in enumerate(items) if i in hold]
    return train, test


def build_samples(
    items: list[LabelItem], server, include_summary_field: bool = True,
) -> list[DynamicTrainingSample]:
    """Embed items and emit stage-2 samples (real traffic is generalization)."""
    prepared: list[tuple[LabelItem, list[str]]] = []
    texts: set[str] = set()
    for item in items:
        if item.gold_idx is None:
            continue
        field_texts = field_texts_for_item(item, include_summary_field)
        prepared.append((item, field_texts))
        texts.update(field_texts)
        texts.update(item.options)
        if item.question:
            texts.add(item.question)

    emb = embed_texts(server, sorted(texts)) if texts else {}

    samples: list[DynamicTrainingSample] = []
    for item, field_texts in prepared:
        samples.append(DynamicTrainingSample(
            state_fields_emb=np.stack([emb[t] for t in field_texts]),
            option_embs=np.stack([emb[t] for t in item.options]),
            gold_idx=int(item.gold_idx),
            question_type=item.kind,
            question_text=item.question,
            question_emb=emb[item.question] if item.question else None,
            is_variant=True,
        ))
    return samples


def save_items(items: list[LabelItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(it.to_dict(), ensure_ascii=False) for it in items),
        encoding="utf-8",
    )


def load_items(path: Path) -> list[LabelItem]:
    if not path.exists():
        return []
    return [
        LabelItem.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def load_samples(path: Path) -> list[DynamicTrainingSample]:
    """Load a saved samples.pt (weights_only=False: trusted local artifact)."""
    if not path.exists():
        return []
    return torch.load(path, weights_only=False)


def convert_labels(cfg, server, data_dir: Path) -> dict:
    """labels.jsonl → samples.pt + test_real.jsonl. Returns a summary dict."""
    real_dir = data_dir / "real"
    labeled = list(load_labels(real_dir / "labels.jsonl").values())
    labeled = [it for it in labeled if it.gold_idx is not None]

    if cfg.real.exclude_auto:
        labeled = [it for it in labeled if it.source != SOURCE_AUTO]
    if not labeled:
        raise ValueError(
            "no labeled items found — capture traffic and label it first "
            f"(looked in {real_dir / 'labels.jsonl'})"
        )

    train_items, test_items = split_holdout(labeled, cfg.real.holdout_fraction, cfg.real.seed)
    samples = build_samples(train_items, server, cfg.dynamic_head.include_summary_field)

    torch.save(samples, real_dir / "samples.pt")
    save_items(test_items, real_dir / "test_real.jsonl")

    return {
        "labeled": len(labeled),
        "train": len(train_items),
        "test": len(test_items),
        "samples": len(samples),
    }