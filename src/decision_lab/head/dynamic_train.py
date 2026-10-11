"""Train DynamicDecisionHead with curriculum: anchor → generalize.

Phase 1 (anchor): Train on the fixed question bank to establish a baseline.
Phase 2 (generalize): Introduce varied option sets so the head learns that
  options are inputs, not architectural constants.

Training samples are (state_field_set, option_embeddings, gold_idx) tuples.
State field sets are ragged (M_i, input_dim) arrays — padded + masked per
batch at collate time, never materialized as one giant padded tensor.
Option embeddings are pre-computed via the frozen backbone and cached.

v3: each sample also carries a question-text embedding (FiLM-modulates the
option queries — see dynamic_model.py). question_emb=None marks the empty
question (~EMPTY_QUESTION_FRACTION of samples), and is_variant replaces the
old "(variant)" question-text marker for the curriculum split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from decision_lab.config import Config
from decision_lab.head.dynamic_model import (
    DynamicDecisionHead,
    get_device,
    question_option_texts,
    save_dynamic_head,
)

# ---------------------------------------------------------------------------
# Training sample
# ---------------------------------------------------------------------------


@dataclass
class DynamicTrainingSample:
    """One (state, options, gold) training instance.

    state_fields_emb: (M, input_dim) float32 — the state's field-set
        embeddings (M = 1 for the pooled-vector v1 path)
    option_embs:   (n_opts, input_dim) float32 — embedded option texts
    gold_idx:      int — which option is correct
    question_type: "choice" | "score" | "noul"
    question_text: str — for logging
    question_emb:  (input_dim,) float32 question-text embedding, or None
        for the empty question. Stored BY REFERENCE from the embed cache —
        copying would add ~4 KB × ~200k samples (~0.8 GB).
    is_variant:    True for variant-question samples (curriculum phase 2)
    """

    state_fields_emb: np.ndarray
    option_embs: np.ndarray
    gold_idx: int
    question_type: str
    question_text: str = ""
    question_emb: np.ndarray | None = None
    is_variant: bool = False


def collate_dynamic_batch(
    samples: Sequence[DynamicTrainingSample],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad + stack a batch of samples → (states, state_mask, options, questions, golds).

    states:     (B, M_max, input_dim) — zero-padded
    state_mask: (B, M_max) bool — True where a real field exists
    options:    (B, N_max, input_dim) — zero-padded
    questions:  (B, input_dim) float32 — question embeddings, zero rows for
        the empty question (question_emb is None)
    golds:      (B,) long
    """
    batch = len(samples)
    max_m = max(s.state_fields_emb.shape[0] for s in samples)
    max_n = max(s.option_embs.shape[0] for s in samples)
    dim = samples[0].state_fields_emb.shape[-1]

    xb = torch.zeros(batch, max_m, dim, dtype=torch.float32)
    mask = torch.zeros(batch, max_m, dtype=torch.bool)
    ob = torch.zeros(batch, max_n, dim, dtype=torch.float32)
    qb = torch.zeros(batch, dim, dtype=torch.float32)
    yb = torch.tensor([s.gold_idx for s in samples], dtype=torch.long)

    for i, s in enumerate(samples):
        m = s.state_fields_emb.shape[0]
        xb[i, :m] = torch.from_numpy(np.asarray(s.state_fields_emb, dtype=np.float32))
        mask[i, :m] = True
        n = s.option_embs.shape[0]
        ob[i, :n] = torch.from_numpy(np.asarray(s.option_embs, dtype=np.float32))
        if s.question_emb is not None:
            qb[i] = torch.from_numpy(np.asarray(s.question_emb, dtype=np.float32))

    return xb.to(device), mask.to(device), ob.to(device), qb.to(device), yb.to(device)


# ---------------------------------------------------------------------------
# Variant option-set generation
# ---------------------------------------------------------------------------

# Pre-defined synonym maps for generating varied option sets during training.
# Each entry maps an original option key → list of synonym labels for variants.

CHOICE_SYNONYM_MAPS: dict[str, dict[str, list[str]]] = {
    "sentiment": {
        "positive": ["favorable", "good", "approving"],
        "negative": ["unfavorable", "bad", "critical"],
        "neutral": ["neutral", "balanced", "impartial"],
    },
    "urgency": {
        "low": ["minor", "routine", "standard"],
        "medium": ["moderate", "normal", "regular"],
        "high": ["elevated", "important", "significant"],
        "critical": ["severe", "urgent", "immediate"],
    },
    "paddle_direction": {
        "left": ["move_left", "go left", "LEFT"],
        "right": ["move_right", "go right", "RIGHT"],
        "stay": ["stay_put", "hold", "STAY"],
    },
}

SCORE_SYNONYM_MAPS: dict[str, list[list[str]]] = {
    "quality": [
        ["Terrible", "Poor", "Average", "Good", "Excellent"],
        ["1-star", "2-star", "3-star", "4-star", "5-star"],
        ["Unacceptable", "BelowAvg", "Acceptable", "AboveAvg", "Outstanding"],
        ["F", "D", "C", "B", "A"],
    ],
}

# v4: the question bank (question_bank.py) owns phrasings; canonical text
# per qid = bank phrasings entry 0.

# Fraction of training samples that carry NO question (question_emb=None).
# The zero-vector modulation is a learned constant — the head must see it
# during training so serving with an omitted question stays well-behaved.
EMPTY_QUESTION_FRACTION = 0.05


def default_question_text(qid: str) -> str:
    """Canonical question text for a qid (bank phrasings entry 0).

    Falls back to the qid itself for questions outside the v4 bank (the
    serving agent's fixed-bank path still asks config-bank qids).
    """
    from decision_lab.head.question_bank import build_question_bank
    for entry in build_question_bank():
        if entry.qid == qid:
            return entry.phrasings[0]
    return qid


# ---------------------------------------------------------------------------
# Training data generation
# ---------------------------------------------------------------------------


def generate_dynamic_training_data(
    states,
    field_features: Sequence[np.ndarray],
    base_spec: dict,
    server,
    rng: Random | None = None,
    num_variants_per_question: int = 3,
    shape_augmentation: bool = False,
    include_summary_field: bool = True,
) -> list[DynamicTrainingSample]:
    """Generate (state, options, gold) training samples from the v4 bank.

    v4 rewrite: the compositional bank (question_bank.build_question_bank)
    replaces base_spec + hand-written variants. One sample per (state, qid)
    pair — phrasing and option-set drawn with the seeded rng — never the
    full phrasing cross-product (that's 10M+ samples; one draw keeps ~200k).
    Stage 1 (is_stage2=False): canonical phrasing, base option set — every
    qid including the breakout forcing pair. Stage 2: everything else.

    Args:
        states: list of TextState objects with gold labels.
        field_features: per-state (M_i, input_dim) field-set embeddings,
            aligned with ``states``.
        base_spec: kept for signature compatibility (unused by the bank;
            callers may pass the config bank for logging).
        server: LlamaServer (must be running) for embedding option/question texts.
        rng: random state for phrasing/variant draws.
        num_variants_per_question: option-set synonym variants per qid (2).
        shape_augmentation: add shape-variant duplicates of document states
            (flattened / marker-stripped re-renderings with identical gold —
            see states/shapes.py). Breakout states are skipped: they are
            trained bare, and shape invariance there points the wrong way
            (key prefixes flip breakout answers).
        include_summary_field: must match feature extraction (same config key
            as dynamic_head.include_summary_field) so variant field sets
            chunk identically with cached ones.

    Returns:
        list of DynamicTrainingSample.
    """
    if rng is None:
        rng = Random(42)

    from decision_lab.head.question_bank import build_question_bank

    bank = build_question_bank()

    # Option-set synonym variants per choice/score qid (CHOICE/SCORE_SYNONYM_MAPS
    # keyed by qid; bank qids without an entry get base options only).
    option_variants: dict[str, list[dict]] = {}
    for entry in bank:
        if entry.kind == "choice" or entry.kind == "score":
            variants = _variants_for_bank_entry(entry, num_variants_per_question, rng)
            if variants:
                option_variants[entry.qid] = variants

    # Collect all unique option/question texts and embed them in one pass
    all_texts: set[str] = {"false", "true"}
    for entry in bank:
        all_texts.update(entry.options or ())
        all_texts.update(entry.phrasings)
        for v in option_variants.get(entry.qid, []):
            all_texts.update(v["options"])

    unique_texts = sorted(all_texts)
    print(f"  embedding {len(unique_texts)} unique option/question texts...")
    text_to_emb = embed_texts(server, unique_texts)

    # Shape-variant duplicates of document states: same latents → same gold,
    # different surface shape. Their field sets don't exist in the precomputed
    # feature caches (those hold original-shape texts only), so embed them here
    # — the server is already up for option/question embeddings. Breakout
    # states are skipped; see states/shapes.py for why.
    extra: list[tuple[np.ndarray, int]] = []   # (field_set_emb, state_index)
    if shape_augmentation:
        from decision_lab.states.dataset import TextState
        from decision_lab.states.fields import state_field_set
        from decision_lab.states.shapes import shape_variants

        texts: list[str] = []
        pending: list[tuple[int, list[str]]] = []   # (state_index, field_texts)
        for i, state in enumerate(states):
            if state.state_type.startswith("breakout"):
                continue
            for variant in shape_variants(state.text):
                vstate = TextState(doc_id=-1, state_type="custom", text=variant, labels={})
                field_texts = state_field_set(vstate, include_summary_field)
                pending.append((i, field_texts))
                texts.extend(field_texts)
        unique = list(dict.fromkeys(texts))
        print(f"  shape augmentation: embedding {len(unique)} unique variant field texts...")
        emb_by_text = embed_texts(server, unique)
        for state_index, field_texts in pending:
            extra.append((
                np.stack([emb_by_text[t] for t in field_texts]),
                state_index,
            ))
        print(f"  shape augmentation: {len(extra)} variant states added")

    samples = []

    def emit(i: int, state_fields_emb: np.ndarray) -> None:
        """Emit ONE sample per applicable qid for one state embedding."""
        state = states[i]
        for entry in bank:
            if not entry.applicable(state):
                continue

            gold = entry.gold(state)

            # Draw phrasing + option set for this (state, qid) pair
            phrasing_idx = rng.randrange(len(entry.phrasings))
            question_text = entry.phrasings[phrasing_idx]
            # Stage-2 marker: any non-canonical phrasing or option variant.
            # Canonical-phrasing base-option samples are stage 1 — that
            # includes EVERY qid, so the breakout forcing pair anchors the
            # question path in stage 1.
            variant_opts = option_variants.get(entry.qid)
            # occasionally keep base options even in stage 2 samples
            use_variant = variant_opts is not None and (
                phrasing_idx != 0 or rng.random() < 0.5
            )
            if use_variant:
                v = variant_opts[rng.randrange(len(variant_opts))]
                option_texts, gold_map = v["options"], v["gold_map"]
            else:
                option_texts, gold_map = list(entry.options), None

            kind = entry.kind
            if kind == "noul":
                gold_idx = 1 if bool(gold) else 0
                # noul goes through the attention path with [false, true]
                opt_embs = np.stack([text_to_emb["false"], text_to_emb["true"]])
            elif kind == "choice":
                mapped = gold_map[str(gold)] if gold_map else gold
                gold_idx = option_texts.index(str(mapped))
                opt_embs = np.stack([text_to_emb[t] for t in option_texts])
            else:  # score
                mapped = gold_map[str(gold)] if gold_map else gold
                gold_idx = int(mapped)
                opt_embs = np.stack([text_to_emb[t] for t in option_texts])

            # Mostly attach the question embedding; a small fraction is the
            # empty question (None → zero-vector query at forward).
            if rng.random() < EMPTY_QUESTION_FRACTION:
                question_text = ""
                question_emb = None
            else:
                question_emb = text_to_emb[question_text]  # by reference

            is_stage2 = phrasing_idx != 0 or use_variant

            samples.append(DynamicTrainingSample(
                state_fields_emb=state_fields_emb,
                option_embs=opt_embs,
                gold_idx=gold_idx,
                question_type=kind,
                question_text=question_text,
                question_emb=question_emb,
                is_variant=is_stage2,
            ))

    for i in range(len(states)):
        emit(i, field_features[i])
    for field_emb, state_index in extra:
        # Shape variants ride in both curriculum phases (is_variant=False):
        # shape invariance is part of the anchor task, not a generalization
        # axis saved for phase 2.
        emit(state_index, field_emb)

    n_stage1 = sum(1 for s in samples if not s.is_variant)
    print(f"  generated {len(samples)} training samples "
          f"({n_stage1} stage-1 / {len(samples) - n_stage1} stage-2) over "
          f"{len(bank)} bank qids × {len(states) + len(extra)} states incl. shape variants")
    return samples


def embed_texts(server, texts: list[str]) -> dict[str, np.ndarray]:
    """Embed texts in fixed batches → {text: (input_dim,) float32}.

    Shared by synthetic training-data generation and the real-traffic
    converter (real/build_dataset.py).
    """
    emb: dict[str, np.ndarray] = {}
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embs = server.embed(batch)
        for text, e in zip(batch, embs):
            emb[text] = np.array(e, dtype=np.float32)
    return emb


def _variants_for_bank_entry(
    entry, num_variants: int, rng: Random,
) -> list[dict]:
    """Option-set synonym variants for a bank entry (choice/score).

    Choice: one variant per synonym "column", option order shuffled
    (permutation invariance). Score: full alternative rubric sets, gold
    index mapped proportionally. Bank qids without a synonym entry get none.
    """
    kind = entry.kind
    if kind == "choice":
        synonym_map = CHOICE_SYNONYM_MAPS.get(entry.qid, {})
        if not synonym_map:
            return []
        variants = []
        num_cols = min(len(next(iter(synonym_map.values()))), num_variants)
        for col in range(num_cols):
            gold_map = {}
            options = []
            for orig_opt in entry.options:
                syns = synonym_map.get(orig_opt, [orig_opt])
                variant_opt = syns[col] if col < len(syns) else orig_opt
                gold_map[orig_opt] = variant_opt
                options.append(variant_opt)
            idx = list(range(len(options)))
            rng.shuffle(idx)
            shuffled_opts = [options[i] for i in idx]
            shuffled_map = {
                orig: shuffled_opts[idx.index(i)]
                for i, orig in enumerate(gold_map)
            }
            variants.append({"options": shuffled_opts, "gold_map": shuffled_map})
        return variants

    # score: full alternative rubric sets via proportional gold mapping
    level_variants = SCORE_SYNONYM_MAPS.get(entry.qid, [])
    if not level_variants:
        return []
    variants = []
    base_n = len(entry.options)
    for level_set in level_variants[:num_variants]:
        var_n = len(level_set)
        gold_map = {}
        for base_idx in range(base_n):
            var_idx = min(int(round(base_idx * (var_n - 1) / max(base_n - 1, 1))), var_n - 1)
            gold_map[str(base_idx)] = var_idx
        variants.append({"options": list(level_set), "gold_map": gold_map})
    return variants


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_dynamic_head(
    samples: Sequence[DynamicTrainingSample],
    cfg: Config,
    model_path: Path,
    anchor_fraction: float = 0.3,
) -> DynamicDecisionHead:
    """Train a DynamicDecisionHead with curriculum.

    Phase 1 (anchor): train on base-question samples only (anchor_fraction of epochs).
    Phase 2 (generalize): introduce variant samples, train on all data.

    Args:
        samples: all training samples (base + variant).
        cfg: full Config.
        model_path: where to save the checkpoint.
        anchor_fraction: fraction of max_epochs spent on anchor-only training.
    """
    hc = cfg.dynamic_head
    device = get_device()
    print(f"Training dynamic head on device: {device} ({len(samples)} samples)")

    # Separate stage-1 (canonical) vs stage-2 samples for curriculum
    # (explicit is_stage2 flag — the v4 bank's two-stage curriculum)
    stage1_samples = [s for s in samples if not s.is_variant]
    stage2_samples = [s for s in samples if s.is_variant]
    print(f"  stage-1 samples: {len(stage1_samples)}, stage-2 samples: {len(stage2_samples)}")

    # Curriculum pools: stage-1 vs all samples
    is_stage2 = torch.tensor([s.is_variant for s in samples], dtype=torch.bool)
    stage1_positions = torch.nonzero(~is_stage2).squeeze(-1)

    # 3-way split for calibration holdout
    n = len(samples)
    idx = np.random.RandomState(cfg.generator.seed).permutation(n)
    calib_n = max(1, int(n * cfg.calibration.holdout_fraction))
    calib_idx_set = set(idx[:calib_n].tolist())

    # Train/val split on remaining samples
    rest_idx = idx[calib_n:]
    split = int(len(rest_idx) * 0.8)
    train_positions = torch.tensor(rest_idx[:split], dtype=torch.long)
    val_positions = torch.tensor(rest_idx[split:], dtype=torch.long)

    # Stage-1 pool: canonical-phrasing/base-option samples in the train split
    stage2_arr = np.array([s.is_variant for s in samples], dtype=bool)
    in_train = np.zeros(n, dtype=bool)
    in_train[rest_idx[:split]] = True
    stage1_pool = torch.tensor(
        np.where(~stage2_arr & in_train)[0], dtype=torch.long,
    )

    print(f"  train: {len(train_positions)}, val: {len(val_positions)}, "
          f"calibration holdout: {calib_n} "
          f"(stage-1 pool: {len(stage1_pool)})")

    # Model — v4 question-field attention architecture
    model = DynamicDecisionHead(
        input_dim=1024,
        hidden_dim=hc.hidden_dim,
        d_k=hc.d_k,
        dropout=hc.dropout,
        state_set=True,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=hc.learning_rate, weight_decay=hc.weight_decay)

    anchor_epochs = max(1, int(hc.max_epochs * anchor_fraction))

    best_val_acc = -1.0
    best_state = None
    patience_counter = 0

    def batch_at(positions: torch.Tensor, start: int, batch_size: int) -> list[DynamicTrainingSample]:
        bpos = positions[start : start + batch_size]
        return [samples[i] for i in bpos.tolist()]

    batch_size = hc.batch_size
    for epoch in range(1, hc.max_epochs + 1):
        # Curriculum: first anchor_epochs use stage-1 samples only. Both
        # phases train on the train split only — never on val/calibration.
        if epoch <= anchor_epochs and len(stage1_pool) > 0:
            pool = stage1_pool
            phase = "stage-1"
        else:
            pool = train_positions
            phase = "stage-2"

        model.train()
        train_loss = 0.0
        # Shuffle for this epoch
        order = pool[torch.randperm(len(pool))]

        for start in range(0, len(order), batch_size):
            xb, mask, ob, qb, yb = collate_dynamic_batch(
                batch_at(order, start, batch_size), device,
            )

            optimizer.zero_grad()
            scores = model.forward_choice(xb, ob, mask, qb)      # (B, N)
            loss = F.cross_entropy(scores, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(yb)

        train_loss /= len(order)

        # Validation (full val set + stage-1-only subset for the guard)
        model.eval()
        val_correct = 0
        val_total = 0
        s1_correct = 0
        s1_total = 0
        with torch.no_grad():
            for start in range(0, len(val_positions), batch_size):
                batch = batch_at(val_positions, start, batch_size)
                xb, mask, ob, qb, yb = collate_dynamic_batch(batch, device)
                scores = model.forward_choice(xb, ob, mask, qb)
                pred = scores.argmax(dim=1)
                val_correct += int((pred == yb).sum().item())
                val_total += len(yb)
                # stage-1 subset for the escape guard
                s1_idx = [j for j, s in enumerate(batch) if not s.is_variant]
                if s1_idx:
                    s1_correct += int((pred[s1_idx] == yb[s1_idx]).sum().item())
                    s1_total += len(s1_idx)

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        s1_val_acc = s1_correct / s1_total if s1_total > 0 else val_acc

        # Escape guard: breakout-domain training can sit at a near-symmetric
        # plateau (all options reading the same field) whose escape is a
        # stochastic bootstrap event. Computed on the STAGE-1 val subset so
        # the 0.65 threshold keeps its meaning (the full val set mixes in
        # stage-2 samples the anchor-phase model can't answer yet). A
        # healthy run has escaped by a few epochs into stage 2 (val_acc
        # jumps ~0.52 → ~0.80); if it hasn't, further epochs are wasted —
        # abort so the caller can retry with a different init.
        guard_epoch = anchor_epochs + 10
        if epoch == guard_epoch and s1_val_acc < 0.65:
            print(f"  ESCAPE GUARD: stage-1 val_acc={s1_val_acc:.3f} at epoch {epoch} — "
                  f"still at the symmetric plateau, aborting (caller should retry "
                  f"with a different init)")
            return None

        if epoch == 1 or epoch % 10 == 0 or epoch == hc.max_epochs:
            print(f"  epoch {epoch:3d} [{phase}]: train_loss={train_loss:.4f}  "
                  f"val_acc={val_acc:.3f}  stage-1 val_acc={s1_val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= hc.patience:
            print(f"  early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()

    # Global temperature calibration (single scalar, not per-question)
    temperature = _fit_global_temperature(model, samples, calib_idx_set, device)

    save_dynamic_head(model, str(model_path), temperature=temperature)
    print(f"  saved to {model_path}  (best val_acc={best_val_acc:.3f}, T={temperature:.3f})")
    return model


def _fit_global_temperature(
    model: DynamicDecisionHead,
    samples: Sequence[DynamicTrainingSample],
    calib_idx_set: set,
    device: torch.device,
    lr: float = 1e-2,
    max_iter: int = 1000,
    batch_size: int = 256,
) -> float:
    """Fit a single scalar temperature on the calibration holdout set.

    Question types mix option counts, so logits are collected per batch
    (ragged widths) and the closure sums batch-wise weighted CE instead of
    concatenating into one tensor.
    """
    calib_samples = [samples[i] for i in sorted(calib_idx_set) if i < len(samples)]
    if not calib_samples:
        return 1.0

    model.eval()
    batch_logits: list[tuple[torch.Tensor, torch.Tensor, int]] = []
    with torch.no_grad():
        for start in range(0, len(calib_samples), batch_size):
            chunk = calib_samples[start : start + batch_size]
            xb, mask, ob, qb, yb = collate_dynamic_batch(chunk, device)
            logits = model.forward_choice(xb, ob, mask, qb)
            batch_logits.append((logits.cpu().detach(), yb.cpu().detach(), len(chunk)))

    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        t = log_t.exp().clamp_min(1e-3)
        total = sum(n for _, _, n in batch_logits)
        loss = sum(
            F.cross_entropy(logits / t, labels, reduction="sum") / total
            for logits, labels, _ in batch_logits
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


# ---------------------------------------------------------------------------
# Label encoding for dynamic questions
# ---------------------------------------------------------------------------


def encode_dynamic_labels(states, question_config: dict) -> np.ndarray:
    """Convert TextState gold labels into int class indices for a dynamic question.

    Args:
        states: list of TextState.
        question_config: dynamic question dict with "qid" and optional "gold_map".

    Returns:
        (N,) int64 array of class indices.
    """
    qid = question_config["qid"]
    gold_map = question_config.get("gold_map")
    q = question_config["question"]
    kind = q["type"]

    indices = []
    for s in states:
        gold = s.labels.get(qid)
        if gold is None:
            indices.append(-1)  # sentinel for missing
            continue

        if gold_map is not None:
            mapped = gold_map[str(gold)]
        else:
            mapped = gold

        if kind == "noul":
            idx = 1 if bool(mapped) else 0
        elif kind == "choice":
            option_texts = question_option_texts(q)
            idx = option_texts.index(str(mapped))
        else:
            idx = int(mapped)
        indices.append(idx)

    return np.array(indices, dtype=np.int64)