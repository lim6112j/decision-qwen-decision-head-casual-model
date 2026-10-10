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
    make_choice_question,
    make_noul_question,
    make_score_question,
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

# v3: hand-written natural-language phrasings per qid. The backbone uses
# "last"-token pooling (order-sensitive), so these are FIXED lists — no
# generated or permuted phrasings. Entry 0 is the canonical/default question;
# variants sample from the whole list.
QUESTION_PHRASINGS: dict[str, list[str]] = {
    "paddle_direction": [
        "Which direction should the paddle move?",
        "which direction the paddle move?",
        "paddle direction?",
        "Where should the paddle go next?",
    ],
    "ball_motion": [
        "Which way is the ball moving horizontally?",
        "ball horizontal motion?",
        "Is the ball drifting left or right?",
    ],
    "sentiment": [
        "What is the sentiment of this message?",
        "sentiment of the message?",
        "How does this message feel tonally?",
    ],
    "urgency": [
        "How urgent is this item?",
        "urgency level?",
        "How soon does this need attention?",
    ],
    "quality": [
        "What is the quality of this text?",
        "quality rating?",
        "How well written is this?",
    ],
}

# Fraction of training samples that carry NO question (question_emb=None).
# The zero-vector modulation is a learned constant — the head must see it
# during training so serving with an omitted question stays well-behaved.
EMPTY_QUESTION_FRACTION = 0.05


def default_question_text(qid: str) -> str:
    """Canonical question text for a qid (entry 0 of QUESTION_PHRASINGS)."""
    phrasings = QUESTION_PHRASINGS.get(qid)
    return phrasings[0] if phrasings else qid


def _generate_choice_variants(
    base_options: list[str],
    qid: str,
    rng: Random,
    num_variants: int,
) -> list[dict]:
    """Generate variant option sets for a choice question.

    Returns list of {"options": [...], "gold_map": {orig_opt: variant_opt}}.
    """
    synonym_map = CHOICE_SYNONYM_MAPS.get(qid, {})
    if not synonym_map:
        return []

    variants = []
    # Each "column" of synonyms becomes a variant
    num_cols = min(len(next(iter(synonym_map.values()))), num_variants)
    for col in range(num_cols):
        gold_map = {}
        options = []
        for orig_opt in base_options:
            syns = synonym_map.get(orig_opt, [orig_opt])
            variant_opt = syns[col] if col < len(syns) else orig_opt
            gold_map[orig_opt] = variant_opt
            options.append(variant_opt)
        # Shuffle option order (head must learn permutation invariance)
        idx = list(range(len(options)))
        rng.shuffle(idx)
        shuffled_opts = [options[i] for i in idx]
        shuffled_map = {
            orig: shuffled_opts[idx.index(i)]
            for i, orig in enumerate(gold_map)
        }
        variants.append({"options": shuffled_opts, "gold_map": shuffled_map})

    return variants


def _generate_score_variants(
    base_levels: list[str],
    qid: str,
    rng: Random,
    num_variants: int,
) -> list[dict]:
    """Generate variant level sets for a score question.

    Each variant has a different number of levels and/or different labels.
    """
    level_variants = SCORE_SYNONYM_MAPS.get(qid, [])
    if not level_variants:
        return []

    variants = []
    for level_set in level_variants[:num_variants]:
        # Map base level index → variant level index via proportional scaling
        base_n = len(base_levels)
        var_n = len(level_set)
        gold_map = {}
        for base_idx in range(base_n):
            # Scale: base_idx ∈ [0, base_n-1] → var_idx ∈ [0, var_n-1]
            var_idx = min(int(round(base_idx * (var_n - 1) / max(base_n - 1, 1))), var_n - 1)
            gold_map[str(base_idx)] = var_idx
        variants.append({"levels": list(level_set), "gold_map": gold_map})

    return variants


def generate_variant_questions(
    base_spec: dict,
    rng: Random | None = None,
    num_choice_variants: int = 3,
    num_score_variants: int = 3,
) -> list[dict]:
    """Generate varied question configs from the base question spec.

    Returns list of dynamic question configs. Each config has the same
    question type as the base but with different option/level labels.
    """
    if rng is None:
        rng = Random(42)

    variants = []
    for qid, spec in base_spec.items():
        kind = spec["type"]
        if kind == "choice":
            choice_variants = _generate_choice_variants(
                spec["options"], qid, rng, num_choice_variants,
            )
            for v in choice_variants:
                variants.append({
                    "qid": qid,
                    "base_type": "choice",
                    "question": make_choice_question(
                        v["options"], rng.choice(QUESTION_PHRASINGS.get(qid, [qid])),
                    ),
                    "gold_map": v["gold_map"],
                    "is_variant": True,
                })
        elif kind == "score":
            score_variants = _generate_score_variants(
                spec["levels"], qid, rng, num_score_variants,
            )
            for v in score_variants:
                variants.append({
                    "qid": qid,
                    "base_type": "score",
                    "question": make_score_question(
                        v["levels"], rng.choice(QUESTION_PHRASINGS.get(qid, [qid])),
                    ),
                    "gold_map": v["gold_map"],
                    "is_variant": True,
                })

    return variants


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
    """Generate (state, options, gold) training samples.

    For each state and each question (base + variants), produces one sample
    with option embeddings and a gold label index.

    Args:
        states: list of TextState objects with gold labels.
        field_features: per-state (M_i, input_dim) field-set embeddings,
            aligned with ``states`` (v1 pooled vectors = M_i == 1).
        base_spec: normalized question bank from build_question_spec(cfg.questions).
        server: LlamaServer (must be running) for embedding option texts.
        rng: random state for shuffling.
        num_variants_per_question: how many option-set variants to generate.
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

    # Generate variant question configs
    variants = generate_variant_questions(base_spec, rng, num_variants_per_question, num_variants_per_question)

    # Build base question configs
    base_questions = []
    for qid, spec in base_spec.items():
        kind = spec["type"]
        if kind == "choice":
            base_questions.append({
                "qid": qid,
                "base_type": "choice",
                "question": make_choice_question(spec["options"], default_question_text(qid)),
                "gold_map": None,  # no mapping needed (1:1)
            })
        elif kind == "score":
            base_questions.append({
                "qid": qid,
                "base_type": "score",
                "question": make_score_question(spec["levels"], default_question_text(qid)),
                "gold_map": None,
            })
        elif kind == "noul":
            base_questions.append({
                "qid": qid,
                "base_type": "noul",
                "question": make_noul_question(spec.get("question", qid)),
                "gold_map": None,
            })

    all_question_configs = base_questions + variants

    # Collect all unique option texts AND question texts and embed them
    all_option_texts: set[str] = set()
    # Always include false/true for noul binary head (they're implied labels)
    all_option_texts.update(["false", "true"])
    for qc in all_question_configs:
        q = qc["question"]
        if q["type"] != "noul":
            for t in question_option_texts(q):
                all_option_texts.add(t)
        # v3: question strings ride the same embed pass (FiLM-modulate queries)
        if q.get("question"):
            all_option_texts.add(q["question"])

    # Embed all option/question texts in one batch
    unique_texts = sorted(all_option_texts)
    print(f"  embedding {len(unique_texts)} unique option/question texts...")
    text_to_emb = {}
    if unique_texts:
        batch_size = 32
        for i in range(0, len(unique_texts), batch_size):
            batch = unique_texts[i : i + batch_size]
            embs = server.embed(batch)
            for text, emb in zip(batch, embs):
                text_to_emb[text] = np.array(emb, dtype=np.float32)

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
        emb_by_text: dict[str, np.ndarray] = {}
        batch_size = 32
        for i in range(0, len(unique), batch_size):
            batch = unique[i : i + batch_size]
            embs = server.embed(batch)
            for text, emb in zip(batch, embs):
                emb_by_text[text] = np.array(emb, dtype=np.float32)
        for state_index, field_texts in pending:
            extra.append((
                np.stack([emb_by_text[t] for t in field_texts]),
                state_index,
            ))
        print(f"  shape augmentation: {len(extra)} variant states added")

    samples = []

    def emit(i: int, state_fields_emb: np.ndarray) -> None:
        """Emit one sample per question config for one state embedding."""
        state = states[i]
        for qc in all_question_configs:
            q = qc["question"]
            qid = qc["qid"]
            gold = state.labels.get(qid)

            if gold is None:
                continue  # question not applicable to this state

            kind = q["type"]
            if kind == "noul":
                gold_idx = 1 if bool(gold) else 0
                # noul goes through the attention path with [false, true]
                opt_embs = np.stack([text_to_emb["false"], text_to_emb["true"]])
            else:
                # Map gold label to variant option index
                gold_map = qc.get("gold_map")
                if gold_map is not None:
                    mapped_gold = gold_map[str(gold)]
                else:
                    mapped_gold = gold

                option_texts = question_option_texts(q)
                if kind == "choice":
                    # gold is option key → find its index in the variant's option list
                    gold_idx = option_texts.index(str(mapped_gold))
                else:
                    # gold is level index → mapped_gold is the variant level index
                    gold_idx = int(mapped_gold)

                opt_embs = np.stack([text_to_emb[t] for t in option_texts])

            # v3: mostly attach the question embedding; a small fraction is
            # the empty question (None → zero-vector modulation at forward).
            if rng.random() < EMPTY_QUESTION_FRACTION:
                question_text = ""
                question_emb = None
            else:
                question_text = q.get("question", "")
                # noul falls back to qid when the spec has no question text
                if not question_text:
                    question_text = default_question_text(qid)
                question_emb = text_to_emb[question_text]  # by reference

            samples.append(DynamicTrainingSample(
                state_fields_emb=state_fields_emb,
                option_embs=opt_embs,
                gold_idx=gold_idx,
                question_type=kind,
                question_text=question_text,
                question_emb=question_emb,
                is_variant=bool(qc.get("is_variant", False)),
            ))

    for i in range(len(states)):
        emit(i, field_features[i])
    for field_emb, state_index in extra:
        # Shape variants ride in both curriculum phases (is_variant=False):
        # shape invariance is part of the anchor task, not a generalization
        # axis saved for phase 2.
        emit(state_index, field_emb)

    print(f"  generated {len(samples)} training samples "
          f"({len(base_questions)} base + {len(variants)} variant questions × "
          f"{len(states) + len(extra)} states incl. shape variants)")
    return samples


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

    # Separate base vs variant samples for curriculum (explicit flag —
    # question texts no longer carry the "(variant)" marker)
    base_samples = [s for s in samples if not s.is_variant]
    variant_samples = [s for s in samples if s.is_variant]
    print(f"  anchor samples: {len(base_samples)}, variant samples: {len(variant_samples)}")

    # Curriculum pools: base vs variant samples
    is_variant = torch.tensor([s.is_variant for s in samples], dtype=torch.bool)
    base_positions = torch.nonzero(~is_variant).squeeze(-1)

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

    # Anchor pool: base-question samples restricted to the train split
    variant_arr = np.array([s.is_variant for s in samples], dtype=bool)
    in_train = np.zeros(n, dtype=bool)
    in_train[rest_idx[:split]] = True
    anchor_pool = torch.tensor(
        np.where(~variant_arr & in_train)[0], dtype=torch.long,
    )

    print(f"  train: {len(train_positions)}, val: {len(val_positions)}, "
          f"calibration holdout: {calib_n} "
          f"(anchor pool: {len(anchor_pool)})")

    # Model — v2 field-set architecture
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
        # Curriculum: first anchor_epochs use base samples only. Both phases
        # train on the train split only — never on val/calibration samples.
        if epoch <= anchor_epochs and len(anchor_pool) > 0:
            pool = anchor_pool
            phase = "anchor"
        else:
            pool = train_positions
            phase = "generalize"

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

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for start in range(0, len(val_positions), batch_size):
                xb, mask, ob, qb, yb = collate_dynamic_batch(
                    batch_at(val_positions, start, batch_size), device,
                )
                scores = model.forward_choice(xb, ob, mask, qb)
                pred = scores.argmax(dim=1)
                val_correct += int((pred == yb).sum().item())
                val_total += len(yb)

        val_acc = val_correct / val_total if val_total > 0 else 0.0

        # Escape guard: breakout-domain training can sit at a near-symmetric
        # plateau (all options reading the same field) whose escape is a
        # stochastic bootstrap event. A healthy run has escaped by a few
        # epochs into the generalize phase (val_acc jumps ~0.52 → ~0.80);
        # if it hasn't, further epochs are wasted — abort so the caller can
        # retry with a different init instead of burning the full schedule.
        guard_epoch = anchor_epochs + 10
        if epoch == guard_epoch and val_acc < 0.65:
            print(f"  ESCAPE GUARD: val_acc={val_acc:.3f} at epoch {epoch} — "
                  f"still at the symmetric plateau, aborting (caller should retry "
                  f"with a different init)")
            return None

        if epoch == 1 or epoch % 10 == 0 or epoch == hc.max_epochs:
            print(f"  epoch {epoch:3d} [{phase}]: train_loss={train_loss:.4f}  val_acc={val_acc:.3f}")

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