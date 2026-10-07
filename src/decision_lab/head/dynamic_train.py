"""Train DynamicDecisionHead with curriculum: anchor → generalize.

Phase 1 (anchor): Train on the fixed question bank to establish a baseline.
Phase 2 (generalize): Introduce varied option sets so the head learns that
  options are inputs, not architectural constants.

Training samples are (state_embedding, option_embeddings, gold_idx) tuples.
Option embeddings are pre-computed via the frozen backbone and cached.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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

    state_emb:     (input_dim,) float32 vector
    option_embs:   (n_opts, input_dim) float32 — embedded option texts
    gold_idx:      int — which option is correct
    question_type: "choice" | "score" | "noul"
    question_text: str — for logging
    """

    state_emb: np.ndarray
    option_embs: np.ndarray
    gold_idx: int
    question_type: str
    question_text: str = ""


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
}

SCORE_SYNONYM_MAPS: dict[str, list[list[str]]] = {
    "quality": [
        ["Terrible", "Poor", "Average", "Good", "Excellent"],
        ["1-star", "2-star", "3-star", "4-star", "5-star"],
        ["Unacceptable", "BelowAvg", "Acceptable", "AboveAvg", "Outstanding"],
        ["F", "D", "C", "B", "A"],
    ],
}


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
                    "question": make_choice_question(v["options"], f"{qid} (variant)"),
                    "gold_map": v["gold_map"],
                })
        elif kind == "score":
            score_variants = _generate_score_variants(
                spec["levels"], qid, rng, num_score_variants,
            )
            for v in score_variants:
                variants.append({
                    "qid": qid,
                    "base_type": "score",
                    "question": make_score_question(v["levels"], f"{qid} (variant)"),
                    "gold_map": v["gold_map"],
                })

    return variants


# ---------------------------------------------------------------------------
# Training data generation
# ---------------------------------------------------------------------------


def generate_dynamic_training_data(
    states,
    features: np.ndarray,
    base_spec: dict,
    server,
    rng: Random | None = None,
    num_variants_per_question: int = 3,
) -> list[DynamicTrainingSample]:
    """Generate (state, options, gold) training samples.

    For each state and each question (base + variants), produces one sample
    with option embeddings and a gold label index.

    Args:
        states: list of TextState objects with gold labels.
        features: (N, input_dim) pre-extracted state embeddings.
        base_spec: normalized question bank from build_question_spec(cfg.questions).
        server: LlamaServer (must be running) for embedding option texts.
        rng: random state for shuffling.
        num_variants_per_question: how many option-set variants to generate.

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
                "question": make_choice_question(spec["options"], qid),
                "gold_map": None,  # no mapping needed (1:1)
            })
        elif kind == "score":
            base_questions.append({
                "qid": qid,
                "base_type": "score",
                "question": make_score_question(spec["levels"], qid),
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

    # Collect all unique option texts and embed them
    all_option_texts: set[str] = set()
    # Always include false/true for noul binary head (they're implied labels)
    all_option_texts.update(["false", "true"])
    for qc in all_question_configs:
        q = qc["question"]
        if q["type"] != "noul":
            for t in question_option_texts(q):
                all_option_texts.add(t)

    # Embed all option texts in one batch
    unique_texts = sorted(all_option_texts)
    print(f"  embedding {len(unique_texts)} unique option texts...")
    text_to_emb = {}
    if unique_texts:
        batch_size = 32
        for i in range(0, len(unique_texts), batch_size):
            batch = unique_texts[i : i + batch_size]
            embs = server.embed(batch)
            for text, emb in zip(batch, embs):
                text_to_emb[text] = np.array(emb, dtype=np.float32)

    samples = []
    for i, state in enumerate(states):
        state_emb = features[i]
        for qc in all_question_configs:
            q = qc["question"]
            qid = qc["qid"]
            gold = state.labels.get(qid)

            if gold is None:
                continue  # question not applicable to this state

            kind = q["type"]
            if kind == "noul":
                gold_idx = 1 if bool(gold) else 0
                # noul uses dedicated head — option embeddings are [false, true]
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

            samples.append(DynamicTrainingSample(
                state_emb=state_emb,
                option_embs=opt_embs,
                gold_idx=gold_idx,
                question_type=kind,
                question_text=q.get("question", qid),
            ))

    print(f"  generated {len(samples)} training samples "
          f"({len(base_questions)} base + {len(variants)} variant questions × {len(states)} states)")
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
    hc = cfg.head
    device = get_device()
    print(f"Training dynamic head on device: {device} ({len(samples)} samples)")

    # Separate base vs variant samples for curriculum
    base_samples = [s for s in samples if "variant" not in s.question_text]
    variant_samples = [s for s in samples if "variant" in s.question_text]
    print(f"  anchor samples: {len(base_samples)}, variant samples: {len(variant_samples)}")

    # Pack into tensors
    def pack_samples(s_list: list[DynamicTrainingSample]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (state_embs, option_embs_stack, gold_idxs, n_opts_per_sample)."""
        states = torch.tensor(np.stack([s.state_emb for s in s_list]), dtype=torch.float32)
        golds = torch.tensor([s.gold_idx for s in s_list], dtype=torch.long)

        # Pad option embeddings to max n_opts
        max_opts = max(s.option_embs.shape[0] for s in s_list)
        opt_dim = s_list[0].option_embs.shape[1]
        opt_padded = torch.zeros(len(s_list), max_opts, opt_dim, dtype=torch.float32)
        n_opts = torch.zeros(len(s_list), dtype=torch.long)
        for i, s in enumerate(s_list):
            n = s.option_embs.shape[0]
            opt_padded[i, :n] = torch.tensor(s.option_embs, dtype=torch.float32)
            n_opts[i] = n
        return states, opt_padded, golds, n_opts

    x_base, opts_base, y_base, n_base = pack_samples(base_samples)
    x_var, opts_var, y_var, n_var = pack_samples(variant_samples) if variant_samples else (
        torch.zeros(0, 1024), torch.zeros(0, 1, 1024), torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)
    )

    # 3-way split for calibration holdout
    n = len(samples)
    idx = np.random.RandomState(cfg.generator.seed).permutation(n)
    calib_n = max(1, int(n * cfg.calibration.holdout_fraction))
    calib_idx_set = set(idx[:calib_n].tolist())

    # Train/val split on remaining samples
    rest_idx = idx[calib_n:]
    split = int(len(rest_idx) * 0.8)
    train_idx_set = set(rest_idx[:split].tolist())
    val_idx_set = set(rest_idx[split:].tolist())

    # Build tensors for all samples together
    x_all, opts_all, y_all, n_all = pack_samples(list(samples))

    # Mask functions
    def mask_by_set(idx_set):
        return torch.tensor([i in idx_set for i in range(len(samples))], dtype=torch.bool)

    train_mask = mask_by_set(train_idx_set)
    val_mask = mask_by_set(val_idx_set)

    x_train, opts_train, y_train, n_train = x_all[train_mask], opts_all[train_mask], y_all[train_mask], n_all[train_mask]
    x_val, opts_val, y_val, n_val = x_all[val_mask], opts_all[val_mask], y_all[val_mask], n_all[val_mask]

    print(f"  train: {len(x_train)}, val: {len(x_val)}, calibration holdout: {calib_n}")

    # Model
    model = DynamicDecisionHead(
        input_dim=1024,
        hidden_dim=hc.hidden_dim,
        d_k=128,
        dropout=hc.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=hc.learning_rate, weight_decay=hc.weight_decay)

    anchor_epochs = max(1, int(hc.max_epochs * anchor_fraction))

    best_val_acc = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, hc.max_epochs + 1):
        # Curriculum: first anchor_epochs use base samples only
        if epoch <= anchor_epochs and len(base_samples) > 0:
            epoch_samples = base_samples
            x_ep, opts_ep, y_ep, n_ep = x_base, opts_base, y_base, n_base
            phase = "anchor"
        else:
            epoch_samples = list(samples)
            x_ep, opts_ep, y_ep, n_ep = x_all, y_all.new_tensor([]), y_all, n_all
            phase = "generalize"

        model.train()
        train_loss = 0.0
        # Shuffle for this epoch
        perm = torch.randperm(len(x_ep)) if phase == "anchor" else torch.randperm(len(x_all))
        if phase == "anchor":
            x_shuf, opts_shuf, y_shuf = x_ep[perm], opts_ep[perm], y_ep[perm]
        else:
            x_shuf, opts_shuf, y_shuf = x_all[perm], opts_all[perm], y_all[perm]

        batch_size = hc.batch_size
        for start in range(0, len(x_shuf), batch_size):
            end = min(start + batch_size, len(x_shuf))
            xb = x_shuf[start:end].to(device)
            ob = opts_shuf[start:end].to(device)
            yb = y_shuf[start:end].to(device)

            optimizer.zero_grad()
            # Forward pass: attention over options
            h = model.trunk(xb)
            q = model.query_proj(h)
            K = model.key_proj(ob)
            scores = torch.bmm(K, q.unsqueeze(-1)).squeeze(-1) / math.sqrt(model.d_k)

            loss = F.cross_entropy(scores, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * (end - start)

        train_loss /= len(x_shuf)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for start in range(0, len(x_val), batch_size):
                end = min(start + batch_size, len(x_val))
                xb = x_val[start:end].to(device)
                ob = opts_val[start:end].to(device)
                yb = y_val[start:end].to(device)

                h = model.trunk(xb)
                q = model.query_proj(h)
                K = model.key_proj(ob)
                scores = torch.bmm(K, q.unsqueeze(-1)).squeeze(-1) / math.sqrt(model.d_k)
                pred = scores.argmax(dim=1)
                val_correct += int((pred == yb).sum().item())
                val_total += end - start

        val_acc = val_correct / val_total if val_total > 0 else 0.0

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
    temperature = _fit_global_temperature(model, x_all, opts_all, y_all, calib_idx_set, device)

    save_dynamic_head(model, str(model_path), temperature=temperature)
    print(f"  saved to {model_path}  (best val_acc={best_val_acc:.3f}, T={temperature:.3f})")
    return model


def _fit_global_temperature(
    model: DynamicDecisionHead,
    x: torch.Tensor,
    opts: torch.Tensor,
    y: torch.Tensor,
    calib_idx_set: set,
    device: torch.device,
    lr: float = 1e-2,
    max_iter: int = 1000,
) -> float:
    """Fit a single scalar temperature on the calibration holdout set."""
    calib_mask = torch.tensor([i in calib_idx_set for i in range(len(x))], dtype=torch.bool)
    x_calib = x[calib_mask].to(device)
    opts_calib = opts[calib_mask].to(device)
    y_calib = y[calib_mask].to(device)

    if len(x_calib) == 0:
        return 1.0

    model.eval()
    with torch.no_grad():
        h = model.trunk(x_calib)
        q = model.query_proj(h)
        K = model.key_proj(opts_calib)
        logits = torch.bmm(K, q.unsqueeze(-1)).squeeze(-1) / math.sqrt(model.d_k)

    logits = logits.detach().to("cpu").clone()
    labels = y_calib.detach().to("cpu").clone()
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        t = log_t.exp().clamp_min(1e-3)
        loss = F.cross_entropy(logits / t, labels)
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