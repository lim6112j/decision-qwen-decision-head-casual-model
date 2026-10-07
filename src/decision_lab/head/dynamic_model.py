"""Dynamic decision head: attention-based slot filling.

Instead of baking option counts into architecture (nn.Linear(h, n_opts)),
this head treats options as inputs — embedding them and using scaled
dot-product attention to score each option against the state representation.

State embedding → trunk → query (d_k)
Option embeddings → key projection → keys (n_opts × d_k)
Logits = q · K^T / √d_k

This means a single trained head handles any number of options at inference
without retraining.
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

D_K = 128
DEFAULT_TEMPERATURE = 1.0

# ---------------------------------------------------------------------------
# Dynamic question configuration helpers
# ---------------------------------------------------------------------------


def make_choice_question(options: list[str], question_text: str = "") -> dict:
    """Build a dynamic choice question config.

    Args:
        options: list of option label strings, e.g. ["first", "second", "third"].
        question_text: optional prompt describing what's being decided.

    Returns:
        {"type": "choice", "options": [...], "question": "..."}
    """
    return {"type": "choice", "options": list(options), "question": question_text}


def make_score_question(levels: list[str], question_text: str = "") -> dict:
    """Build a dynamic score question config.

    Args:
        levels: ordered rubric labels, e.g. ["Bottom", "Low", "Mid", "High", "Top"].
        question_text: optional prompt.

    Returns:
        {"type": "score", "levels": [...], "question": "..."}
    """
    return {"type": "score", "levels": list(levels), "question": question_text}


def make_noul_question(question_text: str) -> dict:
    """Build a dynamic noul (boolean) question config.

    Args:
        question_text: the yes/no question, e.g. "Is this actionable?".

    Returns:
        {"type": "noul", "question": "..."}
    """
    return {"type": "noul", "question": question_text}


def question_option_texts(question: dict) -> list[str]:
    """Return the list of option/level strings a dynamic question expects."""
    kind = question["type"]
    if kind == "choice":
        return list(question["options"])
    if kind == "score":
        return list(question["levels"])
    if kind == "noul":
        return ["false", "true"]
    raise ValueError(f"unknown question type '{kind}'")


def question_num_classes(question: dict) -> int:
    """Number of output classes for a dynamic question."""
    kind = question["type"]
    if kind == "noul":
        return 2
    if kind == "choice":
        return len(question["options"])
    return len(question["levels"])


# ---------------------------------------------------------------------------
# DynamicDecisionHead
# ---------------------------------------------------------------------------


class DynamicDecisionHead(nn.Module):
    """Attention-based decision head: options are inputs, not architecture.

    Architecture:
        trunk:     input_dim → hidden_dim  (shared state processor)
        query_proj: hidden_dim → d_k        (state → query)
        key_proj:  input_dim → d_k          (option embedding → key)
        noul_head: hidden_dim → 2           (dedicated binary head)

    forward_choice(state_emb, option_embs) → (batch, n_opts) logits
    forward_score(state_emb, level_embs)   → (batch, n_levels) logits
    forward_noul(state_emb)                → (batch, 2) logits
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        d_k: int = D_K,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.d_k = d_k

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.query_proj = nn.Linear(hidden_dim, d_k)
        self.key_proj = nn.Linear(input_dim, d_k)
        self.noul_head = nn.Linear(hidden_dim, 2)

    def _forward_attention(
        self,
        state_embedding: torch.Tensor,
        option_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Core attention: score each option against the state.

        Args:
            state_embedding: (batch, input_dim) or (input_dim,) — state vector.
            option_embeddings: (n_opts, input_dim) or (batch, n_opts, input_dim) —
                pre-embedded option texts.

        Returns:
            (batch, n_opts) logits (pre-softmax scores).
        """
        # Normalize shapes: state → (B, D), options → (B, N, D)
        if state_embedding.dim() == 1:
            state_embedding = state_embedding.unsqueeze(0)          # (1, D)
        if option_embeddings.dim() == 2:
            # (N, D) — broadcast across batch dim of state
            option_embeddings = option_embeddings.unsqueeze(0)      # (1, N, D)

        batch_size = state_embedding.shape[0]
        n_opts = option_embeddings.shape[1]
        # Expand option embeddings if state batch > 1 (broadcast)
        if option_embeddings.shape[0] == 1 and batch_size > 1:
            option_embeddings = option_embeddings.expand(batch_size, n_opts, -1)

        h = self.trunk(state_embedding)                              # (B, H)
        q = self.query_proj(h)                                       # (B, d_k)
        K = self.key_proj(option_embeddings)                         # (B, N, d_k)

        # Scaled dot-product: (B, N, d_k) × (B, d_k, 1) → (B, N)
        scores = torch.bmm(K, q.unsqueeze(-1)).squeeze(-1)
        return scores / math.sqrt(self.d_k)

    def forward_choice(
        self,
        state_embedding: torch.Tensor,
        option_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Score state against choice options → (batch, n_opts) logits."""
        return self._forward_attention(state_embedding, option_embeddings)

    def forward_score(
        self,
        state_embedding: torch.Tensor,
        level_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Score state against rubric levels → (batch, n_levels) logits."""
        return self._forward_attention(state_embedding, level_embeddings)

    def forward_noul(self, state_embedding: torch.Tensor) -> torch.Tensor:
        """Binary classification: is this true/false? → (batch, 2) logits."""
        if state_embedding.dim() == 1:
            state_embedding = state_embedding.unsqueeze(0)
        h = self.trunk(state_embedding)
        return self.noul_head(h)


# ---------------------------------------------------------------------------
# Decoding: attention scores → probabilities → typed answers
# ---------------------------------------------------------------------------


def dynamic_probabilities(
    scores: torch.Tensor,
    temperature: float = DEFAULT_TEMPERATURE,
) -> torch.Tensor:
    """Softmax attention scores → probability distribution."""
    t = max(float(temperature), 1e-3)
    return F.softmax(scores / t, dim=-1)


def decode_dynamic_answer(
    question: dict,
    scores: torch.Tensor,
    temperature: float = DEFAULT_TEMPERATURE,
) -> dict:
    """Decode attention scores into a typed answer dict.

    Args:
        question: dynamic question config {"type", "options"|"levels"|...}.
        scores: (n_opts,) or (1, n_opts) logits tensor.
        temperature: softmax temperature.

    Returns:
        {"predicted": ..., "distribution": {...}, "confidence": ...}
        For score: also {"expected": float}.
    """
    kind = question["type"]
    row = scores.detach()
    if row.dim() == 2:
        row = row[0]

    if kind == "noul":
        # noul uses dedicated binary head; scores are 2-class logits
        probs = dynamic_probabilities(row, temperature)
        p_false = float(probs[0].item())
        p_true = float(probs[1].item())
        return {
            "predicted": p_true >= 0.5,
            "distribution": {"true": p_true, "false": p_false},
            "confidence": max(p_true, p_false),
        }

    probs = dynamic_probabilities(row, temperature)
    conf = float(probs.max().item())

    if kind == "choice":
        options = question["options"]
        idx = int(probs.argmax().item())
        return {
            "predicted": options[idx],
            "distribution": {opt: float(p) for opt, p in zip(options, probs)},
            "confidence": conf,
        }

    if kind == "score":
        levels = question["levels"]
        idx = int(probs.argmax().item())
        indices = torch.arange(len(levels), dtype=probs.dtype, device=probs.device)
        expected = float((probs * indices).sum().item())
        return {
            "predicted": idx,
            "expected": expected,
            "distribution": {lvl: float(p) for lvl, p in zip(levels, probs)},
            "confidence": conf,
        }

    raise ValueError(f"unknown question type '{kind}'")


def predict_dynamic(
    state_embedding: torch.Tensor,
    questions: list[dict],
    option_embeddings_cache: dict[str, torch.Tensor],
    head: "DynamicDecisionHead",
    temperature: float = DEFAULT_TEMPERATURE,
) -> dict[int, dict]:
    """Run the head on one state against a list of dynamic questions.

    Args:
        state_embedding: (input_dim,) single state vector.
        questions: list of dynamic question configs.
        option_embeddings_cache: {option_text: (input_dim,) embedding} —
            pre-computed embeddings for all option texts referenced by questions.
        head: trained DynamicDecisionHead in eval mode.

    Returns:
        {question_index: decoded_answer_dict}
    """
    head.eval()
    results = {}
    with torch.no_grad():
        for i, q in enumerate(questions):
            kind = q["type"]
            if kind == "noul":
                scores = head.forward_noul(state_embedding)
            else:
                texts = question_option_texts(q)
                opt_embs = torch.stack(
                    [option_embeddings_cache[t] for t in texts]
                )  # (n_opts, D)
                if kind == "choice":
                    scores = head.forward_choice(state_embedding, opt_embs)
                else:
                    scores = head.forward_score(state_embedding, opt_embs)
            results[i] = decode_dynamic_answer(q, scores, temperature)
    return results


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------


def is_correct_dynamic(question: dict, predicted, gold) -> bool:
    """Compare a prediction with gold label for a dynamic question.

    predicted=None (parse failure) is always incorrect.
    """
    if predicted is None:
        return False
    kind = question["type"]
    if kind == "choice":
        return str(predicted).lower() == str(gold).lower()
    if kind == "score":
        return int(predicted) == int(gold)
    if kind == "noul":
        return bool(predicted) == bool(gold)
    raise ValueError(f"unknown question type '{kind}'")


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def save_dynamic_head(
    model: DynamicDecisionHead,
    path: str | Path,
    temperature: float = DEFAULT_TEMPERATURE,
) -> None:
    """Save a DynamicDecisionHead checkpoint (self-describing, no question_spec)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "arch": {
                "type": "dynamic",
                "input_dim": model.input_dim,
                "hidden_dim": model.hidden_dim,
                "d_k": model.d_k,
            },
            "temperature": temperature,
        },
        str(path),
    )


def load_dynamic_head(
    path: str | Path,
    device: torch.device | None = None,
) -> DynamicDecisionHead:
    """Load a DynamicDecisionHead from checkpoint."""
    checkpoint = torch.load(str(path), map_location="cpu")
    arch = checkpoint["arch"]
    if arch.get("type") != "dynamic":
        raise ValueError(
            f"Checkpoint at {path} is not a DynamicDecisionHead "
            f"(arch.type={arch.get('type', 'missing')})"
        )
    model = DynamicDecisionHead(
        input_dim=arch.get("input_dim", 1024),
        hidden_dim=arch.get("hidden_dim", 256),
        d_k=arch.get("d_k", D_K),
    )
    model.load_state_dict(checkpoint["model_state"])
    if device is not None:
        model = model.to(device)
    model.temperature = checkpoint.get("temperature", DEFAULT_TEMPERATURE)
    return model


def create_random_dynamic_head(**kwargs) -> DynamicDecisionHead:
    """Create a DynamicDecisionHead with random init weights (no training)."""
    return DynamicDecisionHead(**kwargs)


def get_device() -> torch.device:
    """Best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")