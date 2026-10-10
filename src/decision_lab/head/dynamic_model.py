"""Dynamic decision head: attention-based slot filling.

Instead of baking option counts into architecture (nn.Linear(h, n_opts)),
this head treats options as inputs — embedding them and using scaled
dot-product attention to score each option against the state representation.

This means a single trained head handles any number of options at inference
without retraining.

Two input modes, selected by the checkpoint arch flag ``state_set``:

v1 (state_set=False) — pooled-state attention (legacy, load-compatible):
    State embedding → trunk → query (d_k)
    Option embeddings → key projection → keys (n_opts × d_k)
    Logits = q · K^T / √d_k

v2 (state_set=True) — field-set cross-attention:
    The state is a SET of field embeddings (sentences, key:value leaves).
    Each option queries the field set, so a OOD token contaminates only the
    field it appears in, and the head can match "the ball is to the right"
    directly against the "right" option.
    Field embeddings → field_enc (shared) → field keys (M × d_k)
    Option embeddings → opt_query (per option, d_k)
    α_ij = softmax_fields(q_j · k_i / √d_k)
    z_j = Σ_i α_ij · field_repr_i → (B, N, hidden)
    Logits = score_head(z_j)  (Linear(hidden → 1), shared over options)

v3 (REMOVED) — question-conditioned FiLM queries AND reads:
    Modulated option queries and the attended summary with four
    zero-initialized Linear(question_emb) gates. Only ever trained on ~25
    fixed question texts (5 qids × a few handwritten phrasings), so a dense
    Linear(1024→d_k) was regulated on 25 points and free everywhere else —
    any unseen question target received a near-random modulation. Replaced
    by v4.

v4 (state_set=True, arch_version 4) — question-field attention:
    The question embedding participates in attention directly instead of
    modulating: it projects to its own cosine-attention query over the
    SAME field keys the options read, yielding a question context vector
    z_q ("which fields does this question point at"). Each option's
    attended summary z_j is then read out through a zero-initialized
    residual MLP on the concatenation:
        z_j' = z_j + qz_readout([z_j ; z_q])
    Attention is a nearest-neighbour op over backbone embeddings, so an
    unseen question's query lands near a trained question's embedding and
    selects the same fields — semantic similarity from the backbone flows
    through an actual computation path, which FiLM (a global dense map,
    unregulated off its 25 training points) could never do. The concat
    form is load-bearing: a plain additive z + g·W·z_q is constant across
    options and cancels in the softmax over options, while the concat MLP
    sees per-option z_j and can re-interpret attended content.
    The question's attended context z_q also SHIFTS each option's query
    before it reads the field set (q_query_shift, zero-init). This is
    load-bearing, not decorative: the same option text must match a
    different field under a different question (breakout "left" matches the
    ball's POSITION field for paddle_direction but with a question-agnostic
    query is pulled by the "moving right" MOTION field). Measured on the
    v4 head without it: paddle_direction 0.680 vs v3's 0.980, and 48.9%
    when the motion word disagreed with the position side. A per-field bias
    SHARED across options cannot substitute — small leaves the distractor
    winning, large homogenizes every option onto one field (identical
    logits, the symmetric plateau). Steering the queries is per-option, so
    it re-points the match without collapsing option discrimination.
    qz_readout is ZERO-initialized (weight AND bias): at init, output ==
    v2 forward for ANY question, preserving the O(1) cosine-attention logit
    spread (see __init__ note). Gradients still reach qz_readout at W=0
    (its input z_q is nonzero), but q_question/attn_scale_q receive no
    gradient until the readout moves off zero — the breakout
    paddle_direction/ball_motion pair (identical option texts, independent
    golds) is the forcing function that makes the loss NEED the question
    path; see the curriculum in dynamic_train.py.
    question_emb=None (empty question) falls back to a zero vector — a
    learned constant query, trained via ~5% empty-question samples.
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

    forward_choice(state, option_embs, state_mask=None) → (batch, n_opts) logits
    forward_score(state, level_embs, state_mask=None)   → (batch, n_levels) logits
    forward_noul(state)                                 → (batch, 2) logits (v1 only)

    state shapes (v2): (input_dim,) → 1 state, 1 field; (M, input_dim) →
    1 state with M fields; (B, M, input_dim) → batch of field sets.
    state_mask: (B, M) bool, True = valid field (v2 only; ignored in v1).

    v1 modules (trunk/query_proj/key_proj/noul_head) and v2+ modules
    (field_enc/field_key/opt_query/score_head/q_question/attn_scale_q/
    q_query_shift/qz_readout) are mutually exclusive: only the ones matching
    ``state_set`` exist.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        d_k: int = D_K,
        dropout: float = 0.1,
        state_set: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.d_k = d_k
        self.state_set = state_set

        if state_set:
            # v2 field-set cross-attention (shared option path)
            self.field_enc = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.field_key = nn.Linear(hidden_dim, d_k)
            self.opt_query = nn.Linear(input_dim, d_k)
            self.score_head = nn.Linear(hidden_dim, 1)
            # Learnable attention scale, init √d_k. Cosine attention (q, k
            # normalized) with this scale starts with O(1) logit spread —
            # with raw q·k/√d_k the initial logits are ~1e-3 (product of two
            # small random projections), softmax is uniform, every option
            # reads the same mean field, and training stalls at ln(n_max).
            self.attn_scale = nn.Parameter(torch.tensor(float(d_k) ** 0.5))
            # v4 question-field attention: the question reads the SAME field
            # keys via its own cosine-attention query. attn_scale_q is a
            # SEPARATE scale from attn_scale (sharing would couple the two
            # heads' softmax temperature).
            self.q_question = nn.Linear(input_dim, d_k)
            self.attn_scale_q = nn.Parameter(torch.tensor(float(d_k) ** 0.5))
            # v4 question-conditioned option queries: the question's attended
            # field context (z_q) shifts every option's query before it reads
            # the field set. This is the per-option steering the task needs:
            # the SAME option text must match a different field under a
            # different question (breakout "left" matches the ball's POSITION
            # field for paddle_direction but must not be pulled by the
            # "moving right" MOTION field). A per-field bias shared across
            # options cannot do that — measured: small bias leaves the
            # distractor winning (paddle_direction 0.68 vs v3's 0.98), large
            # bias homogenizes every option onto one field (identical logits
            # for all options, the symmetric plateau), and the learned scale
            # settles at roughly half attn_scale, i.e. between the two.
            # Zero-init (weight AND bias) → shift 0 → identity at init.
            self.q_query_shift = nn.Linear(hidden_dim, d_k)
            nn.init.zeros_(self.q_query_shift.weight)
            nn.init.zeros_(self.q_query_shift.bias)
            # Zero-init residual read-out: at init, output == v2 forward for
            # ANY question (stall-recurrence guard). LoRA-style bootstrap —
            # first Linear default-init, GELU, LAST Linear zero-init:
            # identity at init holds, the last layer's gradient is alive at
            # init (grad ⊗ GELU(first(x)) ≠ 0), and the first layer follows
            # once the last moves off zero.
            # The GELU nonlinearity is load-bearing: with a purely linear
            # read-out into the linear score_head, the question component
            # v·W2·z_q is constant across options and cancels in the softmax
            # over options — the question could never flip an argmax.
            self.qz_readout = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            nn.init.zeros_(self.qz_readout[2].weight)
            nn.init.zeros_(self.qz_readout[2].bias)
        else:
            # v1: pooled-state attention (legacy, kept for checkpoint compat)
            self.trunk = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.query_proj = nn.Linear(hidden_dim, d_k)
            self.key_proj = nn.Linear(input_dim, d_k)
            # Deprecated: never trained; kept so v1 state_dicts load.
            # noul is scored via the attention path with ["false", "true"].
            self.noul_head = nn.Linear(hidden_dim, 2)

    @staticmethod
    def _normalize_state_set(
        state_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Normalize a v2 state input → (B, M, D) and batch size."""
        if state_embedding.dim() == 1:
            return state_embedding.unsqueeze(0).unsqueeze(0), 1   # (D,)
        if state_embedding.dim() == 2:
            return state_embedding.unsqueeze(0), 1                # (M, D) unbatched
        return state_embedding, state_embedding.shape[0]

    def _forward_attention_set(
        self,
        state_embedding: torch.Tensor,
        option_embeddings: torch.Tensor,
        state_mask: torch.Tensor | None = None,
        question_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """v4 core: options read the field set; the question reads it too → (B, N).

        Cosine attention: q and k are L2-normalized before the dot product
        and scaled by the learnable ``attn_scale``/``attn_scale_q`` (init
        √d_k) — see the __init__ note for why raw q·k/√d_k stalls training.

        The option path is question-agnostic (an OOD question embedding can
        never corrupt it); the question re-interprets attended content only
        via the zero-init residual read-out — at init the output equals the
        v2 forward for ANY question.

        Args:
            question_emb: (input_dim,) or (batch, input_dim) question-text
                embedding, or None for an empty question (zero vector — a
                learned constant query, trained on empty-question samples).
        """
        state, batch_size = self._normalize_state_set(state_embedding)
        if option_embeddings.dim() == 2:
            option_embeddings = option_embeddings.unsqueeze(0)    # (1, N, D)
        if option_embeddings.shape[0] == 1 and batch_size > 1:
            option_embeddings = option_embeddings.expand(batch_size, -1, -1)

        # Question → (B, input_dim); None → zeros (empty-question fallback)
        if question_emb is None:
            question_emb = torch.zeros(
                batch_size, self.input_dim,
                device=state.device, dtype=state.dtype,
            )
        elif question_emb.dim() == 1:
            question_emb = question_emb.unsqueeze(0).expand(batch_size, -1)

        field_repr = self.field_enc(state)                        # (B, M, H)
        K = F.normalize(self.field_key(field_repr), dim=-1)       # (B, M, d_k)

        mask = None
        if state_mask is not None:
            mask = state_mask.unsqueeze(0) if state_mask.dim() == 1 else state_mask

        # v4 question context — the question queries the SAME field keys.
        # Computed FIRST: its attended content steers the option queries.
        Qq = F.normalize(self.q_question(question_emb), dim=-1)   # (B, d_k)
        attn_q = torch.bmm(K, Qq.unsqueeze(-1)).squeeze(-1) * self.attn_scale_q
        if mask is not None:
            attn_q = attn_q.masked_fill(~mask, float("-inf"))
        alpha_q = torch.softmax(attn_q, dim=-1)                   # over fields
        z_q = torch.bmm(alpha_q.unsqueeze(1), field_repr).squeeze(1)  # (B, H)

        # Option queries, steered by the question's attended context
        # (zero-init shift → identity at init). This is what lets the same
        # option text match a different field under a different question.
        shift = self.q_query_shift(z_q).unsqueeze(1)              # (B, 1, d_k)
        Q = F.normalize(
            self.opt_query(option_embeddings) + shift, dim=-1,
        )                                                         # (B, N, d_k)
        # (B, N, d_k) × (B, d_k, M) → (B, N, M)
        attn = torch.bmm(Q, K.transpose(1, 2)) * self.attn_scale
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1), float("-inf"))
        alpha = torch.softmax(attn, dim=-1)                       # over fields
        z = torch.bmm(alpha, field_repr)                          # (B, N, H)

        # Zero-init residual read-out (identity at init — see __init__ note).
        # Concat form is load-bearing: a plain additive z + g·W·z_q would be
        # constant across options and cancel in the softmax over options.
        z_q_exp = z_q.unsqueeze(1).expand(-1, z.size(1), -1)      # (B, N, H)
        z = z + self.qz_readout(torch.cat([z, z_q_exp], dim=-1))  # (B, N, H)
        return self.score_head(z).squeeze(-1)                     # (B, N)

    def _forward_attention(
        self,
        state_embedding: torch.Tensor,
        option_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """v1 core: score each option against the pooled state.

        Args:
            state_embedding: (input_dim,), (batch, input_dim), or (B, M, input_dim)
                with M == 1 (uniform pipeline — reads field 0).
            option_embeddings: (n_opts, input_dim) or (batch, n_opts, input_dim) —
                pre-embedded option texts.

        Returns:
            (batch, n_opts) logits (pre-softmax scores).
        """
        # Uniform pipeline passes field sets; legacy head reads the summary field.
        if state_embedding.dim() == 3:
            state_embedding = state_embedding[:, 0, :]
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
        state_mask: torch.Tensor | None = None,
        question_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score state against choice options → (batch, n_opts) logits.

        question_emb (v4 heads) is a field-selection query over the field
        set; ignored by v1 heads.
        """
        if self.state_set:
            return self._forward_attention_set(
                state_embedding, option_embeddings, state_mask, question_emb,
            )
        return self._forward_attention(state_embedding, option_embeddings)

    def forward_score(
        self,
        state_embedding: torch.Tensor,
        level_embeddings: torch.Tensor,
        state_mask: torch.Tensor | None = None,
        question_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score state against rubric levels → (batch, n_levels) logits.

        question_emb (v4 heads) is a field-selection query over the field
        set; ignored by v1 heads.
        """
        if self.state_set:
            return self._forward_attention_set(
                state_embedding, level_embeddings, state_mask, question_emb,
            )
        return self._forward_attention(state_embedding, level_embeddings)

    def forward_noul(self, state_embedding: torch.Tensor) -> torch.Tensor:
        """Binary classification: is this true/false? → (batch, 2) logits.

        Deprecated v1 path — never trained. In v2 (and in the training/
        serving pipeline) noul is scored via forward_choice with
        ["false", "true"] option embeddings.
        """
        if self.state_set:
            raise NotImplementedError(
                "v2 head scores noul via forward_choice with ['false', 'true'] "
                "option embeddings — see predict_dynamic"
            )
        if state_embedding.dim() == 3:
            state_embedding = state_embedding[:, 0, :]
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
        state_embedding: (input_dim,) single state vector (v1) or
            (M, input_dim) field set (v2).
        questions: list of dynamic question configs.
        option_embeddings_cache: {text: (input_dim,) embedding} — pre-computed
            embeddings for all option texts referenced by questions (including
            "false" and "true" for noul) AND for the question strings
            themselves (v4 question-field attention).
        head: trained DynamicDecisionHead in eval mode.

    Returns:
        {question_index: decoded_answer_dict}
    """
    head.eval()
    results = {}
    with torch.no_grad():
        for i, q in enumerate(questions):
            kind = q["type"]
            texts = question_option_texts(q)
            opt_embs = torch.stack(
                [option_embeddings_cache[t] for t in texts]
            )  # (n_opts, D)
            # v4: empty/missing question → None → zero-vector query
            qtext = (q.get("question") or "").strip()
            question_emb = (
                option_embeddings_cache[qtext] if qtext else None
            )
            scores = head.forward_choice(
                state_embedding, opt_embs, question_emb=question_emb,
            )
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
                "state_set": model.state_set,
                "arch_version": 4,
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
    """Load a DynamicDecisionHead from checkpoint (v4 arch only).

    Clean break: v1/v2/v3 checkpoints are refused — the v3 checkpoint is
    preserved in the main checkout's models/ (and backups); v4 requires
    its own retrain. No silent fallbacks.
    """
    checkpoint = torch.load(str(path), map_location="cpu")
    arch = checkpoint["arch"]
    if arch.get("type") != "dynamic":
        raise ValueError(
            f"Checkpoint at {path} is not a DynamicDecisionHead "
            f"(arch.type={arch.get('type', 'missing')})"
        )
    arch_version = arch.get("arch_version", 2)
    if arch_version < 4:
        raise ValueError(
            f"Checkpoint at {path} is arch_version {arch_version}; this "
            f"build only loads v4 question-attention heads. v1-v3 "
            f"checkpoints are preserved (main checkout models/ + backups) "
            f"but require their own release to serve — no silent upgrade."
        )
    if not arch.get("state_set", False):
        raise ValueError(
            f"Checkpoint at {path} is a v4 non-state_set head — v4 implies "
            f"state_set=True (question-field attention requires field sets)."
        )
    model = DynamicDecisionHead(
        input_dim=arch.get("input_dim", 1024),
        hidden_dim=arch.get("hidden_dim", 256),
        d_k=arch.get("d_k", D_K),
        state_set=True,
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