# Decision Head Dynamic Values — Design Proposal

**Status:** Draft  
**Date:** 2026-10-07  
**Author:** Ben Lim

---

## 1. Problem Statement

### 1.1 Current Behavior

The `TypedDecisionHead` has three question types — `choice`, `score`, `noul` — but the **values** (options, levels, labels) for each question are **baked into the model architecture at training time** and cannot change at inference.

| Question | Type | Values baked into architecture |
|---|---|---|
| `sentiment` | choice | `positive`, `negative`, `neutral` → 3-output linear head |
| `urgency` | choice | `low`, `medium`, `high`, `critical` → 4-output linear head |
| `quality` | score | `Poor`, `Fair`, `Good`, `Excellent` → 4-output linear head |
| `is_actionable` | noul | boolean → 2-output linear head |

This means:
- You cannot ask `sentiment` with different labels (e.g. `optimistic`/`pessimistic`) without retraining the whole model
- You cannot ask a new `choice` question with 5 options without retraining
- You cannot change score levels from 4-point to 7-point without retraining
- The output dimensionality (`nn.Linear(hidden_dim, num_classes)`) is fixed per question

### 1.2 Desired Behavior

At inference time, the user should be able to **parameterize each decision with its specific values**:

```python
# Choice: specify the options dynamically
head.decide(state, question={
    "type": "choice",
    "options": ["first", "second", "third"],
    "option_descriptions": {
        "first": "The first candidate approach",
        "second": "The second candidate approach",
        "third": "The third candidate approach",
    }
})

# Score: specify the range/labels dynamically
head.decide(state, question={
    "type": "score",
    "levels": ["Bottom", "Low", "Mid", "High", "Top"]  # 5-level Likert
})

# Noul: already dynamic (always binary true/false)
head.decide(state, question={
    "type": "noul",
    "question": "Is this document ready for publication?"
})
```

### 1.3 Why This Matters

1. **Flexibility**: One trained head should handle many different questions without retraining
2. **Cost**: Avoids retraining when question definitions change
3. **Generalization**: The head learns the *decision-making capability* (how to map state → option), not specific option semantics
4. **Production use**: In real applications, the question bank changes over time — new categories, revised rubrics, different scales

---

## 2. Current Architecture Analysis

### 2.1 Data Flow

```
┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐    ┌──────────────┐
│  Text State  │───▶│  Qwen Backbone  │───▶│  TypedDecision   │───▶│  Decoded     │
│  (document)  │    │  (frozen)       │    │  Head (trained)  │    │  Answers     │
└──────────────┘    └─────────────────┘    └──────────────────┘    └──────────────┘
                           │                       │
                    1024-dim embedding      {qid: logits}
                                           logits shape = (1, num_classes_qid)
```

### 2.2 Where Values Are Hard-Coded

| Layer | File | What's hard-coded |
|---|---|---|
| **Config** | `config.py:19-29` | `QuestionsConfig` dataclass with fixed `choice`/`score`/`noul` dicts |
| **Config** | `configs/default.yaml:10-29` | Every option, level, and question text |
| **Model init** | `head/model.py:64-74` | `nn.Linear(hidden_dim, len(options))` — output size fixed to option count |
| **Model forward** | `head/model.py:80-90` | Iterates fixed `ModuleDict` keys |
| **Decoding** | `head/model.py:118-161` | `decode_answer` maps index → option key using spec |
| **Training** | `head/train.py:136-161` | `encode_labels` maps option keys → indices using fixed option list |
| **Data generation** | `states/generator.py:30-46` | `Latents` dataclass with fixed fields matching question bank |
| **Prompt LM** | `prompt_lm/parser.py:25-60` | Validates against fixed option lists in `question_spec` |
| **Web UI** | `webapp/app.py:157-160` | `/api/questions` returns fixed spec from config |
| **Checkpoint** | `head/model.py:211-238` | `question_spec` saved in checkpoint, used to reconstruct model |

### 2.3 The Fundamental Constraint

```python
# head/model.py:68 — THIS is the constraint
choice_heads[qid] = nn.Linear(hidden_dim, len(s["options"]))
```

A `nn.Linear(256, 3)` layer cannot become `nn.Linear(256, 5)` without creating a new layer and training it. The number of output neurons is welded into the weight matrix shape.

---

## 3. Design Options

### 3.1 Option A: Token Concatenation (Prompt-Injection)

**How it works:** Instead of separate model inputs for state vs. options, embed the options directly into the text that the backbone processes. The head then operates on a "state + question + options" embedding.

```
Input text: "<state_text>\n\nQuestion: <q_text>\nOptions:\nA: <opt1>\nB: <opt2>\nC: <opt3>"
     │
     ▼
┌─────────────────┐    ┌──────────────────┐    ┌──────────────┐
│  Qwen Backbone  │───▶│  Fixed-Size Head │───▶│  Answers     │
│  (frozen)       │    │  (e.g. 3-class)  │    │              │
└─────────────────┘    └──────────────────┘    └──────────────┘
```

**Head architecture:** Train with a **fixed maximum** number of options (e.g., 10). At inference, pad unused slots. The backbone sees option text in context so the embedding encodes option semantics.

**Pros:**
- Minimal code changes — only the text formatting changes
- Backbone naturally fuses state + option semantics
- No new neural architecture needed

**Cons:**
- The head still has a fixed maximum (10 options). 11 options = retrain
- Option ordering in the prompt affects the embedding (positional bias)
- Backbone context window consumed by option text
- Hard to get calibrated per-option probabilities — the head just picks slot N
- Training must cover varied option counts (0-padded) to learn to ignore padding

**Verdict:** Simplest to implement but least flexible. Good as a quick win but doesn't solve the fundamental fixed-output-size problem.

---

### 3.2 Option B: Attention-Based Slot Filling (Recommended)

**How it works:** The head becomes an **attention mechanism** where:
- The **state embedding** is projected to a **query vector** `q` (shape: `d_k`)
- Each **option value** is embedded and projected to a **key vector** `k_i` (shape: `d_k`)
- Logit for option `i` = scaled dot-product `q · k_i / √d_k`
- Softmax over all options gives a probability distribution

```
                    ┌──────────────────────────┐
State ──▶ Qwen ──▶ │  Query Projection        │──▶ q (d_k)
                    │  nn.Linear(1024, d_k)    │
                    └──────────────────────────┘
                                                    ┌──────────────┐
                    ┌──────────────────────────┐    │  Attention   │
Option texts ─────▶│  Option Embedder          │───▶│  q · K^T     │──▶ logits ──▶ softmax
  ["first",         │  (lightweight or Qwen)   │    │  / √d_k      │
   "second",        │  → K matrix (n_opts, d_k)│    └──────────────┘
   "third"]         └──────────────────────────┘
```

**Architecture details:**

```python
class DynamicDecisionHead(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256, d_k=128):
        # Shared trunk (unchanged)
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        # Query projection from state representation
        self.query_proj = nn.Linear(hidden_dim, d_k)
        # Key projection from option embeddings
        self.key_proj = nn.Linear(input_dim, d_k)  # reuse backbone dim for option embeddings

    def forward(self, state_embedding, option_embeddings, question_type):
        """
        state_embedding: (batch, input_dim)
        option_embeddings: (batch, n_options, input_dim) — embedded option texts
        question_type: "choice" | "score" | "noul"
        """
        h = self.trunk(state_embedding)           # (B, hidden_dim)
        q = self.query_proj(h)                    # (B, d_k)
        K = self.key_proj(option_embeddings)      # (B, n_opts, d_k)
        
        # Scaled dot-product attention
        scores = torch.bmm(K, q.unsqueeze(-1)).squeeze(-1) / math.sqrt(d_k)
        # scores: (B, n_opts) — these are the logits
        
        return scores  # can pass to softmax + temperature scaling as before
```

**For `score` type:** Levels are treated like ordered options. The attention mechanism produces a distribution, and the expected value is computed as before.

**For `noul` type:** Always binary — keep a dedicated 2-class head. The "true"/"false" values don't vary semantically.

**Training strategy:**
1. **Phase 1 (current):** Train on the fixed question bank as today — this anchors the head
2. **Phase 2 (dynamic):** Generate training data with *varied option sets* for choice questions and *varied level counts* for score questions. The same latent state gets labeled against different option configurations.

**Option embedding:** Use the frozen Qwen backbone to embed option texts. This is cheap (one embedding call per option set, cached) and reuses existing infrastructure.

**Pros:**
- **Truly dynamic:** Any number of options at inference without architecture change
- **No retraining needed** when options change
- Preserves per-option calibrated probability distributions
- Option semantics flow through the same backbone → rich representations
- Natural extension of the existing attention-based paradigm

**Cons:**
- More complex architecture (new `query_proj`, `key_proj` layers)
- Option embeddings require backbone calls (can be cached)
- Training needs varied option sets to generalize
- Slightly higher inference latency (attention computation)

**Verdict:** The right long-term solution. Fully dynamic, architecturally clean, and preserves calibration.

---

### 3.3 Option C: Span-Prediction / Token-Level Decoding

**How it works:** Instead of classification heads, use a **generative approach** where the head outputs a token from a constrained vocabulary. The option values define the allowed token set.

```
State ──▶ Qwen ──▶ Hidden ──▶ LM Head ──▶ "positive" (constrained to {"positive","negative","neutral"})
```

This is essentially what the `prompt_lm_zero_shot` agent already does, but with a smaller, faster decoding head instead of full autoregressive generation.

**Pros:**
- Maximum flexibility — any option text, any number
- Leverages backbone's language understanding

**Cons:**
- Requires a language modeling head on top (much larger than current MLP heads: `hidden_dim × vocab_size`)
- Constrained decoding adds complexity
- Harder to get calibrated probabilities
- Loses the efficiency advantage over prompt LM
- The entire point of the typed head is to be *faster and more calibrated* than prompt LM

**Verdict:** Over-engineered. If we want generative flexibility, use the prompt LM agent. The typed head's value is speed + calibration.

---

### 3.4 Option D: Option Embedding Averaging (Simple Baseline)

**How it works:** Keep the current fixed-size head but **pool option embeddings** into a conditioning vector that modulates the head.

```
Options ──▶ Embed ──▶ Mean Pool ──▶ conditioning vector
                                          │
State  ──▶ Qwen ──▶ [conditioned trunk] ──▶ fixed output
```

The output is always e.g. 10-class, but which 10 classes are active is determined by the conditioning vector.

**Pros:**
- Simpler than full attention
- Head size remains fixed

**Cons:**
- Still has a **hard maximum** on option count
- Mean pooling loses option identity → hard to distinguish options
- Conditioning + classification is less principled than direct comparison

**Verdict:** An interesting intermediate step but the hard cap limits its value.

---

## 4. Recommended Approach: Option B (Attention-Based Slot Filling)

### 4.1 Why This Approach

| Criterion | Option A (Token Concat) | Option B (Attention) | Option C (Generative) | Option D (Embed Avg) |
|---|---|---|---|---|
| Unlimited options | ❌ (hard cap) | ✅ | ✅ | ❌ (hard cap) |
| Calibrated probs | ⚠️ (position-bound) | ✅ | ❌ | ⚠️ |
| No retraining per question | ❌ | ✅ | ✅ | ❌ |
| Fast inference | ✅ | ✅ | ❌ | ✅ |
| Clean architecture | ⚠️ | ✅ | ❌ | ⚠️ |
| Reuses backbone | ⚠️ | ✅ | ❌ | ✅ |

Attention-based slot filling is the only approach that is simultaneously:
- Fully dynamic (no hard cap on options)
- Calibrated (proper probability distributions)
- Fast (single forward pass, not autoregressive)
- Clean (extends existing architecture naturally)

### 4.2 Detailed Design

#### 4.2.1 New Model: `DynamicDecisionHead`

```python
class DynamicDecisionHead(nn.Module):
    """
    Decision head where option values are inputs, not architecture constants.
    
    State embedding → query; option embeddings → keys; scores = attention.
    """
    
    def __init__(self, input_dim=1024, hidden_dim=256, d_k=128, dropout=0.1):
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.query_proj = nn.Linear(hidden_dim, d_k)
        self.key_proj = nn.Linear(input_dim, d_k)
        # noul stays as a dedicated binary head (no dynamic values needed)
        self.noul_head = nn.Linear(hidden_dim, 2)
    
    def forward_choice(self, state_embedding, option_embeddings):
        """state: (B, D), options: (B, N_opts, D) → (B, N_opts) logits"""
        h = self.trunk(state_embedding)
        q = self.query_proj(h)              # (B, d_k)
        K = self.key_proj(option_embeddings) # (B, N, d_k)
        scores = torch.bmm(K, q.unsqueeze(-1)).squeeze(-1)
        return scores / (self.query_proj.out_features ** 0.5)
    
    def forward_score(self, state_embedding, level_embeddings):
        """Same mechanism as choice; levels are ordered options."""
        return self.forward_choice(state_embedding, level_embeddings)
    
    def forward_noul(self, state_embedding):
        """Binary classification (unchanged)."""
        h = self.trunk(state_embedding)
        return self.noul_head(h)
```

#### 4.2.2 Option Embedding Strategy

Options are embedded using the **frozen Qwen backbone**, same as states:

```python
def embed_options(options: list[str], server) -> torch.Tensor:
    """Embed option texts → (N_opts, 1024) tensor."""
    embeddings = server.embed(options)
    return torch.tensor(embeddings, dtype=torch.float32)
```

This is efficient because:
- Options are typically short (1-5 words) → fast embedding
- Option embeddings can be **cached per question configuration**
- Same backbone = same semantic space as state embeddings

#### 4.2.3 Training Protocol

**Training with multiple option configurations:**

The key insight: during training, we need the head to see **the same latent state paired with different option sets** to learn that options are variable inputs, not architectural constants.

```
Training example:
  State: "Ticket #12345: degraded performance..."
  
  Example 1 — sentiment question with 3 options:
    options: ["positive", "negative", "neutral"]
    gold: "negative"
  
  Example 2 — sentiment question with 5 options:
    options: ["very negative", "negative", "neutral", "positive", "very positive"]
    gold: "negative"
  
  Example 3 — priority question with 4 options:
    options: ["low", "medium", "high", "critical"]
    gold: "high"
  
  Example 4 — priority question with 3 options:
    options: ["standard", "elevated", "urgent"]
    gold: "urgent"
```

**Training data generation:** Extend `states/generator.py` to produce option-set variants:
1. Generate base latents as today
2. For each base latent, create N variants with different option sets
3. Map gold labels to the correct option in each variant

**Loss function:** Standard cross-entropy, same as today, but computed on the attention scores (which are logits over the dynamic option set).

#### 4.2.4 Backward Compatibility

The `DynamicDecisionHead` can coexist with `TypedDecisionHead`:

```python
# Old API (still works, fixed question bank from config)
head = TypedDecisionHead(question_spec=spec)
outputs = head(state_embedding)

# New API (dynamic options at inference)
head = DynamicDecisionHead()
options_emb = embed_options(["first", "second", "third"], server)
logits = head.forward_choice(state_embedding, options_emb)
```

#### 4.2.5 Web UI Integration

New API endpoints:
- `POST /api/decide` — accepts state + dynamic question definition, returns answer
- `GET /api/question-templates` — predefined question templates (replaces fixed question bank)

Frontend changes:
- Question builder UI: select type → enter options/labels → run
- Template selector: pick from saved question templates
- Results display adapts to dynamic option count

#### 4.2.6 Checkpoint Format

```python
def save_dynamic_head(model, path):
    torch.save({
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "arch": {
            "type": "dynamic",  # discriminator vs "typed"
            "input_dim": 1024,
            "hidden_dim": 256,
            "d_k": 128,
        },
        # No question_spec — options are inputs, not baked in
        # No temperatures — these become per-question-config, stored separately
    }, path)
```

Note: temperatures can no longer be per-question in the checkpoint (since questions are dynamic). Instead, temperature calibration moves to a per-question-config cache or uses a learned temperature predictor.

#### 4.2.7 Calibration Strategy

Since questions are dynamic, per-question temperatures need a new approach:

**Option A (simplest):** Use a single global temperature fitted during training.

**Option B (better):** Learn a small hypernetwork that predicts temperature from option embeddings:
```python
self.temperature_head = nn.Linear(d_k, 1)  # scalar temperature per question config
```

**Option C (most flexible):** Temperature becomes a per-inference parameter that can be tuned with a small calibration set (few-shot calibration).

---

## 5. Implementation Plan

### Phase 1: Core Dynamic Head (Week 1)

**Goal:** `DynamicDecisionHead` model + training that matches `TypedDecisionHead` accuracy on the fixed question bank.

1. **`head/dynamic_model.py`** — New file
   - `DynamicDecisionHead` class with attention-based slot filling
   - `forward_choice`, `forward_score`, `forward_noul` methods
   - `decode_dynamic_answer` — decode attention scores → typed answers
   - `save_dynamic_head` / `load_dynamic_head` — checkpoint I/O

2. **`head/dynamic_train.py`** — New file
   - `train_dynamic_head` — training loop supporting batched (state, options, gold) tuples
   - `DynamicDataset` — PyTorch Dataset that yields (state_emb, option_embs, gold_idx)
   - Integration with existing calibration pipeline

3. **Tests** — `tests/test_dynamic_head.py`
   - Architecture: forward pass shapes, attention score properties
   - Training: convergence on fixed question bank
   - Generalization: accuracy with varied option sets
   - Checkpoint: save/load roundtrip
   - Decoding: correctness across types

### Phase 2: Training Data + Generalization (Week 2)

**Goal:** The head generalizes to unseen option sets.

1. **`states/dynamic_generator.py`** — Extend data generation
   - `generate_dynamic_dataset` — varied option sets per latent
   - Option-set templates (synonyms, granularity changes, label rotations)
   - Curriculum: start with standard bank, gradually introduce variants

2. **Training curriculum**
   - Epochs 1-50: fixed question bank (anchor training)
   - Epochs 51-100: gradually introduce varied option sets
   - Validation: hold out completely novel option sets for eval

3. **Generalization benchmarks**
   - Zero-shot: accuracy on option sets never seen in training
   - Few-shot: accuracy after quick temperature calibration on new sets

### Phase 3: Integration + Web UI (Week 3)

**Goal:** End-to-end dynamic questions in the web UI.

1. **`webapp/app.py`** — New endpoints
   - `POST /api/decide-dynamic` — dynamic question evaluation
   - `GET /api/question-templates` — saved templates
   - Option embedding caching layer

2. **`web/static/app.js`** — UI updates
   - Question builder (type selector + option editor)
   - Template library
   - Dynamic result rendering (adapts to variable option count)

3. **Backward compatibility**
   - `TypedDecisionHead` agent remains available as "head_trained_legacy"
   - `DynamicDecisionHead` agent added as "head_dynamic"
   - Prompt LM agent unchanged

### Phase 4: Polish + Documentation (Week 4)

1. **Documentation**
   - Architecture decision record
   - API documentation
   - Training guide for custom question banks

2. **Performance optimization**
   - Option embedding caching
   - Batch inference for multiple question configs per state
   - Latency benchmarks vs. TypedDecisionHead

3. **Calibration refinement**
   - Temperature prediction from option embeddings
   - Per-question-config calibration cache

---

## 6. Open Questions & Risks

### 6.1 Can attention-based slot filling match fixed-head accuracy?

**Risk:** The attention mechanism might be less discriminative than dedicated per-class linear heads.

**Mitigation:** 
- Train with a curriculum: start anchored to fixed bank, gradually generalize
- The query/key projection is equivalent to a bilinear form: `q^T K = h^T W_q^T W_k O` — with enough capacity this should match or exceed per-class heads
- Benchmark extensively in Phase 1

### 6.2 How to handle option ordering?

**Risk:** Option order might affect attention scores due to positional encoding or sequential processing.

**Mitigation:**
- Attention is permutation-equivariant (no positional encoding on the option side)
- Test with shuffled option orders during training
- The dot-product attention has no inherent ordering bias

### 6.3 What about noul questions?

Noul questions are always binary (true/false). They don't benefit from dynamic values. Keep the dedicated 2-class head for noul, and only use the attention mechanism for choice and score types where option variability matters.

### 6.4 Training data diversity

**Risk:** Without diverse option-set training, the head won't generalize.

**Mitigation:**
- Generate option-set variants programmatically:
  - Synonym replacement (positive → favorable, good)
  - Granularity changes (3-class → 5-class sentiment)
  - Label rotations (low/medium/high → basic/standard/premium)
  - Cross-domain mapping (sentiment labels → priority labels)
- Use the existing template system from `states/generator.py`

### 6.5 Temperature calibration for dynamic questions

**Risk:** Current calibration fits one T per question. With dynamic questions, this doesn't work.

**Mitigation:**
- Phase 1-2: Use a single global temperature
- Phase 3: Implement temperature prediction from option embeddings
- Accept slightly worse calibration as a trade-off for flexibility

---

## 7. Alternative: Minimal Viable Change

If the full `DynamicDecisionHead` is too large a change, a minimal step:

1. **Keep `TypedDecisionHead` as-is** for the fixed question bank
2. **Add option embedding caching** in the web UI so option texts are pre-embedded
3. **Add a new agent** that uses embedding similarity (cosine distance) between state embedding and option embeddings — no training needed

```python
class EmbeddingSimilarityAgent:
    """Zero-shot: pick the option whose embedding is closest to the state."""
    
    def decide(self, state, question_config):
        state_emb = embed(state.text)
        option_embs = embed(question_config["options"])  # cached
        similarities = cosine_similarity(state_emb, option_embs)
        best_idx = argmax(similarities)
        return question_config["options"][best_idx]
```

This requires **zero training** and provides immediate dynamic decision capability, though without calibrated confidence. Could serve as a baseline while the `DynamicDecisionHead` is developed.

---

## 8. Summary

| | Current | Proposed |
|---|---|---|
| Option values | Baked into architecture | Passed as inputs |
| New question | Retrain whole model | Change input text only |
| Max options | `len(options)` at train time | Unlimited |
| Calibration | Per-question temperature | Global or predicted |
| Architecture | `nn.Linear(h, n_opts)` per question | Query/key attention |
| Training | Single option set | Curriculum with varied sets |
| Inference latency | ~0.4 ms | ~0.5 ms (est.) |

The recommended approach (Option B: Attention-Based Slot Filling) provides true dynamic decision capability while preserving the speed and calibration advantages that make the typed head valuable over prompt-based LMs.