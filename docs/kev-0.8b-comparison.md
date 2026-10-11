# Kev-0.8B vs the frozen-head design (and when backbone LoRA would help)

Recorded 2026-10-11. Compares an external solution — **Kev-0.8B** ([jaredpalmer/kev](https://huggingface.co/jaredpalmer/kev-0.8b),
model card [here](https://raw.githubusercontent.com/jaredpalmer/kev/master/docs/model-cards/kev-0.8b.md)) —
with this repo's v4 head and bge-m3 backbone, and works out when fine-tuning the
backbone (LoRA) would actually help here. It is an analysis note, not a plan.

## What Kev-0.8B is

- **LoRA adapter** (rank 16, α 32, **11.3M trainable params**) on a **frozen
  `Qwen/Qwen3.5-0.8B-Base`** (revision `dc7cdfe2`), plus a tiny **pointer head**
  (two projections score each option's closing token against the question's
  final token; softmax → probabilities; one temperature T = 2.35).
- **Prefill-only** — one forward pass, no text generation.
- Input: one **state** (text, or JSON rendered as labelled text) + any number of
  named questions (`choice` / `score` / `noul`). Each question is answered as
  **its own row that continues from the shared state**, so questions are isolated
  and the state is computed once and cached. Serves the System One contract
  `POST /v1/systemone`.
- Trained on ~30k records in three stages (base recipe → dates/missing-evidence →
  documents/skills); the largest stage was 2,818 optimizer steps, 52 min on one
  H200. Startup `uv run --extra serve python -m kev.serve ...`.
- Reported: strong **in-distribution** (`documents-v1` test 0.851), markedly
  weaker **out of distribution** (`transfer-v4` 0.697, `breadth-v1` 0.586,
  MMLU-Pro 0.230; date arithmetic 0.35). The card states the model trails its
  larger sibling by ~21–31 points OOD and warns it is sub-1B and weak off-domain.

## The two projects share a backbone

Kev's base, `Qwen/Qwen3.5-0.8B-Base`, is the **same model this repo freezes**
(`models/Qwen3.5-0.8B-UD-Q4_K_XL.gguf`, `configs/default.yaml:38`). The comparison
is therefore a clean head-to-head on one axis: **freeze the backbone and build a
smart head** (this repo) **vs. fine-tune the backbone and keep the head trivial**
(Kev).

| Axis | Kev-0.8B | v4 → bge (this repo) |
|---|---|---|
| Backbone | **LoRA fine-tuned** (11.3M params) | **frozen**, embeddings only (`README.md:160`) |
| Conditioning happens | in-context cross-attention inside the backbone (state + question in one sequence) | in-head attention over frozen field embeddings |
| Head | pointer head (~tiny) | attention head, **≈ 820,867 params** (v4) |
| State input | raw text, tokenized, up to 65k tokens | **field set**: ≤ 16 sentence/leaf fields + summary field (`states/fields.py:24,154-174`) |
| Question handling | separate row per question, shared cached state | question emb → cosine query over fields → gates + shifts option queries (`dynamic_model.py`, v4 `z_q`) |
| Generation | none (prefill) | none |
| Generalization mechanism | LoRA reshapes the representation | **cosine nearest-neighbour** over frozen embeddings — an unseen question lands near a trained one |
| Latency / hardware | 16–200 ms, GPU 3.8 GB | **~11–12 ms**, on-device (MPS) |

**Numbers are not comparable across the two rows.** This repo's held-out is
unseen *templates of its own synthetic generator* (an easier OOD); Kev's held-out
is genuinely different domains. Neither "beats" the other on the other's suite.

## Bad embedding geometry has two remedies — and bge already took the cheap one

The bge experiment (`README.md` "Non-autoregressive backbone") diagnosed the
bottleneck as **frozen-backbone embedding geometry**: last-token pooling on a
causal LM gives weak whole-document vectors, so a 4-level rubric (`quality`) sat
at 56%. There are two ways to fix bad geometry:

1. **Swap the backbone** for one whose embeddings already have good geometry.
2. **Train the backbone** (Kev-style LoRA) so its geometry becomes task-shaped.

This repo took (1): `bge-m3` is a contrastively-trained encoder, so good
semantic geometry is its native property — the swap lifted held-out 75.7% →
91.6% with **no gradients at all**. Kev-style LoRA is remedy (2), which is the
fix for a backbone (a decoder-only causal LM) whose geometry is wrong *to begin
with*. **On bge the LoRA lever is largely redundant** — it would be fixing a
problem bge already solved, and more cheaply.

It is also **impractical on the current seam**: `llama-server` is forward-only
(`/v1/embeddings`, `backbone/llama_server.py:203-211`), with no gradient path; a
bge LoRA would force training off the frozen-GGUF architecture into PyTorch,
which is the opposite of the bge route's "5-minute backbone swap".

## When LoRA on the Qwen backbone *would* help

These are the only cases where fine-tuning the backbone adds a capability the
frozen pipeline structurally lacks. All three should be falsified with frozen
alternatives first (see below).

1. **Question-conditioned document representations.** Today the field
   embeddings are computed **question-agnostically**; the question is reconciled
   afterwards, in the head, by cosine attention. LoRA lets the backbone see
   state + question **in one sequence** and cross-attend them, so the *same
   document is encoded differently depending on the question* — Kev's row design
   exists for exactly this. The breakout failure in `docs/v4-open-issues.md` §1
   ("the same option text `left` must match the ball's *position* field under
   `paddle_direction` but the *motion* field under `ball_motion`") is a
   conditional-matching problem; the head approximates it with `gate`/`shift`,
   but a backbone that conditions directly is stronger.
2. **Task-specific contrasts.** bge-m3 gives *generic* semantic similarity; this
   task's discriminating features are *specific* — a numeric threshold
   (`stay = |gap| ≤ 24 px`), polarity inversions, PII presence. LoRA with
   minimal-pair / distractor training (Kev attaches a minimal pair to a quarter
   of choice records) sharpens exactly these contrasts.
3. **Long / unstructured documents.** Fields are hard-capped at **16**
   (`states/fields.py:24`, merged to fit). A 10k-token document cannot be
   represented as 16 fields, and field splitting itself degrades. A fine-tuned
   backbone attending over the whole document (Kev serves 8k–65k tokens) wins
   this regime; the frozen field-set head structurally cannot follow. Short
   structured states → frozen head wins on cost; long unstructured documents →
   fine-tuned backbone wins on capability.

## Costs and risks

- **Numeric reasoning is not fixed by LoRA.** Held-out `contains_pii` (79.0%) and
  `is_urgent` (79.5%) did not improve under bge, and the repo diagnoses them as
  label ambiguity / numeric reasoning (`README.md:513-516`) — not geometry. Kev's
  own strongest negative result is exactly this class: date arithmetic 0.35 vs
  its sibling's 0.95. A sub-1B LoRA does not buy arithmetic.
- **It risks eroding v4's headline capability.** v4's reason to exist is
  answering *unseen question targets* (`not_urgent` 3.0% → 97.4%), and that rides
  on the frozen backbone's *general* semantic geometry. Fine-tuning the backbone
  narrows that geometry (catastrophic-forgetting of general similarity), which
  can regress compositional generalization — the opposite direction from bge,
  which *preserved* the geometry.
- **Serving seam inversion.** llama-server exposes forward passes only. A
  backbone LoRA means training in PyTorch/peft and merging the adapter back for
  serving, which invalidates the frozen-GGUF design (feature-cache fingerprint,
  `arch.backbone_id` guard, the text cache).
- **Cost.** ~0.82M → 11M+ trainable params (~14×), plus per-domain retraining.

## Correction to two pieces of prior framing

1. **`README.md:473` frame.** The bge section calls Qwen3.5 a "decoder-only"
   model where "every hidden state is causally masked." Qwen3.5-0.8B is
   **hybrid: 24 layers = 18 Gated DeltaNet (linear attention) + 6 full
   attention**. The DeltaNet layers carry a rolling recurrent state and are
   *not* block-causally masked. The *conclusion* (last-token pooling on a
   next-token model gives weak cosine geometry) still holds; only the stated
   mechanism is off.
2. **`README.md:161` "LoRA does not help."** That was a **post-hoc low-rank
   transform trained on already-frozen embeddings** (v1). Kev puts LoRA *inside*
   the backbone, with gradients flowing through all 24 layers. Re-weighting
   frozen vectors and adapting the representation are different experiments; the
   v1 negative result does not refute the fine-tune route. (It also does not
   argue *for* it here, since bge already won the geometry problem.)

## Recommended order

Test the frozen alternatives before reaching for LoRA:

- (1) question conditioning → widen the head's `gate`/`shift` to a 2-layer MLP
  (`docs/v4-open-issues.md` §1);
- (2) task-specific matching → raise `breakout_weight`;
- (3) long documents → not reachable with field splitting at all, so this is the
  one case where the backbone-fine-tuning conversation is genuinely open.

LoRA is the card to play when a **conditional / matching failure survives bge**
— not a next step on top of it.

## Sources

- [Hugging Face — jaredpalmer/kev-0.8b](https://huggingface.co/jaredpalmer/kev-0.8b)
- [Kev-0.8B model card (raw)](https://raw.githubusercontent.com/jaredpalmer/kev/master/docs/model-cards/kev-0.8b.md)
- [AgentLensHQ — "Kev: Open-Source Decision Models built on Qwen3.5"](https://agentlenshq.com/blogs/002241e1-d1fd-4374-91ea-61edfeedfdd2)
