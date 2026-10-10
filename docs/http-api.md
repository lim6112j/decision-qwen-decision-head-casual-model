# HTTP API — Using the Dynamic Head from Another Project

The trained dynamic decision head (`models/head_dynamic.pt`) can be consumed
over HTTP from other codebases (e.g. a project that already runs its own
decision models). No code import is needed — this repo runs as a small
inference service.

## Starting the service

```bash
uv run python -m decision_lab ui
```

This starts two processes:

- `llama-server` (llama.cpp) on port **8080**, serving the frozen Qwen3.5-0.8B
  backbone used for embedding extraction (GGUF: `models/Qwen3.5-0.8B-UD-Q4_K_XL.gguf`,
  last-token pooling, embedding dim **1024**)
- the FastAPI app on `http://127.0.0.1:8000` (host/port configurable via
  `web.host` / `web.port` in `configs/default.yaml`)

Health check — wait for `ready: true` before sending requests:

```bash
curl -s http://127.0.0.1:8000/api/status   # {"ready":true}
```

## Endpoint: `POST /api/decide-dynamic`

Answers fully caller-defined questions about a text state in a single forward
pass. The output space is determined by the option/level strings in the
request — no retraining or server restart for new question types.

### Request

```json
{
  "custom_text": "URGENT: prod server is down, need someone now",
  "questions": [
    {"type": "noul",   "question": "Is this actionable?"},
    {"type": "choice", "options": ["low", "medium", "high"], "question": "How urgent is this item?"},
    {"type": "score",  "levels": ["Poor", "Fair", "Good", "Excellent"], "question": "What is the quality of this text?"}
  ]
}
```

- `custom_text` (string) — raw text to evaluate. Wins over `doc_id` when both
  are given.
- `custom_fields` (list of strings, optional) — caller-controlled field
  chunking of `custom_text` for the v2 field-set head. **Chunking must match
  training**: omit it and the server applies the same heuristic splitter
  (`src/decision_lab/states/fields.py:split_state_fields`) used at feature
  extraction — JSON objects split per leaf (`"ball.x: 369"`), newline
  blocks per line, log lines per `KEY=VALUE` token, prose per sentence
  (capped at 16 fields, merged when exceeded). Pass it only if you chunk
  yourself and can keep that chunking stable. Requires the v2 checkpoint;
  a legacy (v1, pooled-state) checkpoint rejects it with a 400.
- `doc_id` (int) — alternatively pick one of the pre-generated states served by
  `GET /api/states`.
- `questions` (list, ≥1) — one of:
  - `{"type": "noul", "question": "..."}` — boolean; output space is always `true`/`false`
  - `{"type": "choice", "options": [...], "question": "..."}` — ≥2 option strings
  - `{"type": "score", "levels": [...], "question": "..."}` — ≥2 ordered rubric levels

Note: the `question` string conditions the head (v3 question-fused queries)
— the same state text with different question strings can produce different
answers. It may be omitted (learned null-question behavior). Legacy
pre-v3 checkpoints ignore it. Option/level label strings are embedded via
the backbone, cached server-side per label — sending identical option
strings across calls is cheap.

**Use natural-language phrasings** ("How urgent is this item?") — the head
was trained on natural questions, not bare identifiers. A raw qid like
`"urgency"` as the question is off-distribution and silently degrades
predictions (no error is raised; verified 2026-10-10: breakout
`ball_motion` flipped from 8/8 correct to 3/8 with `"ball_motion"` as the
question string).

### Supported question domains

The served head is trained on two state domains; option sets from other
domains are not learned and can fail unpredictably (see the novel-domain
generalization numbers in README.md):

1. **Office documents** — sentiment / urgency / quality / actionable / PII
   questions over synthetic document states (the original benchmark).
2. **Breakout paddle control** — choice questions with options
   `left` / `right` / `stay` over game-state text describing where the ball
   is relative to the paddle (gap in px, motion direction, bricks/score/lives).
   Gold convention: move **toward the ball's horizontal position**; `stay`
   when the ball is within ~24 px of the paddle.

Breakout example:

```json
{
  "custom_text": "The ball is clearly to the LEFT of the paddle (gap 124 px) and moving right and up, away from the paddle.",
  "questions": [
    {"type": "choice", "options": ["left", "right", "stay"],
     "question": "which direction the paddle move?"}
  ]
}
```

→ `{"predicted": "left", ...}` — the ball's side decides the direction even
when its motion points the other way ("moving right … away from the paddle").

Since v3, the `question` string is functional: it is embedded with the same
backbone and FiLM-modulates each option's query before it reads the state
field set. One head answers different questions over the same `custom_text`
(e.g. Breakout `paddle_direction` vs `ball_motion` share the options
`left/right/stay` but have independent golds). Pre-v3 checkpoints ignore
the question string — retrain and restart the server to enable it.

### Response

```json
{
  "state": {"doc_id": -1, "state_type": "custom", "text": "...", "has_gold": false},
  "answers": [
    {
      "predicted": true,
      "distribution": {"true": 0.91, "false": 0.09},
      "confidence": 0.91
    },
    {
      "predicted": "high",
      "distribution": {"low": 0.05, "medium": 0.27, "high": 0.68},
      "confidence": 0.68
    },
    {
      "predicted": 2,
      "distribution": {"Poor": 0.08, "Fair": 0.24, "Good": 0.55, "Excellent": 0.13},
      "confidence": 0.55,
      "expected": 1.73
    }
  ],
  "latency_ms": 14.2
}
```

- `predicted` — argmax label (boolean for `noul`, one of `options` for
  `choice`, **0-based index into `levels`** for `score`)
- `distribution` — softmax probabilities (temperature-calibrated, T is stored
  in the checkpoint)
- `confidence` — probability of the predicted label
- `expected` — score questions only: expected value over level indices

`answers` is parallel to `questions` in the request (same order, same length).

### Errors

| Status | Meaning |
|---|---|
| 503 | server still loading (check `/api/status`) |
| 400 | dynamic head not trained (`models/head_dynamic.pt` missing); malformed request — missing `custom_text`/`doc_id`, or a question entry with unknown `type` or a missing/empty `options`/`levels` list |
| 404 | unknown `doc_id` |
| 422 | body does not match the request schema (e.g. no `questions` list) — the response `detail` names the offending field |
| 500 | decision failed (see `detail`) |

## Minimal client (Python)

```python
import requests

DECISION_LAB = "http://127.0.0.1:8000"

def decide(state_text: str, questions: list[dict]) -> list[dict]:
    r = requests.post(
        f"{DECISION_LAB}/api/decide-dynamic",
        json={"custom_text": state_text, "questions": questions},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["answers"]
```

## Operational notes

- The endpoint is **synchronous**; concurrent requests are safe but share one
  llama-server instance, so throughput is bounded by backbone embedding speed.
  (`/api/run` — the SSE evaluation stream — is the only endpoint serialized
  by a lock.)
- Embeddings require the **same backbone** the head was trained on: the
  Qwen3.5-0.8B Q4_K_XL GGUF with last-token pooling (dim 1024). Swapping the
  GGUF or pooling mode degrades accuracy silently.
- v2 field-set head: the state is a **set of field embeddings** (sentences /
  key:value leaves), and each option cross-attends over that set. One
  out-of-distribution token contaminates only the field it appears in, not
  the whole state. The full-text embedding is always prepended as field 0
  (summary field) to keep global context.
- Boolean (`noul`) questions are scored via attention over the embeddings of
  the strings `"false"`/`"true"` — the same path they are trained on
  (`forward_choice`), so training and serving agree by construction.
