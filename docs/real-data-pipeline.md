# Real-traffic data pipeline

Turn real `/api/decide-dynamic` calls into training data, so the dynamic
head sees the input distribution it actually serves (the synthetic bank is
templated and its labels are the generator's latents).

Pipeline: **capture → label in the UI → `build-real` → mix into training → eval**.

The API returns predictions, not labels, so a logged call is an *input* with
no gold. Labels come from either you (in the Label tab) or the OpenRouter
auto-label buttons.

## 0. Privacy

Logged text is real user content (this domain asks `contains_pii`). Two
disclosures, both opt-in:

- **Capture** is off unless `web.log_traffic: true` **and** the request sends
  `X-Decision-Lab-Log: 1`. Rows are written locally to `data/traffic/`.
- **Auto-labeling sends the state text to OpenRouter** (a third party). The
  manual path stays entirely local. `OPENROUTER_API_KEY` is read from the
  environment — never hardcode it.

`data/` is gitignored, so logs and labels never leave the machine via git.

## 1. Capture

```yaml
# configs/default.yaml
web:
  log_traffic: true
```

Then any client opts in per request:

```bash
curl -s http://127.0.0.1:8000/api/decide-dynamic \
  -H 'Content-Type: application/json' \
  -H 'X-Decision-Lab-Log: 1' \
  -d '{"custom_text":"The ball is left of the paddle (gap 124 px)",
       "questions":[{"type":"choice","options":["left","right","stay"],
                     "question":"Which way should the paddle move?"}]}'
```

Rows land in `data/traffic/YYYY-MM-DD.jsonl` (one JSON object per call).

## 2. Label in the UI

Open the web UI and switch to the **Label** tab. It loads every pending
(call, question) pair as a card — state text, question, and the option
buttons with the head's prediction marked `★`.

- **Manual:** click the correct option. Clicking the `★` records `accepted`;
  any other option records `overridden`.
- **Remove (✕):** drop an item from the queue *without* labeling it — for
  calls you can't adjudicate. Recorded in `data/real/discarded.jsonl` and
  excluded from the queue (never used for training).
- **Auto-label 1:** label the first pending item with OpenRouter.
- **Auto-label all:** label every pending item (SSE progress stream).

Labels append to `data/real/labels.jsonl`, deduped by content hash and
idempotent by `item_id`, so repeated traffic is labeled once.

> **Bias note.** The card pre-fills the head's prediction, so `accepted`
> labels are model-informed, not blind gold. Auto labels are a cheaper but
> weaker signal. Set `real.exclude_auto: true` to train on human labels only.

## 3. Build training samples

```bash
no_proxy=localhost,127.0.0.1 uv run python -m decision_lab \
  --config configs/default.yaml build-real
```

Reads `data/real/labels.jsonl`, embeds field sets + options (chunking matches
serving exactly — caller `custom_fields` verbatim, else `states/fields.py`),
and writes:

- `data/real/samples.pt` — `DynamicTrainingSample` rows (`is_variant=True`, stage 2)
- `data/real/test_real.jsonl` — the held-out items (`real.holdout_fraction`)

## 4. Train with the real mix

```yaml
dynamic_head:
  real_data_path: "data/real/samples.pt"
  real_weight: 1        # integer oversample of the real samples
```

Real samples join curriculum **stage 2** (generalization), so the canonical
anchor phase is unchanged. `real_data_path: ""` (default) = synthetic only.

## 5. Evaluate honestly

The benchmark adds a `test_real` column — head accuracy on items held out
from training. That number, not the in-distribution score, is what says
whether real data helped. Keep the `test_real` split **human-labeled** so the
eval is not grading the auto-labeler against itself.

```bash
uv run python -m decision_lab eval      # writes results/, includes test_real
```

If `test_real` does not improve, report that plainly — more real data (or a
better labeler) is needed, not a config tweak.