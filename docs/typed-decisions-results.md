# typed-decisions: the real-data pivot (2026-10-13)

Recorded 2026-10-13. Supersedes the synthetic-bank direction. The synthetic
bank is **deprecated as a training source** — it is inflated (see below) and
contributes ~nothing on real data.

## Why we pivoted

The synthetic bank (344k samples from ~10k templated states) turned out to be
mostly redundant: the document latents span only `3 sentiments x 4 urgencies x
4 qualities x 2 actionable x 2 pii = 192` label combinations, so 10k states are
~52 re-renderings of each; ~half the samples were shape-augmentation
duplicates. Shrinking to 40% of the samples (136k) trained **4x faster with no
accuracy loss**, and training instability was large (seed-only held-out spread
**~8pp**: 0.916 vs 0.993 for the same config). See
[toxic-question-regression.md](toxic-question-regression.md) for the full
synthetic-era trail.

## The dataset

[`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
— 1,200 train cases / 6,000 decisions, 400 test cases / 2,000 decisions, four
workflows (agent-trace observability, customer service, invoice processing,
security incidents). Their decisions are the **same typed system** we use:
`choice` (criteria dict), `score` (criteria list), `noul`. This is the data
`convaiinnovations/laya-typed-decisions` fine-tunes on and reports **0.766** on
the test split.

Converted with `scripts/convert_typed_decisions.py` (parquet → our JSONL +
`labels.jsonl`) and evaluated with `scripts/eval_typed_decisions.py`. The case
`state` is a nested JSON object — used verbatim (our field splitter chunks it).

**Gotcha:** their JSON states exceed llama.cpp's default physical batch (512
tokens → HTTP 500 "input is too large to process"). `model.batch_size` was
added to the config and LlamaServer to raise `--batch-size`/`--ubatch-size`.

## Results (our dynamic head, frozen bge-m3, ~820k head)

| setting | typed-decisions test | train time | our bank (held-out) |
|---|---|---|---|
| our head, **no training** on this data | 28.2% (= chance) | – | 0.916 |
| our bank **+ 10%** typed-decisions mixed | **67.2%** | 20.8 min | **0.765** (collapsed) |
| **real-only** (typed-decisions only, bank skipped) | **66.3%** | **52 s** | n/a |
| Laya (fine-tuned ModernBERT), from the model card | **76.6%** | ~5 min | n/a |

### Findings

1. **We can learn this domain.** 28% → 66–67% with a frozen backbone and an
   ~820k head.
2. **The synthetic bank is useless here.** Mixed (67.2%) ≈ real-only (66.3%) —
   90% synthetic data bought ~1pp and cost our bank 0.15.
3. **Multi-domain interference is real and reproducible.** Mixing the new
   domain collapsed our bank 0.916 → 0.765 (every question down). This is the
   `toxic` regression amplified — so that regression was **not** just seed
   noise; the shared frozen head genuinely cannot hold two heterogeneous
   domains.
4. **The remaining gap to Laya is ~10pp** (66.3% vs 76.6%). Candidate causes:
   (a) Laya fine-tunes its backbone end-to-end (we freeze ours) — **now tested:
   worth only +2.9pp, so not the main cause** (see "Backbone adaptation"
   below); (b) Laya encodes state+question in one sequence (we embed them
   separately); (c) head-design differences.
5. **Real-only training is 52 s** (vs 18–70 min on the synthetic bank) — the
   synthetic bank was the iteration bottleneck, not the head.

## Decision

- **Abandon the synthetic bank as a training source.** The default path is
  real-only on the typed-decisions data (`dynamic_head.real_only: true` +
  `real_data_path`). The bank code (`states/generator.py`,
  `head/question_bank.py`) is left in place but is vestigial.
- Keep the synthetic-era checkpoints/experiments for reference only.

## Caveats

- **Not apples-to-apples with Laya**: they trained on this data only and
  fine-tuned the backbone; our 66.3% is a frozen backbone and separate
  embeddings. The ~10pp is a gap to explain, not a like-for-like defeat.
- The 0.766 is the model card's number on the official split, not re-measured
  here.

## Backbone adaptation — LoRA is worth +2.9pp, a third of the gap (2026-10-14)

`scripts/finetune_backbone.py` trains the bge-m3 encoder (LoRA r=16 + the
~820k head, end-to-end) — the Laya-style route the frozen llama-server pipeline
cannot take. First attempted on MPS and parked (2026-10-13): a full backward
through the 568M encoder needed `--freeze-below` + fp16, at 80 min–4.5 h per
epoch. **Now run on one A100-40GB.**

### Headline: 66.3% → 69.2% — real, but only a third of the way to Laya

| setting | typed-decisions test | backbone | train time |
|---|---|---|---|
| frozen bge-m3, real-only (2026-10-13) | 66.3% | frozen | 52 s |
| adapted, **4 epochs** (2026-10-13, fp32) | 59.5% | LoRA | 3.4 h |
| **adapted, 15 epochs** (2026-10-14, best-val @ ep 12) | **69.2%** | LoRA | **2.1 h** |
| Laya (fine-tuned ModernBERT), model card | 76.6% | fine-tuned | ~5 min |

**Adapting the backbone is worth +2.9pp over freezing it** (66.3 → 69.2), not
the ~10pp the gap needed. So freezing the encoder was **not** the main cause of
the gap to Laya; the remaining ~7pp is still open (their ModernBERT, one
sequence per sample, head design, 6 epochs on 6,000 decisions vs our 12).

### Why 4 epochs was worse than doing nothing (59.5%)

The first run's 4 epochs were **compute-starved, not hyperparameter-limited**:
fp32 with **TF32 matmul off** (`torch.backends.cuda.matmul.allow_tf32` defaults
to False; `float32_matmul_precision` was `highest`), so each epoch cost 50 min.
The run looked like “adaptation hurts” when it was just under-trained. Two
fixes: enable TF32 + autocast **bf16** on the encoder (embeddings and the head
stay fp32), and cap eval forwards (`--eval-batch`; 256 samples flatten ~3.8k
texts and OOM). 50 min/epoch → **8.2 min/epoch**, ~6x.

### The 15-epoch curve (per-epoch val, 600 decisions / 120 cases)

Loss falls monotonically, val peaks at **epoch 12** (0.7033) and is flat-to-down
after: ep 9 0.6767, ep 12 **0.7033**, ep 15 0.6900. Best-val selection is worth
0.13pp here — small, but the last epoch is demonstrably not the best one.

### Calibration (temperature, fit on 600 held-out decisions never trained on)

| split | acc | NLL | ECE |
|---|---|---|---|
| calibration, T=1 | – | 0.8564 | – |
| calibration, T=2.193 | – | **0.7104** | – |
| val (600) | 0.7033 | 0.9608 → **0.7518** | 0.1246 → **0.0708** |
| **test (2000)** | **0.6925** | 0.8223 → **0.6905** | 0.1207 → **0.0361** |

Temperature scaling **cuts ECE 3.3x** (0.1207 → 0.0361) and NLL 16%, with
accuracy untouched (softmax rescaling is rank-preserving). The model was
over-confident — it needed T>1. **T is clamped to [0.1, 20]**: an unbounded
LBFGS fit on a small or short-run calibration slice runs away to T in the
thousands and reports meaningless confidence (observed: T=3186 on a 5-row
slice).

### Where it helps — the gain is not uniform

By question type: `choice` **50.7% → 68.0% (+17.3pp)**, `score` +8.5pp,
`noul` +4.0pp. The frozen backbone's weakness was exactly the hardest, most
compositional questions — `category` **+43pp** (51→94%), `churn_risk` +23pp,
`discrepancy_severity` +21pp, `outcome` +19pp. Two questions moved *backwards*
(`risk` −10pp, `matches_order` −1pp); `duplicate` (+0.0pp) was already at 89%
and stayed. By workflow, customer service gained most (+20.2pp), the three
others +5–7pp.

### Reproduce

```bash
# one-off: backbone adaptation on a GPU, ~2 h
CUDA_VISIBLE_DEVICES=1 uv run python scripts/finetune_backbone.py \
  --model BAAI/bge-m3 --train /tmp/td_train.jsonl --test /tmp/td_test.jsonl \
  --epochs 15 --batch 32 --micro-batch 16 --eval-batch 32 --max-len 256 \
  --precision bf16 --grad-checkpoint --seed 42 \
  --out models/head_dynamic_bge_ft15.pt
```

`--grad-checkpoint` is what keeps the 16-sample micro-batch (≈230 texts) at
14.2 GB; without it bf16 alone OOMs at that micro-batch. `CUDA_VISIBLE_DEVICES`
remaps the visible GPU to `cuda:0`, so "GPU 0" in the logs/errors is physical
GPU 1. Metrics land in `models/head_dynamic_bge_ft15.metrics.json`; the
checkpoint carries `head`, `encoder_lora`, `temperature` and `metrics`.

### Caveats

- **The checkpoint's `encoder_lora` needs the PEFT-wrapped bge-m3 to load**, so
  it is **not** drop-in for the llama-server serving path (that path takes a
  GGUF and a head-only checkpoint). Serving the adapted model means either
  merging the LoRA into the encoder or a non-GGUF server.
- Test accuracy is computed here as the argmax over `forward_choice` logits;
  69.2% (this script) vs 69.3% (its own pass) is the sampling difference
  between two eval loops, not a real change.
- Single seed (42) on a single split. The synthetic-era held-out spread was
  ~8pp, so treat ±2pp as noise and anything under ~5pp as unresolved.

## Phase 0 — where the remaining error actually is (2026-10-14)

Before spending more GPU time, three diagnostics. They **changed the plan**:
the ceiling is ~74%, not ~85%, so the fine-tuning gap was mostly miscalibrated
expectations rather than a fixable modeling failure. Scripts:
`results_ft_bge/diag_phase0.py`, `teacher_ceiling.py`, `compare_sample.py`.

### The ceiling: 71.9% (ours) vs 74.2% (teacher) on the same 299 items

A stratified 20-decisions-per-question sample (300 items, 1 call failed → 299)
labeled by `deepseek/deepseek-v4.1-flash` — the teacher already wired into
`real/teacher.py` — then scored head-to-head:

| metric | ours (ft15) | teacher |
|---|---|---|
| accuracy (299 items) | **71.9%** | **74.2%** |
| vs full-test | 69.2% (2000 items) | – |
| prediction agreement | 68% (203/299) | |
| both wrong | 36 items (12%) | |
| on the 89 disagreements | ours right 41 / teacher right 48 | |

SE at n=299 is ~2.6pp, so **−2.3pp is not distinguishable from zero**. An
~820k head on a LoRA-adapted 568M encoder is statistically level with a
frontier LLM here. Only 12% of items are missed by *both*; the two models
disagree more than they jointly fail, which is the signature of ambiguous
labels rather than a systematic weakness either could fix.

**Correction (see Phase 0b below):** this 299-item sample was drawn from the 15
`qid` names, which are not globally unique — it therefore under-sampled the
ambiguous questions. On the corrected 400-item sample (20 decisions per real
question) the identical comparison is **67.0% (ours) vs 69.3% (teacher)**. The
conclusion is unchanged: the two are level.

**Consequence:** the *one-sequence encoding* test — previously the top
recommendation for closing the ~10pp Laya gap — is **dropped**. Two different
architectures converging on the low 70s is not a reason to out-engineer a
ceiling that is this close. Same for swapping the encoder.

### Majority-class collapse is the dominant real failure mode

Whole-test accuracy 69.3% vs a **49.5% majority-class baseline** (+19.7pp of
real signal). But per question the head often just learns the prior — plain
accuracy was inflated ~20pp over the base rate and scored `duplicate` at 0.89
where the model has **zero** skill:

| qid | acc | majority | balanced acc | worst class recall |
|---|---|---|---|---|
| `duplicate` | 0.890 | **0.890** | **0.500** | 11/11 positives missed |
| `discrepancy_severity` | 0.480 | 0.470 | **0.334** | class 3: 0.11, class 1: 0.00 |
| `churn_risk` | 0.680 | 0.470 | **0.483** | class 0: 0.00 (3/3) |
| `disposition` | 0.615 | 0.560 | **0.464** | class 3: 0.00 (13/13) |
| `outcome` | 0.630 | 0.500 | 0.581 | class 0: 0.00 (5/5) |
| `action` | 0.640 | 0.415 | 0.516 | 5 classes, half the cases |
| `severity` | 0.690 | 0.490 | 0.523 | 5 classes |

The pattern is consistent: the model almost never predicts the **rarest, most
severe** class. Loss is unweighted cross-entropy over an imbalanced label
distribution, so "no duplicate" is overwhelmingly rewarded. **Prediction
classes are produced by argmax, which temperature scaling cannot fix** —
calibration corrects NLL/ECE, not the decision rule.

(`action`, `disposition` and `severity` above were also merged across
workflows; see Phase 0b. `urgency`, the worst case, is not even in this table
because its merged bucket hid it.)

### Truncation is not a cause (a hypothesis killed)

| measure | value |
|---|---|
| fields > 256 tokens | **7.01%** (1525 / 21750) |
| field tokens | mean 48, p50 13, p90 145, p99 378, max 603 |
| fields per sample | mean 10.9, max 15 (MAX_FIELDS=16 merges, never drops) |

A 7% clip rate against a measured ~11% cost from truncation cannot explain a
26% error rate. The "separate embeddings can't do cross-field reasoning"
mechanism is partly real but **not** the dominant error source.

### What is worth fixing

The head and the teacher fail on *different* questions, which localizes the
headroom:

| qid | ours | teacher | reading |
|---|---|---|---|
| `discrepancy_severity` | 0.400 | **0.900** | real gap (−50pp) — invoice reconciliation |
| `matches_order` | 0.600 | **1.000** | real gap (−40pp) — invoice reconciliation |
| `needs_human` | 0.550 | 0.700 | ours worse |
| `churn_risk` | 0.650 | 0.750 | ours worse |
| `needs_review` | **0.950** | 0.650 | ours better (+30pp) |
| `disposition` | **0.750** | 0.500 | ours better (+25pp) |
| `credential_compromise` | **0.850** | 0.600 | ours better (+25pp) |

Both regressions are the invoice questions (comparing invoice vs purchase order
vs delivery), where the teacher scores 0.9–1.0 — so those labels **are**
learnable and that is genuine headroom.

**Superseded by Phase 0b below.** These gaps were computed with `action` /
`disposition` / `urgency` merged across workflows and on the unbalanced 299-item
sample; under correct grouping the invoice gaps shrink (`matches_order` −35pp,
`discrepancy_severity` −15pp) and `urgency` turns out to be the head's *best*
relative question. The *direction* holds: the invoice reconciliation questions
are the one place with real, learnable headroom.

### Next (revised)

Superseded by **Phase 0b's Next** below (which adds majority-only accuracy and
deprioritizes the class-weighted loss). Kept for the trail:

1. ~~Report balanced accuracy, not accuracy.~~ Plain accuracy was inflated
   ~20pp by the base rate; use it as the headline from here on.
2. **Class-weighted loss** (+ balanced sampling) to attack the collapse above.
   Targets `discrepancy_severity` / `matches_order` too (documented headroom).
   Cheap, and requires no architecture change.
3. **Ordinal decoding for `score`** (round(expected) vs argmax, tuned on the
   calibration split) — the head already computes `expected`.
4. **Do NOT** build one-sequence encoding or swap the encoder: the ceiling is
   ~74% and we are at ~72% on the sample.
5. Still worth measuring: a second seed (the +2.9pp LoRA gain is inside noise),
   and the full 2,000-decision teacher run (~659k input tokens, under $0.10).

## Phase 0b — the labels are the ceiling, and a `qid` bug hid the break-down (2026-10-14)

The parquet carries the annotators' own statistics, so the ambiguity above does
not have to be inferred from a teacher. Two things came out of auditing them;
the second invalidated the per-question break-downs printed in Phase 0.

### ~40% of the test labels are a MINORITY vote — unreachable by design

`label_agreement[qid]["argmax_agree"]` says whether the stored gold equals the
annotators' majority. It does **not**, for 40.6% of test decisions (812/2000;
train 35.3%). Those golds are a label that lost the vote, so any model answering
their *natural* question is scored wrong:

| gold is the… | n | ours (ft15) | teacher (same sample) |
|---|---|---|---|
| annotators' majority | 1188 | **78.9%** | 81.7% |
| a minority label | **812 (40.6%)** | **55.2%** | 50.3% |

Our head **beats the teacher on the minority labels** (55.2% vs 50.3%). Two
independent models both landing near chance on the same items is the signature
of unreachable labels, not of headroom either could close.

Accuracy tracks the chosen label's vote fraction monotonically — a clean noise
curve, not a capability cliff:

| vote fraction | n | ours | teacher |
|---|---|---|---|
| [0.00, 0.40) | 591 | 57.0% | – |
| [0.40, 0.50) | 345 | 64.1% | – |
| [0.50, 0.60) | 324 | 72.8% | – |
| [0.60, 0.80) | 366 | 75.1% | – |
| [0.80, 0.95) | 255 | 83.5% | – |
| [0.95, 1.01) | 119 | **86.6%** | 98.7% |

Summarised for the teacher sample (n=400): where annotators are split the
teacher scores **51.1%**; where they agree [0.8, 1.0] it scores **98.7%**.
Mean between-annotator total variation is 0.202 (p90 0.444).

**Consequence:** the accuracy a model can reach is bounded by how often the gold
is the majority — **59.4%** if it always matched the annotators' vote. Our 69.2%
is above that only because the head sometimes reproduces the *specific* minority
choice. So the honest headline on this data is majority-only accuracy (77.7%) or
balanced accuracy, and the residual error is at least partly unmeasurable label
noise — which is why the class-weighted loss proposed in Phase 0 is now
**deprioritized**: it cannot recover labels that are minority votes.

### `qid` is only unique within a workflow — 15 names, 20 questions

Three `qid` names appear in more than one workflow and mean different questions:

| qid | workflows | differs how |
|---|---|---|
| `urgency` | all 4 | four different phrasings ("how quickly does this trace need attention" vs "how time-sensitive is this conversation" vs …) |
| `action` | agent_trace_observability (4 options), customer_service (5) | different question + answer space |
| `disposition` | invoice_processing, security_incidents | "this vendor invoice" vs "this security alert" |

`scripts/convert_typed_decisions.py` keyed questions only by
`(type, options, instructions)` and dropped `workflow`, and every eval script
bucketed by bare `qid`. So `urgency` (four questions) was reported as one, and
its accuracy was an artifact of mixing four answer distributions. The converted
JSONL now carries `question_id = "{workflow}:{qid}"`; eval/teacher/diagnostic
scripts key on that. **2000 test decisions: 15 `qid` → 20 `question_id`.**

### Corrected per-question break-down (ours, 100 decisions each)

This is the table Phase 0's "majority-class collapse" section should have shown.
The sharpest failure, `urgency`, was invisible before: its merged bucket averaged
the four workflows into a benign-looking number.

| question_id | acc | majority | balanced acc | worst class recall |
|---|---|---|---|---|
| `invoice_processing:duplicate` | 0.890 | **0.890** | **0.500** | 11/11 positives missed |
| `security_incidents:urgency` | 0.610 | 0.500 | **0.361** | class 0: 0.00, class 1: 0.15 |
| `invoice_processing:disposition` | 0.480 | 0.450 | **0.339** | class 3: 0.00 (5/5) |
| `invoice_processing:discrepancy_severity` | 0.480 | 0.470 | **0.334** | class 1: 0.00 (1/1), class 3: 0.11 |
| `security_incidents:disposition` | 0.750 | **0.740** | **0.373** | class 0: 0.00 (3/3), class 3: 0.00 (8/8) |
| `customer_service:action` | 0.610 | 0.520 | **0.440** | class 3: 0.00 (7/7) |
| `customer_service:churn_risk` | 0.680 | 0.470 | **0.483** | class 0: 0.00 (3/3) |
| `agent_trace_observability:action` | 0.670 | 0.360 | 0.506 | class 3: 0.00 (2/2) |
| `agent_trace_observability:outcome` | 0.630 | 0.500 | 0.581 | class 0: 0.00 (5/5) |
| `security_incidents:severity` | 0.690 | 0.490 | 0.523 | class 0: 0.00 (1/1) |
| `customer_service:needs_human` | 0.700 | 0.670 | 0.630 | class 0: 0.42 |

The **rarest, most severe class is almost never predicted** (0.00 recall on
8–11 items) — the unweighted-CE prior problem is real and now has exact numbers.
But it is entangled with the minority-label rate above: several of those rare
classes are also the ones annotators were split on.

### Corrected head-to-head (balanced 400-item sample: 20 per real question)

Re-running the teacher on 20 decisions per `question_id` — the earlier
stratification gave the four `urgency` questions 6–13 items each — gives ours
**67.0% vs teacher 69.3%** (−2.3pp, SE ~2.3pp → still not distinguishable).

| question_id | ours | teacher | diff |
|---|---|---|---|
| `security_incidents:urgency` | 0.500 | 0.550 | −0.05 |
| `agent_trace_observability:urgency` | **0.600** | 0.200 | **+0.400** |
| `agent_trace_observability:needs_review` | **0.850** | 0.550 | +0.300 |
| `security_incidents:disposition` | **0.800** | 0.750 | +0.05 |
| `invoice_processing:matches_order` | 0.600 | **0.950** | **−0.350** |
| `customer_service:urgency` | 0.450 | 0.650 | −0.200 |
| `customer_service:churn_risk` / `:needs_human` | 0.600 / 0.550 | 0.750 / 0.700 | −0.150 |
| `invoice_processing:discrepancy_severity` | 0.700 | 0.850 | −0.150 |

`matches_order` is the one large, unambiguous, learnable gap (the teacher gets
19/20 and we get 12/20). Prediction agreement is 68%; 18% of items are missed by
both. (The two `urgency` questions at ±8 items looked like the head's best and
worst questions; at 20 items that noise collapses to −5pp / +40pp.)

### Next (Phase 0b revision)

1. **Report majority-only and balanced accuracy alongside plain accuracy.**
   Plain accuracy is bounded by the 59.4% majority rate and inflated by it.
2. **Deprioritize class-weighted loss** — the collapse is partly label noise it
   cannot fix. If tried, evaluate on balanced accuracy and per-class recall, not
   accuracy.
3. **`matches_order` / invoice reconciliation is the real target** — the
   documented, learnable gap.
4. ~~One-sequence encoding / encoder swap~~ — still dropped (Phase 0).
5. **Consider a majority-vote re-labeling pass** before spending more GPU: if
   the benchmark's own gold is a minority vote 40% of the time, no model can
   chase it, and training on it teaches the head to guess.

### Reproduce

```bash
# per-question ambiguity + accuracy-vs-vote-fraction + majority/minority split
HF_HOME=models/.cache/huggingface CUDA_VISIBLE_DEVICES=1 \
  uv run python results_ft_bge/label_audit.py

# balanced teacher sample (20 per question_id) + head-to-head on the same items
uv run python results_ft_bge/teacher_ceiling.py --per-qid 20
HF_HOME=models/.cache/huggingface CUDA_VISIBLE_DEVICES=1 \
  uv run python results_ft_bge/compare_sample.py
```

## Running on a GPU server (fresh clone)

The backbone-adaptation experiment is self-contained (only HF downloads needed):

```bash
pip install -e ".[dev]" && pip install transformers peft pyarrow

# 1. data: parquet → our JSONL (+ labels.jsonl)
mkdir -p /tmp/typed-decisions
cd /tmp/typed-decisions
for s in train test; do
  curl -sL "https://huggingface.co/datasets/LocalLLaMA/typed-decisions/resolve/refs%2Fconvert%2Fparquet/all/$s/0000.parquet" -o "all_$s.parquet"
done
cd -        # back to the repo
python scripts/convert_typed_decisions.py /tmp/typed-decisions/all_train.parquet \
    --labels-out data_bge/real/labels.jsonl -o /tmp/td_train.jsonl
python scripts/convert_typed_decisions.py /tmp/typed-decisions/all_test.parquet \
    -o /tmp/td_test.jsonl

# 2. backbone adaptation (GPU): LoRA bge-m3 encoder + head, end-to-end
python scripts/finetune_backbone.py \
    --model BAAI/bge-m3 --train /tmp/td_train.jsonl --test /tmp/td_test.jsonl \
    --epochs 4 --batch 32 --max-len 256      # drop --freeze-below/--amp on a big GPU
```

Notes:
- The command above is the **MPS-era recipe** (`--freeze-below`/`--amp` are
  MPS workarounds). On a GPU use the bf16 recipe in the section above.
- `--frozen-encoder` runs the **frozen baseline for the same encoder**
  (head-only), so you can A/B adapt-vs-frozen on bge-m3 itself.
- The `66.3%` frozen baseline in the table above was produced by the **local
  llama-server pipeline** (`build-real` + `train-dynamic` real_only), which
  needs `models/bge-m3-Q8_0.gguf` + llama.cpp — both gitignored, so reproduce
  those numbers only if you also fetch the GGUF. The `finetune_backbone.py`
  route needs neither.

## Next

Superseded by **Phase 0 / Phase 0b** above — the "~7pp to Laya" framing below
was miscalibrated (the measured ceiling is ~74%, we are level with the teacher
on the balanced sample, and ~40% of the gold labels are minority votes). Kept
for the trail:

1. ~~Close the remaining ~7pp to Laya (69.2% vs 76.6%)~~ — **dropped**: 76.6%
   is Laya's number on the *official* split and is not comparable to ours.
2. **Merge the LoRA into the encoder** so the adapted model can actually be
   served — today the checkpoint only loads through PEFT (see caveats).
3. ~~Test joint state+question encoding (one sequence)~~ — **dropped**: two
   different architectures both land in the low 70s.
4. Adopt this real-only + typed-decisions setup as the standard experiment
   harness (52 s per run frozen; 2.1 h adapted) — **with balanced accuracy**
   report as the headline metric.
