"""Typed decision head: shared trunk + per-question output heads.

Evaluates a fixed question bank (choice / score / noul) over a state
embedding in a single forward pass. Heads emit raw logits; probabilities are
produced via softmax with per-question temperatures fitted post-training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from decision_lab.config import QuestionsConfig

NOUL_FALSE_IDX = 0
NOUL_TRUE_IDX = 1
DEFAULT_TEMPERATURE = 1.0


def build_question_spec(questions: QuestionsConfig) -> dict:
    """Normalize the YAML question bank into a JSON-serializable spec.

    Returns {qid: {"type": "choice", "options": [keys], "option_descriptions": {...}}}
           | {qid: {"type": "score", "levels": [labels]}}
           | {qid: {"type": "noul", "question": text}}
    """
    spec: dict = {}
    for qid, s in questions.choice.items():
        spec[qid] = {
            "type": "choice",
            "options": list(s["options"].keys()),
            "option_descriptions": dict(s["options"]),
        }
    for qid, s in questions.score.items():
        spec[qid] = {"type": "score", "levels": list(s["levels"])}
    for qid, text in questions.noul.items():
        spec[qid] = {"type": "noul", "question": text}
    return spec


class TypedDecisionHead(nn.Module):
    """Multi-head MLP: one shared trunk, one small linear head per question.

    forward(x) → {qid: logits tensor}; all questions answered in one pass.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        question_spec: dict | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not question_spec:
            raise ValueError("TypedDecisionHead requires a non-empty question_spec")
        self.question_spec = dict(question_spec)

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        choice_heads, score_heads, noul_heads = {}, {}, {}
        for qid, s in self.question_spec.items():
            kind = s["type"]
            if kind == "choice":
                choice_heads[qid] = nn.Linear(hidden_dim, len(s["options"]))
            elif kind == "score":
                score_heads[qid] = nn.Linear(hidden_dim, len(s["levels"]))
            elif kind == "noul":
                noul_heads[qid] = nn.Linear(hidden_dim, 2)
            else:
                raise ValueError(f"question '{qid}' has unknown type '{kind}'")

        self.choice_heads = nn.ModuleDict(choice_heads)
        self.score_heads = nn.ModuleDict(score_heads)
        self.noul_heads = nn.ModuleDict(noul_heads)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """x: (batch, input_dim) → {qid: (batch, num_classes) logits}."""
        h = self.trunk(x)
        outputs: dict[str, torch.Tensor] = {}
        for qid, head in self.choice_heads.items():
            outputs[qid] = head(h)
        for qid, head in self.score_heads.items():
            outputs[qid] = head(h)
        for qid, head in self.noul_heads.items():
            outputs[qid] = head(h)
        return outputs

    def num_classes(self, qid: str) -> int:
        s = self.question_spec[qid]
        if s["type"] == "noul":
            return 2
        if s["type"] == "choice":
            return len(s["options"])
        return len(s["levels"])


# ---------------------------------------------------------------------------
# Decoding: logits → probabilities → answers
# ---------------------------------------------------------------------------

def probabilities(
    outputs: dict[str, torch.Tensor],
    temperatures: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Softmax per question, scaled by fitted temperature (T=1 if absent)."""
    temperatures = temperatures or {}
    probs = {}
    for qid, logits in outputs.items():
        t = max(float(temperatures.get(qid, DEFAULT_TEMPERATURE)), 1e-3)
        probs[qid] = F.softmax(logits / t, dim=-1)
    return probs


def decode_answer(
    spec_entry: dict,
    prob_row: torch.Tensor,
) -> dict:
    """Decode one question's probability row into its typed answer.

    Returns {"predicted", "distribution", "confidence"[, "expected"]} where
      choice → predicted option key, distribution {option_key: p}
      score  → predicted level index, distribution {level_label: p}, expected level
      noul   → predicted bool, distribution {"true": P(true), "false": P(false)}
    confidence = max probability over classes.
    """
    kind = spec_entry["type"]
    row = prob_row.detach()
    conf = float(row.max().item())

    if kind == "choice":
        options = spec_entry["options"]
        idx = int(row.argmax().item())
        return {
            "predicted": options[idx],
            "distribution": {opt: float(p) for opt, p in zip(options, row)},
            "confidence": conf,
        }
    if kind == "score":
        levels = spec_entry["levels"]
        idx = int(row.argmax().item())
        indices = torch.arange(len(levels), dtype=row.dtype, device=row.device)
        expected = float((row * indices).sum().item())
        return {
            "predicted": idx,
            "expected": expected,
            "distribution": {lvl: float(p) for lvl, p in zip(levels, row)},
            "confidence": conf,
        }
    if kind == "noul":
        p_false = float(row[NOUL_FALSE_IDX].item())
        p_true = float(row[NOUL_TRUE_IDX].item())
        return {
            "predicted": p_true >= 0.5,
            "distribution": {"true": p_true, "false": p_false},
            "confidence": max(p_true, p_false),
        }
    raise ValueError(f"unknown question type '{kind}'")


def predict_all(
    spec: dict,
    outputs: dict[str, torch.Tensor],
    temperatures: dict[str, float] | None = None,
) -> dict[str, dict]:
    """Full decode: embedding batch → {qid: decoded answer dict}.

    Uses the first batch row if a batch is passed; batch size 1 is the norm
    for single-state inference.
    """
    probs = probabilities(outputs, temperatures)
    return {
        qid: decode_answer(spec[qid], prob_row[0])
        for qid, prob_row in probs.items()
    }


def is_correct(spec_entry: dict, predicted, gold) -> bool:
    """Compare a prediction with gold in the canonical label space.

    predicted=None (parse failure) is always incorrect.
    """
    if predicted is None:
        return False
    kind = spec_entry["type"]
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

def get_device() -> torch.device:
    """Best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_head(model: TypedDecisionHead, path: str, temperatures: dict | None = None) -> None:
    torch.save(
        {
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "question_spec": model.question_spec,
            "temperatures": temperatures or {},
            "arch": {"hidden_dim": model.trunk[0].out_features,
                     "input_dim": model.trunk[0].in_features},
        },
        path,
    )


def load_head(path: str, device: torch.device | None = None) -> TypedDecisionHead:
    """Load a checkpoint saved by save_head (spec + arch are self-describing)."""
    checkpoint = torch.load(path, map_location="cpu")
    arch = checkpoint.get("arch", {})
    model = TypedDecisionHead(
        question_spec=checkpoint["question_spec"],
        hidden_dim=arch.get("hidden_dim", 256),
        input_dim=arch.get("input_dim", 1024),
    )
    model.load_state_dict(checkpoint["model_state"])
    if device is not None:
        model = model.to(device)
    model.temperatures = checkpoint.get("temperatures", {})
    return model


def create_random_head(question_spec: dict, **kwargs) -> TypedDecisionHead:
    """Create a head with random init weights (no training)."""
    return TypedDecisionHead(question_spec=question_spec, **kwargs)