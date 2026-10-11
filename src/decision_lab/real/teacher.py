"""OpenRouter-backed auto-labeler for real-traffic items.

Sends the state text + question to an OpenRouter chat model and asks it to
pick exactly one of the item's allowed labels, constrained by a strict
JSON-schema ``enum`` (see the OpenRouter structured-outputs guide). The
label is mapped back to an index into ``item.options``.

The model is a *teacher*, not ground truth — its labels inherit its errors.
The held-out ``test_real`` split should stay human-labeled so the eval is
not grading the teacher against itself.
"""

from __future__ import annotations

import json
import os

import requests

from decision_lab.real.labels import LabelItem

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"
DEFAULT_TIMEOUT = 60

SYSTEM_PROMPT = (
    "You label text states for a decision dataset. You are given a state "
    "text and a question about it. Answer using ONLY information in the "
    "state text. Choose exactly one of the allowed labels — never invent a "
    "label, never hedge, never explain. Reply with JSON only."
)


class LabelingError(RuntimeError):
    """A label could not be produced (missing key, bad response, bad label)."""


def allowed_labels(item: LabelItem) -> list[str]:
    """The model's answer space — the item's option texts."""
    return list(item.options)


def build_user_prompt(item: LabelItem) -> str:
    labels = allowed_labels(item)
    question = item.question or "(no question text — infer the intent from the options)"
    return (
        f"STATE TEXT:\n{item.text}\n\n"
        f"QUESTION: {question}\n\n"
        f"ALLOWED LABELS: {json.dumps(labels, ensure_ascii=False)}\n\n"
        f"Pick exactly one allowed label for the state."
    )


def build_response_format(labels: list[str]) -> dict:
    """Strict JSON schema whose ``answer`` is constrained to ``labels``."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "decision_label",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string", "enum": labels}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    }


class OpenRouterLabeler:
    """Label items with an OpenRouter model. Raises early if the key is absent."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise LabelingError(
                "OPENROUTER_API_KEY is not set — auto-labeling needs an "
                "OpenRouter API key in the environment."
            )
        self.api_key = key
        self.model = model
        self.timeout = timeout

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def label(self, item: LabelItem) -> int:
        """Return the gold index for ``item`` (index into ``item.options``)."""
        labels = allowed_labels(item)
        if len(labels) < 2:
            raise LabelingError(f"item {item.item_id} has < 2 options; nothing to label")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(item)},
            ],
            "response_format": build_response_format(labels),
            "temperature": 0,
        }

        try:
            resp = requests.post(
                OPENROUTER_URL, headers=self._headers(), json=payload, timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise LabelingError(f"OpenRouter request failed: {exc}") from exc

        try:
            content = resp.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise LabelingError(f"unexpected OpenRouter response shape: {exc}") from exc

        try:
            answer = json.loads(content)["answer"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise LabelingError(f"model did not return a usable label: {content!r}") from exc

        if answer not in labels:
            raise LabelingError(f"model returned {answer!r}, not in allowed labels {labels}")
        return labels.index(answer)