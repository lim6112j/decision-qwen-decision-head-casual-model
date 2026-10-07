"""Prompt-based agent: answers the full typed question bank via chat completion."""

from enum import Enum

from decision_lab.config import Config
from decision_lab.head.model import build_question_spec
from decision_lab.prompt_lm.parser import parse_typed_answers


class PromptMode(Enum):
    ZERO_SHOT = "zero_shot"
    COT = "cot"


class PromptAgent:
    """Answers all bank questions for a text state in one chat completion."""

    def __init__(
        self,
        cfg: Config,
        server,
        mode: PromptMode = PromptMode.ZERO_SHOT,
    ):
        self._cfg = cfg
        self._server = server
        self.mode = mode
        pc = cfg.prompt_lm
        self._system = pc.system_prompt_zero_shot if mode == PromptMode.ZERO_SHOT else pc.system_prompt_cot
        self._temperature = pc.temperature
        self._max_tokens = pc.max_tokens

    @property
    def system_prompt(self) -> str:
        return self._system

    def decide(self, state) -> tuple[dict, str]:
        """Return ({qid: predicted value or None if parse failed}, raw_output)."""
        messages = [
            {"role": "system", "content": self._system},
            {"role": "user", "content": state.render()},
        ]
        raw = self._server.chat(messages, temperature=self._temperature, max_tokens=self._max_tokens)
        answers = parse_typed_answers(raw, build_question_spec(self._cfg.questions))
        return answers, raw