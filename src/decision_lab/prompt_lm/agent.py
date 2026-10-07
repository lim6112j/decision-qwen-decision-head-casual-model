"""Prompt-based agent: zero-shot and CoT variants for gridworld decisions."""

from enum import Enum
from typing import Optional

from decision_lab.config import Config
from decision_lab.env.gridworld import GridState
from decision_lab.prompt_lm.parser import parse_action


class PromptMode(Enum):
    ZERO_SHOT = "zero_shot"
    COT = "cot"


class PromptAgent:
    """Decides gridworld actions via chat completion (prompt-based)."""

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

    def decide(self, state: GridState) -> tuple[Optional[int], str]:
        """Return (action_index_or_None_if_parse_failed, raw_output)."""
        messages = [
            {"role": "system", "content": self._system},
            {"role": "user", "content": state.render()},
        ]
        raw = self._server.chat(messages, temperature=self._temperature, max_tokens=self._max_tokens)
        action = parse_action(raw)
        return action, raw