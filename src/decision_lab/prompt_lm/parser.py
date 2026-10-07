"""Parse action from LM text output."""

import re

from decision_lab.env.gridworld import ACTION_NAMES

# Matches "Answer: <word>", case-insensitive, trailing free text ok
ANSWER_RE = re.compile(r"Answer:\s*(\w+)", re.IGNORECASE)

ACTION_MAP = {name: i for i, name in enumerate(ACTION_NAMES)}
ACTION_MAP.update({name[0]: i for i, name in enumerate(ACTION_NAMES)})  # u/d/l/r/w


def parse_action(text: str) -> int | None:
    """Extract action index (0-4) from LM output. None if unparseable."""
    m = ANSWER_RE.search(text)
    if m is None:
        return None
    word = m.group(1).strip().lower()
    return ACTION_MAP.get(word)