"""Parse typed answers (`key: value` lines) from LM text output.

Canonical answer values match the gold-label space:
  choice → option key (str); score → level index (int); noul → bool.
"""

import re

# "qid: value" line, case-insensitive key, trailing free text ok
LINE_RE_TEMPLATE = r"^\s*{qid}\s*[:：]\s*(.+?)\s*$"

TRUE_TOKENS = {"true", "yes", "t", "1"}
FALSE_TOKENS = {"false", "no", "f", "0"}


def _parse_bool(token: str) -> bool | None:
    token = token.strip().lower().rstrip(".")
    if token in TRUE_TOKENS:
        return True
    if token in FALSE_TOKENS:
        return False
    return None


def parse_typed_answers(text: str, question_spec: dict) -> dict:
    """Extract every bank question's answer from `qid: value` lines.

    Returns {qid: predicted value} — missing/unparseable questions are absent.
    """
    lines = text.splitlines()
    answers: dict = {}
    for qid, spec in question_spec.items():
        pattern = re.compile(LINE_RE_TEMPLATE.format(qid=re.escape(qid)), re.IGNORECASE | re.MULTILINE)
        match = None
        for line in lines:
            m = pattern.match(line)
            if m is not None:
                match = m
                break
        if match is None:
            continue
        token = match.group(1)

        if spec["type"] == "noul":
            value = _parse_bool(token)
            if value is not None:
                answers[qid] = value
        elif spec["type"] == "choice":
            token_clean = token.strip().lower()
            for opt in spec["options"]:
                if opt.lower() == token_clean:
                    answers[qid] = opt
                    break
        elif spec["type"] == "score":
            digits = re.findall(r"\d+", token)
            if digits:
                level = int(digits[0])
                if 0 <= level < len(spec["levels"]):
                    answers[qid] = level
    return answers


__all__ = ["parse_typed_answers"]