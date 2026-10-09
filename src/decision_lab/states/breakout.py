"""Breakout paddle-control state generator with gold labels from geometry.

States describe ball/paddle positions and velocities; the gold paddle
direction is exact by construction: move toward the ball's horizontal
position ("stay" when the ball is within STAY_THRESHOLD_PX of the paddle).
Ball velocity components are sampled independently of the ball's side, so
surface motion phrases ("moving right", "away from the paddle") carry no
shortcut to the gold label — the head must learn the geometry.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from random import Random

from decision_lab.config import Config
from decision_lab.states.dataset import TextState, save_dataset
from decision_lab.states.fields import _flatten_json, split_sentences

BREAKOUT_STATE_TYPE = "breakout"
PADDLE_QID = "paddle_direction"

# |ball_x - paddle_x| <= this → gold "stay"
STAY_THRESHOLD_PX = 24
# Side states get a gap strictly larger than the stay band
GAP_MAX_PX = 320
GAP_MIN_SKIP = 8  # side gaps start at STAY_THRESHOLD_PX + this

PADDLE_X_RANGE = (360, 480)   # keeps ±GAP_MAX_PX ball positions on a 800px board
BALL_Y_RANGE = (100, 500)
VELOCITIES = (-3, -2, -1, 1, 2, 3)   # nonzero vx/vy magnitudes

BREAKOUT_TEMPLATES: tuple[str, ...] = (
    "prose", "prose_short", "prose_vague", "structured", "log",
)

BREAKOUT_OPTIONS = ("left", "right", "stay")
GOLD_BY_SIDE = {"left": "left", "center": "stay", "right": "right"}


@dataclass
class BreakoutLatents:
    """Latent variables that drive both rendering and the gold label."""

    side: str        # "left" | "center" | "right" — ball side relative to paddle
    gap_px: int      # horizontal |ball_x - paddle_x|
    ball_vx: int     # signed horizontal velocity (decorrelated from side)
    ball_vy: int     # signed vertical velocity (decorrelated from side)
    bricks: int      # distractor: bricks remaining (of 40)
    score: int       # distractor
    lives: int       # distractor
    template: str

    def labels(self) -> dict:
        """Gold label: move toward the ball's horizontal position."""
        return {PADDLE_QID: GOLD_BY_SIDE[self.side]}


def _sample_latents(rng: Random) -> BreakoutLatents:
    """Sample one state's latents; gold label uniform over left/right/stay."""
    gold = rng.choice(BREAKOUT_OPTIONS)
    side = {"left": "left", "right": "right", "stay": "center"}[gold]
    if side == "center":
        gap = rng.randint(0, STAY_THRESHOLD_PX)
    else:
        gap = rng.randint(STAY_THRESHOLD_PX + GAP_MIN_SKIP, GAP_MAX_PX)
    return BreakoutLatents(
        side=side,
        gap_px=gap,
        ball_vx=rng.choice(VELOCITIES),
        ball_vy=rng.choice(VELOCITIES),
        bricks=rng.randint(0, 40),
        score=rng.randint(0, 300),
        lives=rng.randint(1, 3),
        template=rng.choice(BREAKOUT_TEMPLATES),
    )


def make_breakout_state(doc_id: int, rng: Random) -> TextState:
    """Sample latents, render one Breakout state, return TextState with gold."""
    latents = _sample_latents(rng)
    render = _RENDERERS[latents.template]
    text, fields = render(rng, latents)
    return TextState(
        doc_id=doc_id,
        state_type=f"{BREAKOUT_STATE_TYPE}_{latents.template}",
        text=text,
        labels=latents.labels(),
        fields=fields,
    )


def generate_breakout_dataset(cfg: Config, data_dir: Path, rng: Random) -> None:
    """Generate train_breakout / test_breakout JSONL datasets.

    Uses its own Random (caller passes Random(cfg.generator.seed + 1)) so the
    document splits stay byte-identical for the same seed.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    splits = [
        ("train_breakout", cfg.generator.num_breakout_train),
        ("test_breakout", cfg.generator.num_breakout_test),
    ]
    doc_id = 0
    for name, count in splits:
        states = []
        for _ in range(count):
            states.append(make_breakout_state(doc_id, rng))
            doc_id += 1
        path = data_dir / f"{name}.jsonl"
        save_dataset(states, path)
        print(f"  {name}: {len(states)} states → {path}")


def breakout_question_spec() -> dict:
    """Dynamic question spec for paddle-direction choice questions."""
    return {
        PADDLE_QID: {
            "type": "choice",
            "options": list(BREAKOUT_OPTIONS),
            "option_descriptions": {
                "left": "Move the paddle left",
                "right": "Move the paddle right",
                "stay": "Keep the paddle still",
            },
        },
    }


# ---------------------------------------------------------------------------
# Gold re-derivation from rendered text (used by tests and diagnostics)
# ---------------------------------------------------------------------------

def gold_from_text(text: str) -> str:
    """Re-derive the gold paddle direction from rendered text alone."""
    s = text.strip()
    if s.startswith("{"):
        doc = json.loads(s)
        dx = doc["ball"]["x"] - doc["paddle"]["x"]
    elif "GAP=" in s:
        dx = int(re.search(r"GAP=([+-]?\d+)", s).group(1))
    elif "offset" in s:
        dx = 0
    else:
        m = re.search(r"gap (\d+) px", s)
        if m:
            dx = -int(m.group(1)) if "LEFT" in s else int(m.group(1))
        elif "well left of" in s or "left side" in s:
            dx = -(STAY_THRESHOLD_PX + GAP_MIN_SKIP)
        elif "well right of" in s or "right side" in s:
            dx = STAY_THRESHOLD_PX + GAP_MIN_SKIP
        else:
            dx = 0
    return _gold_from_dx(dx)


def _gold_from_dx(dx: int) -> str:
    if abs(dx) <= STAY_THRESHOLD_PX:
        return "stay"
    return "left" if dx < 0 else "right"


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _side_phrase(side: str, gap: int) -> str:
    """Prose clause locating the ball relative to the paddle."""
    if side == "center":
        return f"almost directly above the paddle (offset {gap} px)"
    word = "LEFT" if side == "left" else "RIGHT"
    return f"clearly to the {word} of the paddle (gap {gap} px)"


def _motion_phrase(vx: int, vy: int) -> str:
    """Full motion sentence: horizontal + vertical + toward/away clause.

    The toward/away clause follows the vertical motion (rising reads as
    "away" regardless of side), matching the phrasing real game states use.
    """
    horiz = "moving right" if vx > 0 else "moving left"
    vert = "up" if vy < 0 else "down"
    relation = "away from the paddle" if vy < 0 else "toward the paddle"
    return f"{horiz} and {vert}, {relation}"


def _short_motion(vx: int, vy: int) -> str:
    """Compact motion words for the short/vague templates."""
    parts = []
    if vx > 0:
        parts.append("drifting right")
    elif vx < 0:
        parts.append("drifting left")
    parts.append("rising" if vy < 0 else "falling")
    return ", ".join(parts)


def _distractor_prose(rng: Random, lat: BreakoutLatents) -> str:
    """One of several game-status phrasings (no bearing on gold)."""
    style = rng.randint(0, 2)
    if style == 0:
        return f"{lat.bricks} of 40 remain; score {lat.score}, {lat.lives} lives left."
    if style == 1:
        return f"Bricks remaining: {lat.bricks}. Score: {lat.score}. Lives: {lat.lives}."
    return f"{lat.bricks} bricks still standing, score {lat.score}, {lat.lives} lives."


# --------------------------------------------------------------------------
# Templates (all gold-computable)
# --------------------------------------------------------------------------

def _motion_echo(vy: int, side: str) -> str:
    """Echo sentence real game states append ("Rising away from the paddle.")."""
    if side == "center":
        return "The paddle is under the ball"
    return "Rising away from the paddle" if vy < 0 else "Falling toward the paddle"


def _render_prose(rng: Random, lat: BreakoutLatents) -> tuple[str, list[str]]:
    """User-style prose with randomized sentence composition.

    Covers the shapes real game clients send: geometry first with status
    last (the common layout), status first with geometry last, and
    geometry-only texts with no status sentence at all. Randomization here
    is what keeps the head from keying on sentence position.

    Returns (text, fields) — fields are the rendered sentences in text
    appearance order; text is byte-identical to the pre-field renderer.
    """
    if lat.side == "center":
        geometry = (
            f"The ball is {_side_phrase(lat.side, lat.gap_px)} "
            f"and {_short_motion(lat.ball_vx, lat.ball_vy)}."
        )
    else:
        geometry = (
            f"The ball is {_side_phrase(lat.side, lat.gap_px)} "
            f"and {_motion_phrase(lat.ball_vx, lat.ball_vy)}."
        )
    echo = f" {_motion_echo(lat.ball_vy, lat.side)}." if rng.random() < 0.5 else ""
    status = _distractor_prose(rng, lat)

    geometry_field = geometry
    echo_field = echo.strip() if echo else None
    # Status may contain multiple sentences; emit them separately so the
    # field set matches split_state_fields(text) exactly — training and
    # inference must chunk identically.
    status_fields = split_sentences(status)

    if rng.random() < 0.5:
        # Geometry first; status sometimes omitted entirely
        fields = [geometry_field]
        if echo_field:
            fields.append(echo_field)
        if rng.random() < 0.2:
            return f"{geometry}{echo}", fields
        fields.extend(status_fields)
        return f"{geometry}{echo} {status}", fields
    fields = [*status_fields, geometry_field]
    if echo_field:
        fields.append(echo_field)
    return f"{status} {geometry}{echo}", fields


def _render_prose_short(rng: Random, lat: BreakoutLatents) -> tuple[str, list[str]]:
    """Short prose with side words but no px number."""
    if lat.side == "center":
        geom = f"Ball hovering just above the paddle, {_short_motion(lat.ball_vx, lat.ball_vy)}"
    else:
        word = "left" if lat.side == "left" else "right"
        geom = f"Ball well {word} of the paddle, {_short_motion(lat.ball_vx, lat.ball_vy)}"
    status = f"{lat.bricks} bricks left, score {lat.score}, {lat.lives} lives."
    return f"{geom}. {status}", [f"{geom}.", status]


def _render_prose_vague(rng: Random, lat: BreakoutLatents) -> tuple[str, list[str]]:
    """Vague side words, no measurements at all."""
    if lat.side == "center":
        geom = "The ball hovers just above the paddle."
    else:
        word = "left" if lat.side == "left" else "right"
        geom = f"The ball sits well off to the {word} side of the paddle."
    motion = f"It is {_short_motion(lat.ball_vx, lat.ball_vy)}."
    status = f"Bricks: {lat.bricks}. Score: {lat.score}. Lives: {lat.lives}."
    return (
        f"{geom} {motion} {status}",
        [geom, motion, *split_sentences(status)],
    )


def _render_structured(rng: Random, lat: BreakoutLatents) -> tuple[str, list[str]]:
    """JSON state — gold comes only from x-coordinates, no direction words.

    Fields are per-leaf "path: value" strings (same format as the generic
    JSON splitter in states/fields.py), isolating each coordinate from
    syntax tokens.
    """
    paddle_x = rng.randint(*PADDLE_X_RANGE)
    sign = 1 if lat.side == "right" else -1 if lat.side == "left" else rng.choice((-1, 1))
    ball_x = paddle_x + sign * lat.gap_px
    ball_y = rng.randint(*BALL_Y_RANGE)
    doc = {
        "ball": {"x": ball_x, "y": ball_y, "vx": lat.ball_vx, "vy": lat.ball_vy},
        "paddle": {"x": paddle_x},
        "bricks": lat.bricks,
        "score": lat.score,
        "lives": lat.lives,
    }
    return json.dumps(doc), _flatten_json(doc)


def _render_log(rng: Random, lat: BreakoutLatents) -> tuple[str, list[str]]:
    """Signed-number log line — gold only from the signed gap."""
    sign = 1 if lat.side == "right" else -1 if lat.side == "left" else rng.choice((-1, 1))
    gap_signed = sign * lat.gap_px
    fields = [
        f"GAP={gap_signed:+d}",
        f"VX={lat.ball_vx:+d}",
        f"VY={lat.ball_vy:+d}",
        f"BRICKS={lat.bricks}",
        f"SCORE={lat.score}",
        f"LIVES={lat.lives}",
    ]
    return " ".join(fields), fields


_RENDERERS = {
    "prose": _render_prose,
    "prose_short": _render_prose_short,
    "prose_vague": _render_prose_vague,
    "structured": _render_structured,
    "log": _render_log,
}
