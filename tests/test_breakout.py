"""Tests for the Breakout paddle-control state generator."""

import tempfile
from dataclasses import replace
from pathlib import Path
from random import Random

from decision_lab.config import load_config
from decision_lab.states.breakout import (
    BREAKOUT_TEMPLATES,
    MOTION_QID,
    PADDLE_QID,
    STAY_THRESHOLD_PX,
    breakout_question_spec,
    generate_breakout_dataset,
    gold_from_text,
    make_breakout_state,
)
from decision_lab.states.dataset import load_dataset, save_dataset


def _states(n: int, seed: int = 7) -> list:
    rng = Random(seed)
    return [make_breakout_state(i, rng) for i in range(n)]


class TestGoldLabels:
    def test_labels_valid_and_balanced(self):
        states = _states(300)
        labels = [s.labels[PADDLE_QID] for s in states]
        assert set(labels) == {"left", "right", "stay"}
        for gold in ("left", "right", "stay"):
            share = labels.count(gold) / len(labels)
            assert share >= 0.25, f"{gold} share {share:.2f} below 0.25"

    def test_gold_matches_stay_threshold(self):
        states = _states(300)
        for s in states:
            gold = s.labels[PADDLE_QID]
            if gold == "stay":
                # stay states never use prose side phrasing with a large gap
                assert "clearly to the" not in s.text
            else:
                assert gold in ("left", "right")

    def test_ball_motion_gold_from_vx(self):
        """ball_motion gold = sign of ball_vx; paddle gold is independent."""
        states = _states(600)
        motion = [s.labels[MOTION_QID] for s in states]
        assert set(motion) == {"left", "right", "stay"}
        # vx uniform over 7 values → stay share ≈ 1/7, left/right ≈ 3/7
        assert motion.count("stay") / len(motion) >= 0.05
        assert motion.count("left") / len(motion) >= 0.25
        assert motion.count("right") / len(motion) >= 0.25

    def test_paddle_and_motion_golds_decorrelated(self):
        """Same state must answer differently under the two questions often
        enough that the head cannot fit both without reading the question."""
        states = _states(600)
        differ = sum(
            1 for s in states
            if s.labels[PADDLE_QID] != s.labels[MOTION_QID]
        )
        assert differ / len(states) >= 0.5

    def test_gold_rederived_from_text_across_templates(self):
        """gold_from_text(text) must equal the generated gold for every template."""
        states = _states(300)
        seen_templates = set()
        for s in states:
            template = s.state_type.removeprefix("breakout_")
            seen_templates.add(template)
            assert gold_from_text(s.text) == s.labels[PADDLE_QID], (
                f"template={s.state_type} text={s.text!r} gold={s.labels[PADDLE_QID]}"
            )
        assert seen_templates == set(BREAKOUT_TEMPLATES)

    def test_velocity_words_decorrelated_from_gold(self):
        """Within each gold label, both 'away' and 'toward' motion phrasings occur."""
        states = [s for s in _states(600) if s.state_type == "breakout_prose"]
        for gold in ("left", "right"):
            texts = [s.text for s in states if s.labels[PADDLE_QID] == gold]
            assert any("away from the paddle" in t for t in texts), f"no 'away' with gold={gold}"
            assert any("toward the paddle" in t for t in texts), f"no 'toward' with gold={gold}"

    def test_coordinate_templates_have_no_direction_words(self):
        """structured/log templates carry gold only via numbers/coordinates."""
        states = _states(400)
        for s in states:
            if s.state_type in ("structured", "log"):
                assert "left" not in s.text.lower()
                assert "right" not in s.text.lower()


    def test_prose_covers_realistic_sentence_orders(self):
        """The prose template must cover the shapes real game clients send:
        geometry-first with status last, status-first, and geometry-only."""
        states = [s for s in _states(800) if s.state_type == "breakout_prose"]
        texts = [s.text for s in states]
        assert any(t.startswith("The ball is") and "score" in t for t in texts), \
            "no geometry-first-with-status prose"
        assert any(
            t.split(".")[0].startswith(("Bricks", "bricks", "B")) and "The ball is" in t
            for t in texts
        ), "no status-first prose"
        assert any("score" not in t.lower() for t in texts), "no geometry-only prose"


class TestSpec:
    def test_question_spec_shape(self):
        spec = breakout_question_spec()
        entry = spec["paddle_direction"]
        assert entry["type"] == "choice"
        assert entry["options"] == ["left", "right", "stay"]
        assert set(entry["option_descriptions"]) == {"left", "right", "stay"}

    def test_ball_motion_spec_shares_options_with_paddle(self):
        """Identical option texts are what make the question text the only
        discriminator between the two breakout questions."""
        spec = breakout_question_spec()
        assert spec[MOTION_QID]["options"] == spec[PADDLE_QID]["options"]
        assert spec[MOTION_QID]["type"] == "choice"


class TestZeroVelocityRendering:
    def test_vx_zero_states_render_across_templates(self):
        """vx=0 (ball_motion 'stay') must render validly for every template,
        with lexically neutral zero-motion wording, and fields must stay
        splittable.

        The neutral wording is load-bearing: phrases like "holding its
        horizontal position" embed near the "stay" option ("hold" synonym
        included), giving the stay query a side-independent match in the
        geometry field — contradictory supervision that stalls head
        training at uniform attention (verified: v2 paddle probe val 0.33
        → 0.99 with the neutral phrasing).
        """
        from decision_lab.states.fields import split_state_fields
        from decision_lab.states.breakout import BreakoutLatents, _RENDERERS

        banned = ("holding its", "holds its", "level vertically")
        for template in BREAKOUT_TEMPLATES:
            rng = Random(11)
            for side in ("left", "center", "right"):
                lat = BreakoutLatents(
                    side=side, gap_px=10 if side != "center" else 5,
                    ball_vx=0, ball_vy=2, bricks=10, score=50,
                    lives=2, template=template,
                )
                text, fields = _RENDERERS[template](rng, lat)
                assert fields == split_state_fields(text), (
                    f"template={template} side={side} text={text!r}"
                )
                if template not in ("structured", "log"):
                    assert "drifting" not in text
                    for phrase in banned:
                        assert phrase not in text, f"{phrase!r} in {text!r}"
                # only the full-prose template spells out neutral horizontal
                # motion; short/vague omit it entirely when vx == 0
                if template == "prose" and side != "center":
                    assert "not moving left or right" in text


class TestDatasetIO:
    def test_generate_split_files_and_determinism(self):
        cfg = load_config("configs/default.yaml")
        cfg = replace(cfg,
                      generator=replace(cfg.generator,
                                        num_breakout_train=30, num_breakout_test=10))
        with tempfile.TemporaryDirectory() as tmp:
            generate_breakout_dataset(cfg, Path(tmp) / "a", Random(101))
            generate_breakout_dataset(cfg, Path(tmp) / "b", Random(101))
            for name in ("train_breakout", "test_breakout"):
                a = (Path(tmp) / "a" / f"{name}.jsonl").read_text()
                b = (Path(tmp) / "b" / f"{name}.jsonl").read_text()
                assert a == b
            states = load_dataset(Path(tmp) / "a" / "train_breakout.jsonl")
            assert len(states) == 30
            assert all(s.state_type.startswith("breakout_") for s in states)
            assert all(
                set(s.labels) == {PADDLE_QID, MOTION_QID,
                                  "is_ball_left", "ball_moving", "ball_rising"}
                for s in states
            )

    def test_jsonl_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "roundtrip.jsonl"
            states = _states(5)
            save_dataset(states, path)
            assert load_dataset(path) == states
