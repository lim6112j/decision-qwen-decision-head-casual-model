"""Tests for the web simulator episode engine."""

import numpy as np
import pytest
from random import Random

from decision_lab.env.gridworld import GOAL, GridLayout, GridState
from decision_lab.webapp.simulator import pick_start_pos, run_episode, summarize


def make_layout(rows=3, cols=3):
    cells = np.zeros((rows, cols), dtype=np.int32)
    cells[rows - 1, cols - 1] = GOAL
    return GridLayout(rows=rows, cols=cols, cells=cells, goal_pos=(rows - 1, cols - 1))


class TestPickStartPos:
    def test_picks_farthest_cell(self):
        layout = make_layout()
        # corner (0,0) is farthest from goal (2,2)
        assert pick_start_pos(layout) == (0, 0)

    def test_never_picks_wall_or_goal(self):
        rng = Random(42)
        layout = GridLayout.random(
            rows=7, cols=7, wall_density=0.3,
            min_path_length=5, rng=rng,
        )
        pos = pick_start_pos(layout)
        r, c = pos
        assert layout.cells[r, c] != 1
        assert pos != layout.goal_pos

    def test_never_picks_unreachable_cell(self):
        # (0,0) is walled off from the goal — must not be chosen despite
        # having INF distance (which would win a naive max()).
        cells = np.zeros((3, 3), dtype=np.int32)
        cells[0, 0] = 1
        cells[0, 1] = 1
        cells[1, 0] = 1
        cells[2, 2] = GOAL
        layout = GridLayout(rows=3, cols=3, cells=cells, goal_pos=(2, 2))
        assert pick_start_pos(layout) != (0, 0)
        assert layout.bfs_distances()[pick_start_pos(layout)] < 10**9


class TestRunEpisode:
    def test_optimal_agent_reaches_goal(self):
        layout = make_layout()
        start = (0, 0)
        distances = layout.bfs_distances()
        optimal_steps = int(distances[start])

        def optimal_decide(state):
            return state.label, "", 1.0

        steps = list(run_episode(layout, start, optimal_decide, max_steps=50))
        summary = summarize(steps, optimal_steps)

        assert summary.success
        assert summary.steps_used == optimal_steps
        assert summary.action_accuracy == 1.0

    def test_stops_at_max_steps(self):
        layout = make_layout()
        start = (0, 0)

        def wait_decide(state):
            return 4, "", 0.5  # wait forever

        steps = list(run_episode(layout, start, wait_decide, max_steps=5))

        assert len(steps) == 5
        summary = summarize(steps, optimal_steps=4)
        assert not summary.success

    def test_parse_failure_treated_as_wait(self):
        layout = make_layout()
        start = (1, 1)  # any move works here, wait should keep it in place

        def broken_decide(state):
            return None, "garbage output", 2.0

        steps = list(run_episode(layout, start, broken_decide, max_steps=1))

        assert len(steps) == 1
        assert steps[0].action is None
        assert steps[0].new_pos == steps[0].agent_pos  # didn't move
        assert steps[0].raw_output == "garbage output"
        summary = summarize(steps, optimal_steps=2)
        assert summary.parse_failures == 1
        assert summary.total_actions == 0

    def test_step_results_carry_optimal_labels(self):
        layout = make_layout()
        start = (0, 0)

        def always_wrong_decide(state):
            return 0, "", 0.1  # always up

        steps = list(run_episode(layout, start, always_wrong_decide, max_steps=2))

        for step in steps:
            assert step.optimal_action in (1, 3)  # down or right toward goal
            assert not step.correct

    def test_does_not_mutate_layout(self):
        layout = make_layout()
        start = (0, 0)
        cells_before = layout.cells.copy()

        def optimal_decide(state):
            return state.label, "", 0.0

        list(run_episode(layout, start, optimal_decide, max_steps=50))
        assert np.array_equal(layout.cells, cells_before)

    def test_zero_max_steps_yields_nothing(self):
        layout = make_layout()

        def optimal_decide(state):
            return state.label, "", 0.0

        assert list(run_episode(layout, (0, 0), optimal_decide, max_steps=0)) == []


class TestSummarize:
    def test_empty_steps(self):
        summary = summarize([], optimal_steps=3)
        assert not summary.success
        assert summary.action_accuracy == 0.0

    def test_parse_failure_counted(self):
        layout = make_layout()
        start = (0, 0)
        calls = {"count": 0}

        def half_broken(state):
            calls["count"] += 1
            if calls["count"] == 1:
                return None, "oops", 0.0
            return state.label, "", 0.0

        steps = list(run_episode(layout, start, half_broken, max_steps=20))
        summary = summarize(steps, optimal_steps=4)
        assert summary.parse_failures == 1
        assert summary.success  # recovers via optimal moves afterward
