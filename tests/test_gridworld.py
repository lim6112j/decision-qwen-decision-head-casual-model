"""Tests for gridworld environment."""

import pytest
from random import Random

from decision_lab.env.gridworld import (
    GridLayout, GridState, ACTION_NAMES, ACTION_DELTAS, EMPTY, WALL, GOAL,
)


class TestGridLayout:
    def test_random_creates_valid_layout(self):
        rng = Random(42)
        layout = GridLayout.random(
            rows=7, cols=7, wall_density=0.15,
            min_path_length=5, rng=rng,
        )
        assert layout.rows == 7
        assert layout.cols == 7
        assert layout.cells.shape == (7, 7)
        assert layout.cells[layout.goal_pos] == GOAL

    def test_bfs_distances_reachable(self):
        # Simple 3x3 layout: no walls, goal at (1,1)
        import numpy as np
        cells = np.zeros((3, 3), dtype=np.int32)
        cells[1, 1] = GOAL
        layout = GridLayout(rows=3, cols=3, cells=cells, goal_pos=(1, 1))
        dist = layout.bfs_distances()

        assert dist[0, 0] == 2  # corner needs two steps
        assert dist[1, 0] == 1
        assert dist[1, 1] == 0
        # corners of 3x3
        assert dist[0, 2] == 2
        assert dist[2, 0] == 2

    def test_bfs_distances_wall_unreachable(self):
        import numpy as np
        cells = np.zeros((3, 3), dtype=np.int32)
        cells[1, 1] = WALL
        cells[0, 2] = GOAL
        cells[0, 0] = WALL
        layout = GridLayout(rows=3, cols=3, cells=cells, goal_pos=(0, 2))
        dist = layout.bfs_distances()

        from decision_lab.env.gridworld import INF
        assert dist[1, 1] == INF  # wall unreachable
        assert dist[0, 0] == INF  # wall

    def test_optimal_action_points_toward_goal(self):
        import numpy as np
        # 3x3, goal at bottom right (2,2), agent at (0,0)
        cells = np.zeros((3, 3), dtype=np.int32)
        cells[2, 2] = GOAL
        layout = GridLayout(rows=3, cols=3, cells=cells, goal_pos=(2, 2))

        # BFS: (0,0) → needs to go down or right. First step: both equally good.
        # Tie-breaking picks direction with smallest distance.
        # distance (0,0)=4 (0,1→0,2→1,2→2,2 is 4, or 1,0→2,0→2,1→2,2 is 4)
        # down: (1,0) dist=3; right: (0,1) dist=3. Both equal.
        # Our tie-break scans actions in order up/down/left/right and picks first min.
        action = layout.optimal_action((0, 0))
        assert action is not None
        assert action in (1, 3)  # down or right

    def test_optimal_action_at_goal_waits(self):
        import numpy as np
        cells = np.zeros((3, 3), dtype=np.int32)
        cells[1, 1] = GOAL
        layout = GridLayout(rows=3, cols=3, cells=cells, goal_pos=(1, 1))
        action = layout.optimal_action((1, 1))
        assert action == 4  # wait

    def test_optimal_action_none_if_unreachable(self):
        import numpy as np
        # enclosed agent
        cells = np.zeros((3, 3), dtype=np.int32)
        cells[0, 0] = WALL
        cells[0, 1] = WALL
        cells[0, 2] = WALL
        cells[1, 0] = WALL
        cells[1, 1] = EMPTY  # trapped
        cells[1, 2] = WALL
        cells[2, 0] = WALL
        cells[2, 1] = WALL
        cells[2, 2] = GOAL
        layout = GridLayout(rows=3, cols=3, cells=cells, goal_pos=(2, 2))
        action = layout.optimal_action((1, 1))
        assert action is None

    def test_to_text_rendering(self):
        import numpy as np
        cells = np.zeros((3, 3), dtype=np.int32)
        cells[0, 0] = WALL
        cells[2, 2] = GOAL
        layout = GridLayout(rows=3, cols=3, cells=cells, goal_pos=(2, 2))
        text = layout.to_text(agent_pos=(1, 1))
        assert "A" in text
        assert "#" in text
        assert "G" in text
        assert "Agent:" in text
        assert "Goal:" in text


class TestGridState:
    def test_step_move_into_empty(self):
        import numpy as np
        cells = np.zeros((3, 3), dtype=np.int32)
        layout = GridLayout(rows=3, cols=3, cells=cells, goal_pos=(2, 2))
        state = GridState(layout=layout, agent_pos=(1, 1))
        new_state = state.step(0)  # up
        assert new_state.agent_pos == (0, 1)

    def test_step_move_into_wall_stays(self):
        import numpy as np
        cells = np.zeros((3, 3), dtype=np.int32)
        cells[0, 1] = WALL
        layout = GridLayout(rows=3, cols=3, cells=cells, goal_pos=(2, 2))
        state = GridState(layout=layout, agent_pos=(1, 1))
        new_state = state.step(0)  # up → blocked by wall
        assert new_state.agent_pos == (1, 1)

    def test_step_wait(self):
        import numpy as np
        cells = np.zeros((3, 3), dtype=np.int32)
        layout = GridLayout(rows=3, cols=3, cells=cells, goal_pos=(2, 2))
        state = GridState(layout=layout, agent_pos=(1, 1))
        new_state = state.step(4)  # wait
        assert new_state.agent_pos == (1, 1)

    def test_step_out_of_bounds_stays(self):
        import numpy as np
        cells = np.zeros((3, 3), dtype=np.int32)
        layout = GridLayout(rows=3, cols=3, cells=cells, goal_pos=(2, 2))
        state = GridState(layout=layout, agent_pos=(0, 0))
        new_state = state.step(0)  # up from row 0 → oob
        assert new_state.agent_pos == (0, 0)

    def test_step_does_not_mutate_original(self):
        import numpy as np
        cells = np.zeros((3, 3), dtype=np.int32)
        layout = GridLayout(rows=3, cols=3, cells=cells, goal_pos=(2, 2))
        state = GridState(layout=layout, agent_pos=(1, 1))
        _ = state.step(0)
        assert state.agent_pos == (1, 1)  # unchanged

    def test_is_terminal(self):
        import numpy as np
        cells = np.zeros((3, 3), dtype=np.int32)
        cells[2, 2] = GOAL
        layout = GridLayout(rows=3, cols=3, cells=cells, goal_pos=(2, 2))
        assert GridState(layout=layout, agent_pos=(2, 2)).is_terminal
        assert not GridState(layout=layout, agent_pos=(0, 0)).is_terminal

    def test_label_returns_action_index(self):
        import numpy as np
        cells = np.zeros((3, 3), dtype=np.int32)
        cells[2, 2] = GOAL
        layout = GridLayout(rows=3, cols=3, cells=cells, goal_pos=(2, 2))
        state = GridState(layout=layout, agent_pos=(2, 2))
        assert state.label == 4  # wait at goal


class TestActionConstants:
    def test_action_names_count(self):
        assert len(ACTION_NAMES) == 5

    def test_action_deltas_shape(self):
        assert len(ACTION_DELTAS) == 5
        assert ACTION_DELTAS[4] == (0, 0)  # wait