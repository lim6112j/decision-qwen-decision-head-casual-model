"""Deterministic gridworld environment with BFS shortest-path labels."""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from itertools import product
from random import Random
from typing import Optional

import numpy as np

# Actions: 0=up, 1=down, 2=left, 3=right, 4=wait
ACTION_NAMES: tuple[str, ...] = ("up", "down", "left", "right", "wait")

ACTION_DELTAS: tuple[tuple[int, int], ...] = (
    (-1, 0),  # up
    (1, 0),   # down
    (0, -1),  # left
    (0, 1),   # right
    (0, 0),   # wait
)

Cell = int
EMPTY: Cell = 0
WALL: Cell = 1
GOAL: Cell = 2


@dataclass
class GridLayout:
    """A gridworld map (walls + goal) without agent position."""

    rows: int
    cols: int
    cells: np.ndarray   # shape (rows, cols), values: 0=empty 1=wall 2=goal
    goal_pos: tuple[int, int]

    @classmethod
    def random(
        cls,
        rows: int,
        cols: int,
        wall_density: float,
        min_path_length: int,
        rng: Random,
    ) -> "GridLayout":
        """Generate a random layout guaranteeing a solvable path exists."""
        while True:
            cells = np.zeros((rows, cols), dtype=np.int32)
            # place walls
            for r in range(rows):
                for c in range(cols):
                    if rng.random() < wall_density:
                        cells[r, c] = WALL

            # pick goal in empty cell
            empty = [(r, c) for r, c in product(range(rows), range(cols)) if cells[r, c] == EMPTY]
            if not empty:
                continue
            goal = tuple(rng.choice(empty))
            cells[goal] = GOAL

            layout = cls(rows=rows, cols=cols, cells=cells, goal_pos=goal)
            # ensure at least one start position has path >= min_path_length
            distances = layout.bfs_distances()
            if np.any((distances >= min_path_length) & (distances < INF)):
                return layout
            # else retry

    def bfs_distances(self) -> np.ndarray:
        """BFS from goal to all cells. INF for unreachable/walls."""
        dist = np.full((self.rows, self.cols), INF, dtype=np.int32)
        dist[self.goal_pos] = 0
        q = deque([self.goal_pos])
        while q:
            r, c = q.popleft()
            for dr, dc in ACTION_DELTAS[:-1]:  # exclude wait
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if self.cells[nr, nc] != WALL and dist[nr, nc] == INF:
                        dist[nr, nc] = dist[r, c] + 1
                        q.append((nr, nc))
        return dist

    def optimal_action(self, pos: tuple[int, int]) -> Optional[int]:
        """BFS shortest path first step from `pos` to goal. None if unreachable."""
        distances = self.bfs_distances()
        if distances[pos] == INF:
            return None
        if pos == self.goal_pos:
            return 4  # wait
        best_action = 4
        best_dist = distances[pos]
        for a in range(4):  # up down left right (exclude wait)
            dr, dc = ACTION_DELTAS[a]
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                if distances[nr, nc] < best_dist:
                    best_dist = distances[nr, nc]
                    best_action = a
        return best_action

    def to_text(self, agent_pos: tuple[int, int]) -> str:
        """Render grid as ASCII. '.'=empty, '#'=wall, 'G'=goal, 'A'=agent."""
        lines = []
        for r in range(self.rows):
            row_chars = []
            for c in range(self.cols):
                if (r, c) == agent_pos:
                    row_chars.append("A")
                elif self.cells[r, c] == WALL:
                    row_chars.append("#")
                elif self.cells[r, c] == GOAL:
                    row_chars.append("G")
                else:
                    row_chars.append(".")
            lines.append("".join(row_chars))
        lines.append(f"Agent: ({agent_pos[0]}, {agent_pos[1]})")
        lines.append(f"Goal: ({self.goal_pos[0]}, {self.goal_pos[1]})")
        return "\n".join(lines)


@dataclass
class GridState:
    """A single decision point: layout + agent position."""

    layout: GridLayout
    agent_pos: tuple[int, int]

    @property
    def label(self) -> Optional[int]:
        """Optimal action index (0-4) per BFS. None if unreachable."""
        return self.layout.optimal_action(self.agent_pos)

    @property
    def is_terminal(self) -> bool:
        return self.agent_pos == self.layout.goal_pos

    def render(self) -> str:
        return self.layout.to_text(self.agent_pos)

    def step(self, action: int) -> "GridState":
        """Return new state after applying action. Does NOT mutate."""
        if action == 4:  # wait
            return GridState(layout=self.layout, agent_pos=self.agent_pos)
        dr, dc = ACTION_DELTAS[action]
        nr = self.agent_pos[0] + dr
        nc = self.agent_pos[1] + dc
        if 0 <= nr < self.layout.rows and 0 <= nc < self.layout.cols and self.layout.cells[nr, nc] != WALL:
            return GridState(layout=self.layout, agent_pos=(nr, nc))
        return GridState(layout=self.layout, agent_pos=self.agent_pos)


INF = 10**9