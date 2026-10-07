"""Episode engine: run an agent through a gridworld, one decision per step.

Framework-free so it can be unit-tested with a scripted agent.
"""

from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterator, Optional

from decision_lab.env.gridworld import ACTION_NAMES, INF, GridLayout, GridState


@dataclass
class StepResult:
    """One decision step in an episode."""

    step_idx: int
    agent_pos: tuple[int, int]
    action: Optional[int]          # predicted action; None if parse failed
    action_name: Optional[str]
    optimal_action: int
    optimal_action_name: str
    correct: bool
    latency_ms: float
    raw_output: str
    new_pos: tuple[int, int]
    reached_goal: bool
    state_text: str                # rendered grid the decision was made on


@dataclass
class EpisodeSummary:
    """Final outcome of an episode."""

    success: bool                  # goal reached within max_steps
    steps_used: int
    optimal_steps: int             # BFS distance from start to goal
    correct_actions: int
    total_actions: int             # steps with a parsed action
    parse_failures: int

    @property
    def action_accuracy(self) -> float:
        return self.correct_actions / self.total_actions if self.total_actions else 0.0


DecideFn = Callable[[GridState], tuple[Optional[int], str, float]]
"""Returns (action_index_or_None, raw_output, latency_ms)."""


def pick_start_pos(layout: GridLayout) -> tuple[int, int]:
    """Pick the empty cell with the largest finite BFS distance to the goal."""
    distances = layout.bfs_distances()
    best_pos = layout.goal_pos
    best_dist = 0
    for r, c in product(range(layout.rows), range(layout.cols)):
        if layout.cells[r, c] != 1 and (r, c) != layout.goal_pos:
            if distances[r, c] < INF and distances[r, c] > best_dist:
                best_dist = distances[r, c]
                best_pos = (r, c)
    return best_pos


def run_episode(
    layout: GridLayout,
    start_pos: tuple[int, int],
    decide_fn: DecideFn,
    max_steps: int,
) -> Iterator[StepResult]:
    """Yield one StepResult per step until the goal is reached or budget spent."""
    state = GridState(layout=layout, agent_pos=start_pos)

    for step_idx in range(max_steps):
        if state.is_terminal:
            return

        action, raw_output, latency_ms = decide_fn(state)
        optimal = state.label

        new_state = state.step(4 if action is None else action)
        yield StepResult(
            step_idx=step_idx,
            agent_pos=state.agent_pos,
            action=action,
            action_name=ACTION_NAMES[action] if action is not None else None,
            optimal_action=optimal,
            optimal_action_name=ACTION_NAMES[optimal] if optimal is not None else "none",
            correct=(action == optimal),
            latency_ms=latency_ms,
            raw_output=raw_output,
            new_pos=new_state.agent_pos,
            reached_goal=new_state.is_terminal,
            state_text=state.render(),
        )
        state = new_state


def summarize(steps: list[StepResult], optimal_steps: int) -> EpisodeSummary:
    """Aggregate step results into an episode summary."""
    reached = bool(steps) and steps[-1].reached_goal
    return EpisodeSummary(
        success=reached,
        steps_used=len(steps),
        optimal_steps=optimal_steps,
        correct_actions=sum(1 for s in steps if s.correct),
        total_actions=sum(1 for s in steps if s.action is not None),
        parse_failures=sum(1 for s in steps if s.action is None),
    )
