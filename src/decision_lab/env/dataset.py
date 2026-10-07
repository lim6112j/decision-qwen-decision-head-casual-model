"""Synthetic dataset: gridworld states → optimal action labels."""

import json
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from random import Random
from typing import Optional

from decision_lab.config import Config
from decision_lab.env.gridworld import GridLayout, GridState, ACTION_NAMES

STATE_SEED = 0x_DEC1DE


@dataclass
class Dataset:
    """States + labels for a collection of gridworld layouts."""

    states: list[GridState] = field(default_factory=list)
    layouts: list[GridLayout] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> GridState:
        return self.states[idx]


def _sample_states(
    layout: GridLayout,
    max_per_layout: int,
    rng: Random,
) -> list[GridState]:
    """Sample non-terminal, solvable states from a layout."""
    empty_cells = [
        (r, c)
        for r, c in product(range(layout.rows), range(layout.cols))
        if layout.cells[r, c] != 1 and (r, c) != layout.goal_pos
    ]
    valid = []
    for pos in empty_cells:
        s = GridState(layout=layout, agent_pos=pos)
        if s.label is not None:
            valid.append(s)
    rng.shuffle(valid)
    return valid[:max_per_layout]


def generate_gridworld_data(cfg: Config, data_dir: Path) -> None:
    """Generate train / in-dist test / heldout test datasets and save as JSONL."""
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = Random(STATE_SEED)

    splits = _generate_splits(cfg, rng)

    for name, layouts in splits.items():
        rows_out = _layouts_to_jsonl(layouts, rng)
        path = data_dir / f"{name}.jsonl"
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows_out))
        print(f"  {name}: {len(rows_out)} states across {len(layouts)} layouts → {path}")


def _generate_splits(
    cfg: Config, rng: Random
) -> dict[str, list[GridLayout]]:
    """Create train, indist_test, heldout_test layout collections."""
    g = cfg.grid

    def make_layouts(n: int, rows: int, cols: int, wd: float) -> list[GridLayout]:
        layouts = []
        for _ in range(n):
            ly = GridLayout.random(
                rows=rows, cols=cols, wall_density=wd,
                min_path_length=g.min_path_length, rng=rng,
            )
            layouts.append(ly)
        return layouts

    train = make_layouts(g.num_train_layouts, g.size, g.size, g.wall_density)
    indist = make_layouts(g.num_indist_test_layouts, g.size, g.size, g.wall_density)

    h_rng = g.heldout_size_min
    h_rmax = g.heldout_size_max + 1
    heldout = []
    for _ in range(g.num_heldout_test_layouts):
        rs = rng.randint(h_rng, h_rmax - 1)
        cs = rng.randint(h_rng, h_rmax - 1)
        wd = g.wall_density + g.heldout_wall_extra
        heldout.append(GridLayout.random(rs, cs, wd, g.min_path_length, rng))

    return {"train": train, "test_indist": indist, "test_heldout": heldout}


def _layouts_to_jsonl(
    layouts: list[GridLayout], rng: Random, max_per_layout: int = 30
) -> list[dict]:
    """Convert layouts to JSONL rows: state text, rendered map, label action, layout metadata."""
    rows = []
    for i, ly in enumerate(layouts):
        states = _sample_states(ly, max_per_layout, rng)
        for s in states:
            label = s.label
            assert label is not None, f"unreachable state {s.agent_pos}"
            rows.append({
                "layout_id": i,
                "layout_rows": ly.rows,
                "layout_cols": ly.cols,
                "goal_pos": list(ly.goal_pos),
                "agent_pos": list(s.agent_pos),
                "text": s.render(),
                "action": int(label),
                "action_name": ACTION_NAMES[label],
            })
    return rows


def load_dataset(path: Path) -> list[GridState]:
    """Load JSONL into list of GridState objects (for inference/eval)."""
    from decision_lab.env.gridworld import GridLayout
    import numpy as np

    state_cache: dict[int, GridLayout] = {}
    states = []
    for line in path.read_text().strip().splitlines():
        if not line:
            continue
        d = json.loads(line)
        lid = d["layout_id"]
        if lid not in state_cache:
            cells = np.zeros((d["layout_rows"], d["layout_cols"]), dtype=np.int32)
            gr, gc = d["goal_pos"]
            cells[gr, gc] = 2
            state_cache[lid] = GridLayout(
                rows=d["layout_rows"], cols=d["layout_cols"],
                cells=cells, goal_pos=(gr, gc),
            )
        states.append(GridState(
            layout=state_cache[lid],
            agent_pos=tuple(d["agent_pos"]),
        ))
    return states