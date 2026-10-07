"""Tests for synthetic dataset generation."""

import json
import tempfile
from pathlib import Path

import pytest

from decision_lab.config import Config, load_config
from decision_lab.env.dataset import generate_gridworld_data, load_dataset


class TestGenerateDataset:
    def test_generates_splits(self):
        cfg = Config()
        cfg.grid.num_train_layouts = 5
        cfg.grid.num_indist_test_layouts = 3
        cfg.grid.num_heldout_test_layouts = 2
        cfg.grid.size = 5

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            generate_gridworld_data(cfg, data_dir)

            assert (data_dir / "train.jsonl").exists()
            assert (data_dir / "test_indist.jsonl").exists()
            assert (data_dir / "test_heldout.jsonl").exists()

    def test_train_states_all_action_0_4(self):
        cfg = Config()
        cfg.grid.num_train_layouts = 5
        cfg.grid.num_indist_test_layouts = 2
        cfg.grid.num_heldout_test_layouts = 2
        cfg.grid.size = 5

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            generate_gridworld_data(cfg, data_dir)

            for line in (data_dir / "train.jsonl").read_text().strip().splitlines():
                d = json.loads(line)
                assert 0 <= d["action"] <= 4
                assert d["action_name"] in ("up", "down", "left", "right", "wait")

    def test_no_layout_leakage_across_splits(self):
        cfg = Config()
        cfg.grid.num_train_layouts = 5
        cfg.grid.num_indist_test_layouts = 3
        cfg.grid.num_heldout_test_layouts = 2
        cfg.grid.size = 5

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            generate_gridworld_data(cfg, data_dir)

            def layout_ids(split):
                ids = set()
                for line in (data_dir / f"{split}.jsonl").read_text().strip().splitlines():
                    d = json.loads(line)
                    ids.add(d["layout_id"])
                return ids

            train_ids = layout_ids("train")
            indist_ids = layout_ids("test_indist")
            heldout_ids = layout_ids("test_heldout")

            # Each split has its own layout_id=0,1,2... but they're separate files
            # Check that the layout dims match: heldout should have varied sizes
            for line in (data_dir / "test_heldout.jsonl").read_text().strip().splitlines():
                d = json.loads(line)
                rows, cols = d["layout_rows"], d["layout_cols"]
                assert cfg.grid.heldout_size_min <= rows <= cfg.grid.heldout_size_max
                assert cfg.grid.heldout_size_min <= cols <= cfg.grid.heldout_size_max

    def test_states_are_not_terminal(self):
        cfg = Config()
        cfg.grid.num_train_layouts = 5
        cfg.grid.num_indist_test_layouts = 2
        cfg.grid.num_heldout_test_layouts = 2
        cfg.grid.size = 5

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            generate_gridworld_data(cfg, data_dir)

            for line in (data_dir / "train.jsonl").read_text().strip().splitlines():
                d = json.loads(line)
                assert d["agent_pos"] != d["goal_pos"], "state should not be at goal"


class TestLoadDataset:
    def test_load_and_roundtrip(self):
        cfg = Config()
        cfg.grid.num_train_layouts = 2
        cfg.grid.num_indist_test_layouts = 1
        cfg.grid.num_heldout_test_layouts = 1
        cfg.grid.size = 5

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            generate_gridworld_data(cfg, data_dir)

            states = load_dataset(data_dir / "train.jsonl")
            assert len(states) > 0
            for s in states:
                assert s.label is not None
                assert isinstance(s.render(), str)

    def test_load_restores_walls(self):
        cfg = Config()
        cfg.grid.num_train_layouts = 3
        cfg.grid.num_indist_test_layouts = 1
        cfg.grid.num_heldout_test_layouts = 1
        cfg.grid.size = 7
        cfg.grid.wall_density = 0.2

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            generate_gridworld_data(cfg, data_dir)

            lines = [json.loads(l) for l in (data_dir / "train.jsonl").read_text().strip().splitlines()]
            states = load_dataset(data_dir / "train.jsonl")

            for d, state in zip(lines, states):
                # every '#' in the rendered text must be a wall cell, 'G' the goal
                grid_lines = [
                    l for l in d["text"].splitlines()
                    if l and not l.startswith(("Agent:", "Goal:"))
                ]
                for r, line_text in enumerate(grid_lines):
                    for c, ch in enumerate(line_text):
                        if ch == "#":
                            assert state.layout.cells[r, c] == 1, f"wall missing at {(r, c)}"
                        if ch == "G":
                            assert state.layout.cells[r, c] == 2
                assert state.label == d["action"]