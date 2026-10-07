"""Tests for feature extraction (uses mock server, no real llama-server)."""

import json
import tempfile
from pathlib import Path

import numpy as np

from decision_lab.backbone.features import extract_features


class MockServer:
    """Mock llama-server for testing feature extraction."""
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t))] * 1024 for t in texts]


class TestExtractFeatures:
    def test_extract_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            data_dir = tmp_dir / "data"
            data_dir.mkdir()

            data_path = data_dir / "test_indist.jsonl"
            lines = [
                json.dumps({
                    "layout_id": 0, "layout_rows": 3, "layout_cols": 3,
                    "goal_pos": [2, 2], "agent_pos": [i, 0],
                    "text": f"...\n.A.\n..G\nAgent: ({i}, 0)\nGoal: (2, 2)",
                    "action": 1, "action_name": "down",
                })
                for i in range(3)
            ]
            data_path.write_text("\n".join(lines))

            cache_path = tmp_dir / "features_test.npz"
            server = MockServer()

            feats = extract_features(data_path, cache_path, server, batch_size=2)
            assert feats.shape == (3, 1024)
            assert feats.dtype == np.float32

            feats2 = extract_features(data_path, cache_path, server, batch_size=2)
            assert np.array_equal(feats, feats2)
            assert cache_path.exists()