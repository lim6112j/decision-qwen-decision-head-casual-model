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


def _state_rows(n: int) -> list[str]:
    return [
        json.dumps({
            "doc_id": i,
            "state_type": "text",
            "text": f"Note {i}: informational update, no action needed.",
            "labels": {
                "sentiment": "neutral", "urgency": "low", "quality": 2,
                "is_actionable": False, "contains_pii": False, "is_urgent": False,
            },
        })
        for i in range(n)
    ]


class TestExtractFeatures:
    def test_extract_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            data_path = tmp_dir / "test_indist.jsonl"
            data_path.write_text("\n".join(_state_rows(3)))

            cache_path = tmp_dir / "features_test.npz"
            feats = extract_features(data_path, cache_path, MockServer(), batch_size=2)
            assert feats.shape == (3, 1024)
            assert feats.dtype == np.float32

            feats2 = extract_features(data_path, cache_path, MockServer(), batch_size=2)
            assert np.array_equal(feats, feats2)
            assert cache_path.exists()