"""Tests for feature extraction (uses mock server, no real llama-server)."""

import json
import tempfile
from pathlib import Path

import numpy as np

from decision_lab.backbone.features import (
    collate_field_batch,
    extract_features,
    extract_field_features,
    load_field_features,
)


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


class TestExtractFieldFeatures:
    def test_ragged_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            data_path = tmp_dir / "split.jsonl"
            data_path.write_text("\n".join(_state_rows(3)))

            cache_path = tmp_dir / "features_split_fields.npz"
            feats, counts = extract_field_features(
                data_path, cache_path, MockServer(),
                include_summary_field=True,
            )
            # Each state: 1 summary (whole text) + 1 sentence field, plus the
            # degenerate-set guard: the text is a single sentence ("Note i: ..."
            # has no ". " break"), so summary duplicates the only field — the
            # guard appends the clause split (3 parts at ", " and ": ") for
            # 5 fields total.
            assert counts.tolist() == [5, 5, 5]
            assert feats.shape == (15, 1024)
            assert cache_path.exists()

            field_sets, counts2 = load_field_features(cache_path)
            assert counts2.tolist() == [5, 5, 5]
            assert len(field_sets) == 3
            for fs in field_sets:
                assert fs.shape == (5, 1024)
            # State i owns rows offset_i : offset_i + M_i (contiguous layout)
            assert np.array_equal(feats[5:10], field_sets[1])

    def test_dedupes_shared_texts(self):
        """Identical fields across states are embedded once."""
        server = MockServer()

        class CountingServer:
            n_calls = 0
            def embed(self, texts):
                self.n_calls += len(texts)
                return [[float(len(t))] * 1024 for t in texts]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            identical = json.dumps({
                "doc_id": 0,
                "state_type": "text",
                "text": "Same sentence for every state. Second sentence.",
                "labels": {"sentiment": "neutral"},
            })
            data_path = tmp_dir / "split.jsonl"
            data_path.write_text("\n".join([identical] * 5))
            counted = CountingServer()
            extract_field_features(data_path, tmp_dir / "f.npz", counted,
                                   include_summary_field=False)
        # 5 states × same 2 sentences → only 2 unique texts embedded
        assert counted.n_calls == 2

    def test_collate_field_batch_pads_with_mask(self):
        import torch

        sets = [np.ones((2, 4), dtype=np.float32), np.ones((1, 4), dtype=np.float32) * 2]
        xb, mask = collate_field_batch(sets, device=torch.device("cpu"))
        assert xb.shape == (2, 2, 4)
        assert mask.tolist() == [[True, True], [True, False]]
        assert xb[1, 1].sum().item() == 0.0  # padded rows are zeros
        assert torch.equal(xb[1, 0], torch.full((4,), 2.0))