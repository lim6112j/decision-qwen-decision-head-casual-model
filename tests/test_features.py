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

class TestBackboneFingerprint:
    """A cache extracted from one embedding backend must never be reused by
    another. Before the backend was folded into the fingerprint, swapping the
    served model (or its pooling) silently returned embeddings from the old
    one — the same failure class as the stale-source-cache incident."""

    class BackendServer:
        def __init__(self, backend_id: str, embed_dim: int = 1024, tag: float = 1.0):
            self.backend_id = backend_id
            self.embed_dim = embed_dim
            self.tag = tag
            self.n_calls = 0

        def embed(self, texts: list[str]) -> list[list[float]]:
            self.n_calls += len(texts)
            return [[self.tag] * self.embed_dim for _ in texts]

    def _write(self, tmp_dir: Path) -> Path:
        data_path = tmp_dir / "test_indist.jsonl"
        data_path.write_text("\n".join(_state_rows(3)))
        return data_path

    def test_same_backend_reuses_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            data_path = self._write(tmp_dir)
            cache = tmp_dir / "features_test.npz"

            first = self.BackendServer("bge-m3|pooling=cls|dim=1024")
            extract_features(data_path, cache, first)
            second = self.BackendServer("bge-m3|pooling=cls|dim=1024")
            extract_features(data_path, cache, second)

        assert second.n_calls == 0, "identical backend must reuse the cache"

    def test_different_backend_forces_reextraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            data_path = self._write(tmp_dir)
            cache = tmp_dir / "features_test.npz"

            qwen = self.BackendServer("qwen3.5-0.8b|pooling=last|dim=1024", tag=1.0)
            extract_features(data_path, cache, qwen)

            bge = self.BackendServer("bge-m3|pooling=cls|dim=1024", tag=2.0)
            feats = extract_features(data_path, cache, bge)

        assert bge.n_calls == 3, "a different backbone must not reuse the cache"
        assert feats[0][0] == 2.0, "cache must hold the new backend's vectors"

    def test_pooling_change_forces_reextraction(self):
        """Same model, different pooling → different vectors → new cache."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            data_path = self._write(tmp_dir)
            cache = tmp_dir / "features_test.npz"

            extract_features(data_path, cache, self.BackendServer("m|pooling=last"))
            mean = self.BackendServer("m|pooling=mean")
            extract_features(data_path, cache, mean)

        assert mean.n_calls == 3

    def test_dim_change_forces_reextraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            data_path = self._write(tmp_dir)
            cache = tmp_dir / "features_test.npz"

            extract_features(data_path, cache, self.BackendServer("m", embed_dim=1024))
            smaller = self.BackendServer("m", embed_dim=768)
            extract_features(data_path, cache, smaller)

        assert smaller.n_calls == 3

    def test_duck_typed_server_still_works(self):
        """Test doubles exposing only embed() must keep caching normally."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            data_path = self._write(tmp_dir)
            cache = tmp_dir / "features_test.npz"

            extract_features(data_path, cache, MockServer())
            feats = extract_features(data_path, cache, MockServer())

        assert feats.shape == (3, 1024)
