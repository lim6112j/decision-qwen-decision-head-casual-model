"""Tests for DynamicDecisionHead — attention-based slot filling.

Covers: architecture, variable option counts, permutation invariance,
decoding, checkpoint roundtrip, correctness checks.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from decision_lab.head.dynamic_model import (
    D_K,
    DEFAULT_TEMPERATURE,
    DynamicDecisionHead,
    create_random_dynamic_head,
    decode_dynamic_answer,
    dynamic_probabilities,
    get_device,
    is_correct_dynamic,
    load_dynamic_head,
    make_choice_question,
    make_noul_question,
    make_score_question,
    predict_dynamic,
    question_num_classes,
    question_option_texts,
    save_dynamic_head,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def head():
    """Fresh DynamicDecisionHead with small dimensions for fast tests."""
    return DynamicDecisionHead(input_dim=64, hidden_dim=32, d_k=16, dropout=0.0)


@pytest.fixture
def state_emb():
    """Single state embedding vector."""
    return torch.randn(64)


@pytest.fixture
def batch_state_emb():
    """Batch of 4 state embeddings."""
    return torch.randn(4, 64)


@pytest.fixture
def choice_3opts():
    """Dynamic choice question with 3 options."""
    return make_choice_question(["first", "second", "third"])


@pytest.fixture
def score_5levels():
    """Dynamic score question with 5 levels."""
    return make_score_question(["Bottom", "Low", "Mid", "High", "Top"])


@pytest.fixture
def noul_q():
    """Dynamic noul question."""
    return make_noul_question("Is this ready for production?")


# ---------------------------------------------------------------------------
# Question config helpers
# ---------------------------------------------------------------------------


class TestQuestionConfig:
    def test_make_choice(self):
        q = make_choice_question(["a", "b", "c"], "Which one?")
        assert q["type"] == "choice"
        assert q["options"] == ["a", "b", "c"]

    def test_make_score(self):
        q = make_score_question(["Low", "High"])
        assert q["type"] == "score"
        assert q["levels"] == ["Low", "High"]

    def test_make_noul(self):
        q = make_noul_question("Is it done?")
        assert q["type"] == "noul"

    def test_option_texts_choice(self):
        q = make_choice_question(["x", "y"])
        assert question_option_texts(q) == ["x", "y"]

    def test_option_texts_score(self):
        q = make_score_question(["A", "B", "C"])
        assert question_option_texts(q) == ["A", "B", "C"]

    def test_option_texts_noul(self):
        q = make_noul_question("done?")
        assert question_option_texts(q) == ["false", "true"]

    def test_num_classes(self):
        assert question_num_classes(make_choice_question(["a", "b"])) == 2
        assert question_num_classes(make_score_question(["p", "q", "r"])) == 3
        assert question_num_classes(make_noul_question("x?")) == 2


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------


class TestDynamicHeadArchitecture:
    def test_init_stores_dims(self, head):
        assert head.input_dim == 64
        assert head.hidden_dim == 32
        assert head.d_k == 16

    def test_trunk_is_sequential(self, head):
        assert isinstance(head.trunk, torch.nn.Sequential)
        assert len(head.trunk) == 3  # Linear, ReLU, Dropout

    def test_query_proj_shape(self, head):
        assert head.query_proj.in_features == 32
        assert head.query_proj.out_features == 16

    def test_key_proj_shape(self, head):
        assert head.key_proj.in_features == 64
        assert head.key_proj.out_features == 16

    def test_noul_head_shape(self, head):
        assert head.noul_head.in_features == 32
        assert head.noul_head.out_features == 2


class TestDynamicHeadForward:
    def test_forward_choice_single_state(self, head, state_emb, choice_3opts):
        """Single state, 3 options → (1, 3) logits."""
        opt_embs = torch.randn(3, 64)  # 3 options, each 64-dim
        scores = head.forward_choice(state_emb, opt_embs)
        assert scores.shape == (1, 3)

    def test_forward_choice_batch(self, head, batch_state_emb, choice_3opts):
        """Batch of 4 states, 3 options → (4, 3) logits."""
        opt_embs = torch.randn(4, 3, 64)  # (batch, n_opts, dim)
        scores = head.forward_choice(batch_state_emb, opt_embs)
        assert scores.shape == (4, 3)

    def test_forward_choice_variable_options(self, head, state_emb):
        """2, 5, 10 options all work without architecture change."""
        for n in [2, 5, 10]:
            opt_embs = torch.randn(n, 64)
            scores = head.forward_choice(state_emb, opt_embs)
            assert scores.shape == (1, n), f"failed for n_opts={n}"

    def test_forward_score_same_as_choice_mechanism(self, head, state_emb):
        """Score uses the same attention mechanism as choice."""
        opt_embs = torch.randn(4, 64)
        head.eval()
        with torch.no_grad():
            c = head.forward_choice(state_emb, opt_embs)
            s = head.forward_score(state_emb, opt_embs)
        assert torch.equal(c, s), "choice and score should use identical mechanism"

    def test_forward_noul_single(self, head, state_emb):
        scores = head.forward_noul(state_emb)
        assert scores.shape == (1, 2)

    def test_forward_noul_batch(self, head, batch_state_emb):
        scores = head.forward_noul(batch_state_emb)
        assert scores.shape == (4, 2)

    def test_forward_choice_1d_option_embs_broadcast(self, head, batch_state_emb):
        """2D option embeddings (n_opts, D) broadcast across batch dim."""
        opt_embs = torch.randn(3, 64)  # not batched
        scores = head.forward_choice(batch_state_emb, opt_embs)
        assert scores.shape == (4, 3)

    def test_forward_1d_state_broadcast(self, head):
        """1D state (D,) broadcasts to (1, D)."""
        state = torch.randn(64)
        opt_embs = torch.randn(3, 64)
        scores = head.forward_choice(state, opt_embs)
        assert scores.shape == (1, 3)


# ---------------------------------------------------------------------------
# Probabilities
# ---------------------------------------------------------------------------


class TestDynamicProbabilities:
    def test_sums_to_one(self):
        scores = torch.randn(1, 5)
        probs = dynamic_probabilities(scores)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-5)

    def test_temperature_sharpens(self):
        scores = torch.tensor([[1.0, 2.0, 3.0]])
        hot = dynamic_probabilities(scores, temperature=0.1)
        cold = dynamic_probabilities(scores, temperature=10.0)
        assert hot.max() > cold.max(), "lower temperature should concentrate mass"

    def test_temperature_preserves_argmax(self):
        torch.manual_seed(0)
        scores = torch.randn(4, 7) * 3
        hot = dynamic_probabilities(scores, 0.3)
        cold = dynamic_probabilities(scores, 8.0)
        assert torch.equal(hot.argmax(dim=-1), cold.argmax(dim=-1))


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


class TestDecodeDynamicAnswer:
    def test_decode_choice(self, choice_3opts):
        # Use large logit differences so softmax ≈ one-hot
        scores = torch.tensor([1.0, 10.0, 1.0])
        d = decode_dynamic_answer(choice_3opts, scores)
        assert d["predicted"] == "second"
        assert set(d["distribution"].keys()) == {"first", "second", "third"}
        assert abs(sum(d["distribution"].values()) - 1.0) < 1e-5

    def test_decode_score(self, score_5levels):
        scores = torch.tensor([1.0, 1.0, 1.0, 10.0, 1.0])
        d = decode_dynamic_answer(score_5levels, scores)
        assert d["predicted"] == 3  # index 3 = "High"
        # All mass should be on predicted class
        assert d["distribution"]["High"] > 0.95

    def test_decode_score_expected(self, score_5levels):
        """Expected value is the probability-weighted average of level indices."""
        # Equal scores → uniform distribution → expected = mean index
        scores = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        d = decode_dynamic_answer(score_5levels, scores)
        # Uniform over indices 0..4 → expected = 2.0
        assert abs(d["expected"] - 2.0) < 1e-4

    def test_decode_noul_true(self, noul_q):
        scores = torch.tensor([1.0, 10.0])
        d = decode_dynamic_answer(noul_q, scores)
        assert d["predicted"] is True
        assert d["distribution"]["true"] > 0.95
        assert d["distribution"]["false"] < 0.05
        assert d["distribution"]["true"] + d["distribution"]["false"] == pytest.approx(1.0)

    def test_decode_noul_false(self, noul_q):
        scores = torch.tensor([10.0, 1.0])
        d = decode_dynamic_answer(noul_q, scores)
        assert d["predicted"] is False

    def test_decode_confidence(self, choice_3opts):
        scores = torch.tensor([1.0, 10.0, 1.0])
        d = decode_dynamic_answer(choice_3opts, scores)
        # Confidence is max probability after softmax — nearly 1.0 with these logits
        assert d["confidence"] > 0.95

    def test_decode_unknown_type_raises(self):
        with pytest.raises(ValueError):
            decode_dynamic_answer({"type": "unknown"}, torch.randn(3))


# ---------------------------------------------------------------------------
# Predict dynamic (integration)
# ---------------------------------------------------------------------------


class TestPredictDynamic:
    def test_predicts_all_questions(self, head, state_emb):
        questions = [
            make_choice_question(["a", "b", "c"]),
            make_score_question(["Low", "High"]),
            make_noul_question("done?"),
        ]
        cache = {
            "a": torch.randn(64),
            "b": torch.randn(64),
            "c": torch.randn(64),
            "Low": torch.randn(64),
            "High": torch.randn(64),
        }
        head.eval()
        results = predict_dynamic(state_emb, questions, cache, head)
        assert len(results) == 3
        assert isinstance(results[0]["predicted"], str)
        assert isinstance(results[1]["predicted"], int)
        assert isinstance(results[2]["predicted"], bool)


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


class TestIsCorrectDynamic:
    def test_choice_case_insensitive(self):
        q = make_choice_question(["Positive", "Negative"])
        assert is_correct_dynamic(q, "positive", "Positive")
        assert not is_correct_dynamic(q, "negative", "Positive")

    def test_score(self):
        q = make_score_question(["a", "b", "c"])
        assert is_correct_dynamic(q, 2, 2)
        assert not is_correct_dynamic(q, 1, 2)

    def test_noul(self):
        q = make_noul_question("x?")
        assert is_correct_dynamic(q, True, True)
        assert not is_correct_dynamic(q, False, True)

    def test_none_is_incorrect(self):
        q = make_choice_question(["a", "b"])
        assert not is_correct_dynamic(q, None, "a")


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


class TestDynamicCheckpoint:
    def test_save_load_roundtrip(self):
        head = DynamicDecisionHead(input_dim=64, hidden_dim=32, d_k=16, dropout=0.0)
        head.eval()
        state = torch.randn(64)
        opt_embs = torch.randn(3, 64)

        with torch.no_grad():
            orig_scores = head.forward_choice(state, opt_embs)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dynamic_head.pt"
            save_dynamic_head(head, path, temperature=2.5)
            loaded = load_dynamic_head(path)
            loaded.eval()

            assert loaded.input_dim == 64
            assert loaded.hidden_dim == 32
            assert loaded.d_k == 16
            assert loaded.temperature == 2.5

            with torch.no_grad():
                loaded_scores = loaded.forward_choice(state, opt_embs)
            assert torch.allclose(orig_scores, loaded_scores, atol=1e-6)

    def test_load_wrong_type_raises(self):
        """Loading a TypedDecisionHead checkpoint should fail cleanly."""
        import torch as _torch
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "not_dynamic.pt"
            _torch.save({"arch": {"type": "typed"}}, str(path))
            with pytest.raises(ValueError, match="not a DynamicDecisionHead"):
                load_dynamic_head(path)

    def test_random_heads_differ(self):
        h1 = create_random_dynamic_head(input_dim=64, hidden_dim=32)
        h2 = create_random_dynamic_head(input_dim=64, hidden_dim=32)
        x = torch.randn(64)
        opts = torch.randn(4, 64)
        with torch.no_grad():
            assert not torch.allclose(
                h1.forward_choice(x, opts), h2.forward_choice(x, opts)
            )


# ---------------------------------------------------------------------------
# Permutation invariance
# ---------------------------------------------------------------------------


class TestPermutationInvariance:
    """Option order should not affect which option is selected when semantics match."""

    def test_same_option_different_order_same_prediction(self, head):
        """If two option sets contain the same texts in different order,
        the head should select the same semantic option."""
        head.eval()
        state = torch.randn(64)

        # Two orderings of the same option texts
        opt_texts_a = ["alpha", "beta", "gamma"]
        opt_texts_b = ["gamma", "alpha", "beta"]  # rotated

        # Same embeddings in corresponding positions
        emb_alpha = torch.randn(64)
        emb_beta = torch.randn(64)
        emb_gamma = torch.randn(64)

        opt_embs_a = torch.stack([emb_alpha, emb_beta, emb_gamma])
        opt_embs_b = torch.stack([emb_gamma, emb_alpha, emb_beta])

        with torch.no_grad():
            scores_a = head.forward_choice(state, opt_embs_a)
            scores_b = head.forward_choice(state, opt_embs_b)

        # The best option text is the same regardless of position
        best_text_a = opt_texts_a[int(scores_a.argmax())]
        best_text_b = opt_texts_b[int(scores_b.argmax())]
        assert best_text_a == best_text_b, (
            f"different option selected: {best_text_a} vs {best_text_b}"
        )

    def test_variable_option_count_deterministic(self, head):
        """Adding an irrelevant option shouldn't change the relative ordering
        of existing options."""
        head.eval()
        torch.manual_seed(42)
        state = torch.randn(64)

        emb_a = torch.randn(64)
        emb_b = torch.randn(64)
        emb_irrelevant = torch.randn(64) * 0.01  # very small → should get low score

        with torch.no_grad():
            s2 = head.forward_choice(state, torch.stack([emb_a, emb_b]))
            s3 = head.forward_choice(state, torch.stack([emb_a, emb_b, emb_irrelevant]))

        # Relative order of a vs b should be preserved
        assert (s2[0, 0] > s2[0, 1]) == (s3[0, 0] > s3[0, 1])


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


def test_get_device_returns_valid():
    assert isinstance(get_device(), torch.device)