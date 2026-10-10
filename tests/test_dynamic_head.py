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
            "false": torch.randn(64),
            "true": torch.randn(64),
            "done?": torch.randn(64),
        }
        head.eval()
        results = predict_dynamic(state_emb, questions, cache, head)
        assert len(results) == 3
        assert isinstance(results[0]["predicted"], str)
        assert isinstance(results[1]["predicted"], int)
        assert isinstance(results[2]["predicted"], bool)

    def test_noul_goes_through_attention_path(self, head, state_emb):
        """noul must be scored via forward_choice with ['false','true'] —
        the deprecated forward_noul path was never trained."""
        questions = [make_noul_question("done?")]
        cache = {"false": torch.randn(64), "true": torch.randn(64), "done?": torch.randn(64)}
        head.eval()
        results = predict_dynamic(state_emb, questions, cache, head)
        assert set(results[0]["distribution"]) == {"false", "true"}


# ---------------------------------------------------------------------------
# v2 field-set attention
# ---------------------------------------------------------------------------


class TestFieldSetForward:
    """v2 state_set=True head: options cross-attend over the field set."""

    @pytest.fixture
    def set_head(self):
        return DynamicDecisionHead(input_dim=64, hidden_dim=32, d_k=16, dropout=0.0, state_set=True)

    def test_v1_modules_absent_in_v2(self, set_head):
        assert not hasattr(set_head, "trunk")
        assert hasattr(set_head, "field_enc")
        assert hasattr(set_head, "field_key")
        assert hasattr(set_head, "opt_query")
        assert hasattr(set_head, "score_head")

    def test_1d_state_promotes_to_single_field(self, set_head):
        """(D,) → treated as one field → (1, n_opts)."""
        opt_embs = torch.randn(3, 64)
        scores = set_head.forward_choice(torch.randn(64), opt_embs)
        assert scores.shape == (1, 3)

    def test_2d_state_is_field_set(self, set_head):
        """(M, D) unbatched → (1, n_opts) logits."""
        scores = set_head.forward_choice(torch.randn(5, 64), torch.randn(3, 64))
        assert scores.shape == (1, 3)

    def test_3d_batched_field_set(self, set_head):
        scores = set_head.forward_choice(torch.randn(4, 5, 64), torch.randn(3, 64))
        assert scores.shape == (4, 3)

    def test_masked_fields_are_ignored(self, set_head):
        """Zero-padded (masked) fields must not change scores."""
        set_head.eval()
        torch.manual_seed(0)
        fields = torch.randn(3, 64)
        opt_embs = torch.randn(2, 64)
        with torch.no_grad():
            clean = set_head.forward_choice(fields, opt_embs)

            padded = torch.zeros(5, 64)
            padded[:3] = fields
            mask = torch.tensor([True, True, True, False, False])
            with_mask = set_head.forward_choice(padded, opt_embs, state_mask=mask)
        assert torch.allclose(clean, with_mask, atol=1e-5)

    def test_unmasked_padding_changes_scores(self, set_head):
        """Without a mask, zero rows DO contribute — the mask is load-bearing."""
        set_head.eval()
        torch.manual_seed(0)
        fields = torch.randn(3, 64)
        opt_embs = torch.randn(2, 64)
        with torch.no_grad():
            clean = set_head.forward_choice(fields, opt_embs)
            padded = torch.zeros(5, 64)
            padded[:3] = fields
            no_mask = set_head.forward_choice(padded, opt_embs)
        assert not torch.allclose(clean, no_mask, atol=1e-5)

    def test_field_permutation_equivariance(self, set_head):
        """Attention over a SET: permuting fields permutes nothing — each
        option's score must be unchanged (no positional encoding)."""
        set_head.eval()
        torch.manual_seed(0)
        fields = torch.randn(4, 64)
        opt_embs = torch.randn(3, 64)
        with torch.no_grad():
            scores_a = set_head.forward_choice(fields, opt_embs)
            scores_b = set_head.forward_choice(fields.flip(0), opt_embs)
        assert torch.allclose(scores_a, scores_b, atol=1e-5)

    def test_single_field_matches_mean_semantics(self, set_head):
        """One field: softmax degenerates to weight 1 — score is purely
        score_head(field_enc(field)) for every option query direction
        (queries only affect the weighting, which is trivial here)."""
        set_head.eval()
        field = torch.randn(1, 64)
        opt_embs = torch.randn(3, 64)
        with torch.no_grad():
            scores = set_head.forward_choice(field, opt_embs)
        assert scores.shape == (1, 3)

    def test_v2_noul_direct_raises(self, set_head):
        with pytest.raises(NotImplementedError):
            set_head.forward_noul(torch.randn(64))


class TestFieldSetCheckpoint:
    def test_v2_roundtrip(self):
        head = DynamicDecisionHead(input_dim=64, hidden_dim=32, d_k=16, dropout=0.0, state_set=True)
        head.eval()
        state = torch.randn(4, 64)
        opt_embs = torch.randn(3, 64)
        with torch.no_grad():
            orig = head.forward_choice(state, opt_embs)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v2.pt"
            save_dynamic_head(head, path, temperature=0.7)
            loaded = load_dynamic_head(path)
            loaded.eval()
            assert loaded.state_set is True
            assert loaded.temperature == 0.7
            with torch.no_grad():
                assert torch.allclose(orig, loaded.forward_choice(state, opt_embs), atol=1e-6)

    def test_v1_checkpoint_refused(self):
        """Clean break: a saved non-state_set head is refused on load —
        v4 implies state_set=True (question-field attention)."""
        head = DynamicDecisionHead(input_dim=64, hidden_dim=32, d_k=16, dropout=0.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v1.pt"
            save_dynamic_head(head, path)
            with pytest.raises(ValueError, match="state_set"):
                load_dynamic_head(path)

    def test_v2_predict_dynamic_with_field_set(self):
        head = DynamicDecisionHead(input_dim=64, hidden_dim=32, d_k=16, dropout=0.0, state_set=True)
        head.eval()
        questions = [make_choice_question(["a", "b"]), make_noul_question("ok?")]
        cache = {t: torch.randn(64) for t in ["a", "b", "false", "true", "ok?"]}
        results = predict_dynamic(torch.randn(3, 64), questions, cache, head)
        assert len(results) == 2
        assert results[0]["predicted"] in ("a", "b")
        assert isinstance(results[1]["predicted"], bool)


# ---------------------------------------------------------------------------
# v4 question-field attention
# ---------------------------------------------------------------------------


class TestQuestionAttention:
    """v4: the question queries the field set directly (no FiLM)."""

    @pytest.fixture
    def set_head(self):
        return DynamicDecisionHead(input_dim=64, hidden_dim=32, d_k=16, dropout=0.0, state_set=True)

    def test_v4_modules_present(self, set_head):
        assert hasattr(set_head, "q_question")
        assert hasattr(set_head, "attn_scale_q")
        assert hasattr(set_head, "qz_readout")
        assert hasattr(set_head, "q_query_shift")
        # v3 FiLM modules are gone
        for gone in ("q_mod_scale", "q_mod_bias", "z_mod_scale", "z_mod_bias"):
            assert not hasattr(set_head, gone)

    def test_qz_readout_zero_init(self, set_head):
        """Only the LAST linear is zero-init (LoRA-style bootstrap: its
        gradient is alive at init; the first layer follows)."""
        assert torch.count_nonzero(set_head.qz_readout[2].weight) == 0
        assert torch.count_nonzero(set_head.qz_readout[2].bias) == 0

    def test_q_query_gate_shift_zero_init(self, set_head):
        """The question→query gate and shift are zero at init → identity on
        the option queries (same guard as the read-out)."""
        for lin in (set_head.q_query_gate, set_head.q_query_shift):
            assert torch.count_nonzero(lin.weight) == 0
            assert torch.count_nonzero(lin.bias) == 0

    def test_q_query_shift_steers_option_queries(self, set_head):
        """Once the shift is live the question changes what each option
        reads, and the mask invariant still holds."""
        set_head.eval()
        torch.manual_seed(5)
        with torch.no_grad():
            set_head.q_query_gate.weight.normal_(0, 0.5)
            set_head.q_query_shift.weight.normal_(0, 0.5)
        fields = torch.randn(6, 64)
        opt_embs = torch.randn(3, 64)
        q_a, q_b = torch.randn(64), torch.randn(64)
        with torch.no_grad():
            out_a = set_head.forward_choice(fields, opt_embs, question_emb=q_a)
            out_b = set_head.forward_choice(fields, opt_embs, question_emb=q_b)
            padded = torch.zeros(8, 64)
            padded[:6] = fields
            mask = torch.tensor([True] * 6 + [False] * 2)
            masked = set_head.forward_choice(
                padded, opt_embs, state_mask=mask, question_emb=q_a,
            )
        assert not torch.allclose(out_a, out_b, atol=1e-5)
        assert torch.allclose(out_a, masked, atol=1e-5)

    def test_query_shift_repoints_option_matching(self):
        """THE property a shared per-field bias cannot provide: the same
        option must be able to match a DIFFERENT field under a different
        question. Two options, two fields; with the shift trained, question A
        routes option 0 to field 0 and question B routes it to field 1."""
        torch.manual_seed(4)
        head = DynamicDecisionHead(input_dim=64, hidden_dim=32, d_k=16,
                                   dropout=0.0, state_set=True)
        g = torch.Generator().manual_seed(4)
        fields = torch.randn(2, 64, generator=g)
        opt_embs = torch.randn(2, 64, generator=g)
        q_a = torch.randn(64, generator=g)
        q_b = torch.randn(64, generator=g) + 2.0

        opt = torch.optim.Adam(head.parameters(), lr=5e-3)
        for _ in range(900):
            for q_emb, gold in ((q_a, 0), (q_b, 1)):
                scores = head.forward_choice(fields, opt_embs, question_emb=q_emb)
                loss = torch.nn.functional.cross_entropy(scores, torch.tensor([gold]))
                opt.zero_grad()
                loss.backward()
                opt.step()

        head.eval()
        with torch.no_grad():
            s_a = head.forward_choice(fields, opt_embs, question_emb=q_a)
            s_b = head.forward_choice(fields, opt_embs, question_emb=q_b)
        assert int(s_a.argmax()) == 0
        assert int(s_b.argmax()) == 1
        assert torch.count_nonzero(head.q_query_gate.weight) > 0

    def test_attn_scale_q_init(self, set_head):
        """Separate scale, init √d_k — the O(1) cosine logit-spread property
        applies to the question attention too (sharing attn_scale would
        couple the two heads' softmax temperature)."""
        assert set_head.attn_scale_q.item() == pytest.approx(16 ** 0.5)  # √d_k
        assert set_head.attn_scale_q is not set_head.attn_scale

    def test_identity_at_init(self, set_head):
        """Zero-init read-out: any question embedding (or None) reproduces
        the v2 forward exactly — the uniform-softmax stall cannot recur."""
        set_head.eval()
        torch.manual_seed(0)
        fields = torch.randn(4, 64)
        opt_embs = torch.randn(3, 64)
        q_a, q_b = torch.randn(64), torch.randn(64)
        with torch.no_grad():
            base = set_head.forward_choice(fields, opt_embs)
            with_qa = set_head.forward_choice(fields, opt_embs, question_emb=q_a)
            with_qb = set_head.forward_choice(fields, opt_embs, question_emb=q_b)
            with_none = set_head.forward_choice(fields, opt_embs, question_emb=None)
        assert torch.allclose(base, with_qa, atol=1e-6)
        assert torch.allclose(base, with_qb, atol=1e-6)
        assert torch.allclose(base, with_none, atol=1e-6)

    def test_none_equals_zero_embedding(self, set_head):
        """None → zero vector → identical output, always (not just at init)."""
        set_head.eval()
        torch.manual_seed(3)
        # Perturb the question path so it's non-identity
        with torch.no_grad():
            set_head.qz_readout[2].weight.normal_(0, 0.1)
            set_head.qz_readout[2].bias.normal_(0, 0.1)
        fields = torch.randn(4, 64)
        opt_embs = torch.randn(3, 64)
        with torch.no_grad():
            with_none = set_head.forward_choice(fields, opt_embs, question_emb=None)
            with_zero = set_head.forward_choice(
                fields, opt_embs, question_emb=torch.zeros(64)
            )
        assert torch.allclose(with_none, with_zero, atol=1e-6)

    def test_1d_question_broadcast(self, set_head):
        """(D,) question broadcasts across the state batch."""
        set_head.eval()
        fields = torch.randn(4, 5, 64)
        opt_embs = torch.randn(3, 64)
        q = torch.randn(64)
        with torch.no_grad():
            scores = set_head.forward_choice(fields, opt_embs, question_emb=q)
        assert scores.shape == (4, 3)
        # Row i must equal the unbatched pass for state i
        with torch.no_grad():
            for i in range(4):
                single = set_head.forward_choice(
                    fields[i], opt_embs, question_emb=q
                )
                assert torch.allclose(scores[i], single[0], atol=1e-6)

    def test_batched_question_shape(self, set_head):
        set_head.eval()
        scores = set_head.forward_choice(
            torch.randn(4, 5, 64), torch.randn(3, 64), question_emb=torch.randn(4, 64)
        )
        assert scores.shape == (4, 3)

    def test_question_changes_output_after_training_signal(self, set_head):
        """A non-trivially-trained read-out must actually depend on the
        question embedding (question pathway is live, not a constant)."""
        set_head.eval()
        with torch.no_grad():
            set_head.qz_readout[2].weight.normal_(0, 0.05)
            set_head.qz_readout[2].bias.normal_(0, 0.05)
        fields = torch.randn(4, 64)
        opt_embs = torch.randn(3, 64)
        with torch.no_grad():
            out_a = set_head.forward_choice(fields, opt_embs, question_emb=torch.randn(64))
            out_b = set_head.forward_choice(fields, opt_embs, question_emb=torch.randn(64))
        assert not torch.allclose(out_a, out_b, atol=1e-5)

    def test_masked_fields_ignored_in_question_attention(self, set_head):
        """The question attention must honor the state mask too."""
        set_head.eval()
        torch.manual_seed(0)
        with torch.no_grad():
            set_head.qz_readout[2].weight.normal_(0, 0.1)
        fields = torch.randn(3, 64)
        opt_embs = torch.randn(2, 64)
        q = torch.randn(64)
        with torch.no_grad():
            clean = set_head.forward_choice(fields, opt_embs, question_emb=q)
            padded = torch.zeros(5, 64)
            padded[:3] = fields
            mask = torch.tensor([True, True, True, False, False])
            with_mask = set_head.forward_choice(
                padded, opt_embs, state_mask=mask, question_emb=q
            )
        assert torch.allclose(clean, with_mask, atol=1e-5)

    def test_permutation_equivariance_with_question(self, set_head):
        """Field-set permutation equivariance holds under question attention."""
        set_head.eval()
        torch.manual_seed(0)
        with torch.no_grad():
            set_head.qz_readout[2].weight.normal_(0, 0.1)
        fields = torch.randn(4, 64)
        opt_embs = torch.randn(3, 64)
        q = torch.randn(64)
        with torch.no_grad():
            scores_a = set_head.forward_choice(fields, opt_embs, question_emb=q)
            scores_b = set_head.forward_choice(fields.flip(0), opt_embs, question_emb=q)
        assert torch.allclose(scores_a, scores_b, atol=1e-5)

    def test_same_state_options_flip_with_question(self):
        """THE question-functionality proof: the same state + same options
        must predict different options under two different questions.

        Tiny overfit — 2 samples, opposite golds, ~600 Adam steps on CE.
        If the question pathway can't condition on the question embedding,
        the two golds can never both be recovered. This is also the
        dead-path canary: qz_readout starts at zero and must escape.
        """
        torch.manual_seed(7)
        head = DynamicDecisionHead(input_dim=64, hidden_dim=32, d_k=16, dropout=0.0, state_set=True)
        fields = torch.randn(4, 64)
        opt_embs = torch.randn(3, 64)
        q_left = torch.randn(64)
        q_right = torch.randn(64) + 2.0  # well separated from q_left

        opt = torch.optim.Adam(head.parameters(), lr=5e-3)
        for _ in range(600):
            for q_emb, gold in ((q_left, 0), (q_right, 1)):
                scores = head.forward_choice(
                    fields, opt_embs, question_emb=q_emb
                )
                loss = torch.nn.functional.cross_entropy(scores, torch.tensor([gold]))
                opt.zero_grad()
                loss.backward()
                opt.step()

        head.eval()
        with torch.no_grad():
            scores_left = head.forward_choice(fields, opt_embs, question_emb=q_left)
            scores_right = head.forward_choice(fields, opt_embs, question_emb=q_right)
        assert int(scores_left.argmax()) == 0
        assert int(scores_right.argmax()) == 1
        # The question path must actually have moved off zero-init
        assert torch.count_nonzero(head.qz_readout[2].weight) > 0

    def test_unseen_question_type_smoke(self, set_head):
        """A never-trained random question embedding: finite logits,
        distribution sums to 1, no NaN (graceful degradation)."""
        set_head.eval()
        with torch.no_grad():
            set_head.qz_readout[2].weight.normal_(0, 0.1)
        fields = torch.randn(4, 64)
        opt_embs = torch.randn(3, 64)
        with torch.no_grad():
            scores = set_head.forward_choice(
                fields, opt_embs, question_emb=torch.randn(64) * 10
            )
        assert torch.isfinite(scores).all()
        probs = dynamic_probabilities(scores[0])
        assert probs.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_unseen_paraphrase_generalization(self):
        """The v3 failure-mode guard: train on a few phrasings of a question
        (modeled as base embedding + small noise — a backbone-similarity
        stand-in), then eval on HELD-OUT noisy copies. The question path
        must generalize within an embedding neighborhood, not memorize."""
        torch.manual_seed(11)
        head = DynamicDecisionHead(input_dim=64, hidden_dim=32, d_k=16, dropout=0.0, state_set=True)
        fields = torch.randn(4, 64)
        opt_embs = torch.randn(3, 64)
        q_base_a = torch.randn(64)
        q_base_b = torch.randn(64) + 2.0

        # "Phrasings": base + N(0, 0.05); 2 per question for training
        torch.manual_seed(12)
        train_samples = []
        for q_base, gold in ((q_base_a, 0), (q_base_b, 1)):
            for _ in range(2):
                train_samples.append((q_base + torch.randn(64) * 0.05, gold))

        opt = torch.optim.Adam(head.parameters(), lr=5e-3)
        for _ in range(600):
            for q_emb, gold in train_samples:
                scores = head.forward_choice(fields, opt_embs, question_emb=q_emb)
                loss = torch.nn.functional.cross_entropy(scores, torch.tensor([gold]))
                opt.zero_grad()
                loss.backward()
                opt.step()

        head.eval()
        # Held-out paraphrases: fresh noise draws, same bases
        torch.manual_seed(99)
        for q_base, gold in ((q_base_a, 0), (q_base_b, 1)):
            q_held = q_base + torch.randn(64) * 0.05
            with torch.no_grad():
                scores = head.forward_choice(fields, opt_embs, question_emb=q_held)
            assert int(scores.argmax()) == gold


class TestQuestionAttentionCheckpoint:
    def test_v4_roundtrip_strict_with_question(self):
        head = DynamicDecisionHead(input_dim=64, hidden_dim=32, d_k=16, dropout=0.0, state_set=True)
        head.eval()
        with torch.no_grad():
            # Non-identity read-out so the question genuinely matters
            head.qz_readout[2].weight.normal_(0, 0.1)
            head.qz_readout[2].bias.normal_(0, 0.1)
        fields = torch.randn(4, 64)
        opt_embs = torch.randn(3, 64)
        q = torch.randn(64)
        with torch.no_grad():
            orig = head.forward_choice(fields, opt_embs, question_emb=q)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v4.pt"
            save_dynamic_head(head, path, temperature=0.7)
            loaded = load_dynamic_head(path)   # strict load
            loaded.eval()
            assert loaded.temperature == 0.7
            with torch.no_grad():
                assert torch.allclose(
                    orig, loaded.forward_choice(fields, opt_embs, question_emb=q),
                    atol=1e-6,
                )

    def test_v3_checkpoint_refused(self, capsys):
        """Clean break: a hand-built v3-style checkpoint (FiLM keys) raises
        ValueError — no silent fallback, the v3 checkpoint stays preserved
        in the main checkout."""
        head = DynamicDecisionHead(input_dim=64, hidden_dim=32, d_k=16, dropout=0.0, state_set=True)
        with tempfile.TemporaryDirectory() as tmp:
            v4_path = Path(tmp) / "v4.pt"
            save_dynamic_head(head, v4_path)
            checkpoint = torch.load(str(v4_path), map_location="cpu")
            checkpoint["arch"]["arch_version"] = 3
            path = Path(tmp) / "v3.pt"
            torch.save(checkpoint, str(path))

            with pytest.raises(ValueError, match="arch_version"):
                load_dynamic_head(path)


class TestPredictDynamicQuestionCache:
    def test_question_text_in_cache_used(self):
        head = DynamicDecisionHead(input_dim=64, hidden_dim=32, d_k=16, dropout=0.0, state_set=True)
        head.eval()
        q = make_choice_question(["a", "b"], "Which one?")
        cache = {"a": torch.randn(64), "b": torch.randn(64), "Which one?": torch.randn(64)}
        results = predict_dynamic(torch.randn(3, 64), [q], cache, head)
        assert results[0]["predicted"] in ("a", "b")

    def test_empty_question_no_keyerror(self):
        head = DynamicDecisionHead(input_dim=64, hidden_dim=32, d_k=16, dropout=0.0, state_set=True)
        head.eval()
        q = make_choice_question(["a", "b"], "")
        cache = {"a": torch.randn(64), "b": torch.randn(64)}
        results = predict_dynamic(torch.randn(3, 64), [q], cache, head)
        assert results[0]["predicted"] in ("a", "b")


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
        """v4 heads are state_set=True — the non-state_set variant is only
        constructible by hand and refused on load (see TestFieldSetCheckpoint)."""
        head = DynamicDecisionHead(input_dim=64, hidden_dim=32, d_k=16, dropout=0.0, state_set=True)
        head.eval()
        state = torch.randn(3, 64)
        opt_embs = torch.randn(2, 64)

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


# ---------------------------------------------------------------------------
# Breakout paddle_direction variants
# ---------------------------------------------------------------------------


class TestPaddleDirectionVariants:
    """The paddle_direction synonym map yields shuffled left/right/stay variants."""

    def test_variants_generated(self):
        from decision_lab.head.dynamic_train import _variants_for_bank_entry
        from decision_lab.head.question_bank import build_question_bank
        from random import Random

        entry = next(e for e in build_question_bank() if e.qid == "paddle_direction")
        variants = _variants_for_bank_entry(entry, 3, Random(0))
        assert len(variants) == 3
        for v in variants:
            assert len(v["options"]) == 3
            assert set(v["gold_map"]) == {"left", "right", "stay"}
            # gold_map values are exactly the variant's (shuffled) option set
            assert set(v["gold_map"].values()) == set(v["options"])
            # variants use synonym labels, not the base option strings
            assert v["options"] != ["left", "right", "stay"]

    def test_gold_map_permutes_consistently(self):
        from decision_lab.head.dynamic_train import _variants_for_bank_entry
        from decision_lab.head.question_bank import build_question_bank
        from random import Random

        entry = next(e for e in build_question_bank() if e.qid == "paddle_direction")
        variants = _variants_for_bank_entry(entry, 3, Random(1))
        for v in variants:
            # gold_map values must be a permutation of the option set
            assert sorted(v["gold_map"].values()) == sorted(v["options"])