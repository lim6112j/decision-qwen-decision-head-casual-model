"""Tests for decision head model."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from decision_lab.head.model import (
    DecisionHead, get_device, save_head, load_head, create_random_head,
)


class TestDecisionHead:
    def test_forward_shape(self):
        model = DecisionHead(input_dim=1024, hidden_dim=256, num_actions=5)
        x = torch.randn(16, 1024)
        out = model(x)
        assert out.shape == (16, 5)

    def test_batch_size_one(self):
        model = DecisionHead(input_dim=1024, hidden_dim=256, num_actions=5)
        x = torch.randn(1, 1024)
        out = model(x)
        assert out.shape == (1, 5)

    def test_logits_not_softmax(self):
        model = DecisionHead(input_dim=1024, hidden_dim=256, num_actions=5)
        x = torch.randn(8, 1024)
        out = model(x)
        # logits can be negative, should sum to not 1
        assert not torch.allclose(out.sum(dim=1), torch.ones(8))

    def test_save_and_load(self):
        model = DecisionHead(input_dim=1024, hidden_dim=256, num_actions=5)
        model.eval()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "head.pt"
            save_head(model, str(path))
            loaded = load_head(str(path), input_dim=1024, hidden_dim=256, num_actions=5)
            loaded.eval()

            x = torch.randn(4, 1024)
            with torch.no_grad():
                out_orig = model(x)
                out_loaded = loaded(x)
            assert torch.allclose(out_orig, out_loaded, atol=1e-6)

    def test_random_vs_trained_different(self):
        """Random init heads should produce different outputs."""
        r1 = create_random_head(input_dim=1024, hidden_dim=256, num_actions=5)
        r2 = create_random_head(input_dim=1024, hidden_dim=256, num_actions=5)
        x = torch.randn(4, 1024)
        # With random init, they almost certainly differ
        assert not torch.allclose(r1(x), r2(x), atol=1e-6)

    def test_get_device_returns_valid(self):
        device = get_device()
        assert isinstance(device, torch.device)
        # MPS reports as mps:0 vs mps string mismatch — verify type not exact string
        test_tensor = torch.zeros(1).to(device)
        assert test_tensor.device.type == device.type

    def test_dropout_disabled_in_eval(self):
        model = DecisionHead(input_dim=1024, hidden_dim=256, num_actions=5, dropout=0.5)
        model.eval()
        x = torch.randn(8, 1024)
        out1 = model(x)
        out2 = model(x)
        assert torch.allclose(out1, out2)  # deterministic in eval