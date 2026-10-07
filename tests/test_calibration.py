"""Tests for temperature calibration."""

import torch

from decision_lab.head.calibration import fit_temperature, nll


class TestFitTemperature:
    def test_fit_reduces_nll(self):
        torch.manual_seed(0)
        # Overconfident logits: temperature should rise above 1.0
        logits = torch.randn(64, 3) * 3 + 4.0   # large margins → overconfident
        labels = torch.randint(0, 3, (64,))

        before = nll(logits, labels, 1.0)
        t = fit_temperature(logits, labels, lr=0.1, max_iter=200)
        after = nll(logits, labels, t)

        assert after <= before + 1e-4   # NLL never worsens materially
        assert t > 0.5

    def test_underconfident_logits_shrink_temperature(self):
        torch.manual_seed(1)
        # Tiny margins around the correct class → T should drop below 1.0
        labels = torch.randint(0, 3, (64,))
        logits = torch.randn(64, 3) * 0.05
        logits += torch.nn.functional.one_hot(labels, 3).float() * 0.5
        t = fit_temperature(logits, labels, lr=0.1, max_iter=200)
        assert t < 1.0

    def test_temperature_positive(self):
        torch.manual_seed(2)
        logits = torch.randn(32, 2)
        labels = torch.randint(0, 2, (32,))
        t = fit_temperature(logits, labels)
        assert t > 0.0

    def test_does_not_mutate_inputs(self):
        torch.manual_seed(3)
        logits = torch.randn(16, 2)
        labels = torch.randint(0, 2, (16,))
        logits_copy = logits.clone()
        fit_temperature(logits, labels)
        assert torch.equal(logits, logits_copy)