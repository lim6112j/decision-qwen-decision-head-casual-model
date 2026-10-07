"""Post-hoc calibration: per-question temperature scaling (Guo et al. 2017).

Temperature scaling preserves the argmax ranking while minimizing NLL on a
held-out slice, fixing over/under-confidence. One scalar T per question.
"""

import numpy as np
import torch
import torch.nn.functional as F

from decision_lab.config import Config
from decision_lab.head.model import TypedDecisionHead, get_device


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    lr: float = 1e-2,
    max_iter: int = 1000,
) -> float:
    """Fit a scalar temperature by minimizing NLL on (logits, labels)."""
    # Fitting runs on CPU: the tensors are tiny and LBFGS params live on CPU.
    logits = logits.detach().to("cpu").clone()
    labels = labels.detach().to("cpu").clone()
    log_t = torch.zeros(1, requires_grad=True)  # optimize log T → T stays positive
    optimizer = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        t = log_t.exp().clamp_min(1e-3)
        loss = F.cross_entropy(logits / t, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


def nll(logits: torch.Tensor, labels: torch.Tensor, temperature: float) -> float:
    """Mean NLL of temperature-scaled logits (for before/after comparison)."""
    with torch.no_grad():
        return float(F.cross_entropy(logits / max(temperature, 1e-3), labels).item())


def fit_all_temperatures(
    model: TypedDecisionHead,
    features: np.ndarray,
    labels_by_qid: dict[str, np.ndarray],
    cfg: Config,
    holdout_idx: np.ndarray | None = None,
) -> dict[str, float]:
    """Fit one temperature per question on a held-out slice of features.

    Args:
        model: trained TypedDecisionHead (eval mode).
        features: (N, input_dim) embeddings of the training pool.
        labels_by_qid: {qid: (N,) int class indices}.
        cfg: provides holdout_fraction, lr, max_iter.
        holdout_idx: explicit row indices to fit on; if None, a random
            holdout_fraction slice is drawn (callers that already carved a
            calibration split should pass it to avoid train contamination).
    """
    device = get_device()
    model.eval()

    if holdout_idx is None:
        n = len(features)
        holdout = max(1, int(n * cfg.calibration.holdout_fraction))
        holdout_idx = np.random.RandomState(0).choice(n, size=holdout, replace=False)

    x = torch.tensor(features[holdout_idx], dtype=torch.float32, device=device)
    with torch.no_grad():
        outputs = model(x)

    temperatures = {}
    for qid, logits in outputs.items():
        labels = torch.tensor(
            labels_by_qid[qid][holdout_idx], dtype=torch.long, device=device
        )
        before = nll(logits, labels, 1.0)
        t = fit_temperature(logits, labels, lr=cfg.calibration.lr, max_iter=cfg.calibration.max_iter)
        after = nll(logits, labels, t)
        temperatures[qid] = t
        print(f"  calibration {qid}: T={t:.3f}  NLL {before:.4f} → {after:.4f}")
    return temperatures