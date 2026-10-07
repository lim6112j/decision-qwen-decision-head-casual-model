"""Decision Head model: MLP classifier on frozen backbone embeddings."""

import torch
import torch.nn as nn


class DecisionHead(nn.Module):
    """Small MLP that maps 1024-dim embeddings → action logits."""

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        num_actions: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, input_dim) → (batch, num_actions) logits."""
        return self.net(x)


def get_device() -> torch.device:
    """Best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_head(model: DecisionHead, path: str) -> None:
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, path)


def load_head(path: str, **kwargs) -> DecisionHead:
    model = DecisionHead(**kwargs)
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model


def create_random_head(**kwargs) -> DecisionHead:
    """Create a head with random init weights (no training)."""
    return DecisionHead(**kwargs)