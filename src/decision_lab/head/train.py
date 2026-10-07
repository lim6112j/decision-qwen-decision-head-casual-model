"""Train Decision Head on pre-extracted features."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from decision_lab.config import Config
from decision_lab.head.model import DecisionHead, get_device, save_head


def train_head(
    features: np.ndarray,
    labels: np.ndarray,
    cfg: Config,
    model_path: Path,
) -> DecisionHead:
    """Train a DecisionHead on (features, labels).

    Args:
        features: (N, 1024) float32 embedding vectors.
        labels: (N,) int action indices 0..4.
        cfg: full Config.
        model_path: where to save best checkpoint.
    """
    hc = cfg.head
    device = get_device()
    print(f"Training head on device: {device}")

    # Split 80/20 for train/val
    n = len(features)
    idx = np.random.RandomState(42).permutation(n)
    split = int(n * 0.8)
    x_train = torch.tensor(features[idx[:split]], dtype=torch.float32)
    y_train = torch.tensor(labels[idx[:split]], dtype=torch.long)
    x_val = torch.tensor(features[idx[split:]], dtype=torch.float32)
    y_val = torch.tensor(labels[idx[split:]], dtype=torch.long)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=hc.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=hc.batch_size)

    model = DecisionHead(
        hidden_dim=hc.hidden_dim,
        num_actions=hc.num_actions,
        dropout=hc.dropout,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hc.learning_rate, weight_decay=hc.weight_decay)

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, hc.max_epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_loader.dataset)

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(dim=1)
                val_correct += (pred == yb).sum().item()
                val_total += len(yb)
        val_acc = val_correct / val_total

        if epoch == 1 or epoch % 10 == 0 or epoch == hc.max_epochs:
            print(f"  epoch {epoch:3d}: train_loss={train_loss:.4f}  val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= hc.patience:
            print(f"  early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    save_head(model, str(model_path))
    print(f"  saved to {model_path}  (best val_acc={best_val_acc:.3f})")
    return model