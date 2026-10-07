"""Train the TypedDecisionHead on pre-extracted features.

One training pass optimizes the sum of cross-entropy losses over all question
heads. After training, per-question temperatures are fitted on a held-out
slice (disjoint from train/val) and stored in the checkpoint.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from decision_lab.config import Config
from decision_lab.head.calibration import fit_all_temperatures
from decision_lab.head.model import (
    TypedDecisionHead,
    get_device,
    save_head,
)


def train_head(
    features: np.ndarray,
    labels_by_qid: dict[str, np.ndarray],
    question_spec: dict,
    cfg: Config,
    model_path: Path,
) -> TypedDecisionHead:
    """Train a TypedDecisionHead on (features, per-question labels).

    Args:
        features: (N, input_dim) float32 embedding vectors.
        labels_by_qid: {qid: (N,) int class indices} for every bank question.
        question_spec: normalized question bank (see head.model.build_question_spec).
        cfg: full Config.
        model_path: where to save the checkpoint (state + temperatures + spec).
    """
    hc = cfg.head
    device = get_device()
    print(f"Training typed head on device: {device} ({len(question_spec)} questions)")

    # 3-way split: calibration holdout first, then 80/20 train/val.
    n = len(features)
    idx = np.random.RandomState(cfg.generator.seed).permutation(n)
    calib_n = max(1, int(n * cfg.calibration.holdout_fraction))
    calib_idx, rest = idx[:calib_n], idx[calib_n:]
    split = int(len(rest) * 0.8)
    train_idx, val_idx = rest[:split], rest[split:]

    def tensors(rows: np.ndarray) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = torch.tensor(features[rows], dtype=torch.float32)
        ys = {qid: torch.tensor(labels_by_qid[qid][rows], dtype=torch.long) for qid in labels_by_qid}
        return x, ys

    x_train, y_train = tensors(train_idx)
    x_val, y_val = tensors(val_idx)

    train_loader = DataLoader(TensorDataset(x_train, *y_train.values()), batch_size=hc.batch_size, shuffle=True)
    qids = list(y_train.keys())
    val_batches = [(x_val[i:i + hc.batch_size], {q: y_val[q][i:i + hc.batch_size] for q in qids})
                   for i in range(0, len(x_val), hc.batch_size)]

    model = TypedDecisionHead(
        hidden_dim=hc.hidden_dim,
        question_spec=question_spec,
        dropout=hc.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=hc.learning_rate, weight_decay=hc.weight_decay)

    def multi_head_loss(outputs: dict, ys: dict[str, torch.Tensor]) -> torch.Tensor:
        loss = None
        for qid, logits in outputs.items():
            q_loss = F.cross_entropy(logits, ys[qid].to(logits.device))
            loss = q_loss if loss is None else loss + q_loss
        return loss

    best_val_acc = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, hc.max_epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            xb = batch[0].to(device)
            ys = {qid: batch[i + 1].to(device) for i, qid in enumerate(qids)}
            optimizer.zero_grad()
            loss = multi_head_loss(model(xb), ys)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_correct: dict[str, int] = {qid: 0 for qid in qids}
        val_total = 0
        with torch.no_grad():
            for xb, ys in val_batches:
                outputs = model(xb.to(device))
                for qid, logits in outputs.items():
                    pred = logits.argmax(dim=1).cpu()
                    val_correct[qid] += int((pred == ys[qid]).sum().item())
                val_total += len(xb)
        per_q_acc = {qid: val_correct[qid] / val_total for qid in qids}
        mean_acc = sum(per_q_acc.values()) / len(per_q_acc)

        if epoch == 1 or epoch % 10 == 0 or epoch == hc.max_epochs:
            accs = "  ".join(f"{qid}={acc:.3f}" for qid, acc in per_q_acc.items())
            print(f"  epoch {epoch:3d}: train_loss={train_loss:.4f}  val_mean_acc={mean_acc:.3f}  ({accs})")

        if mean_acc > best_val_acc:
            best_val_acc = mean_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= hc.patience:
            print(f"  early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)

    print("Fitting per-question temperatures on calibration holdout:")
    temperatures = fit_all_temperatures(model, features, labels_by_qid, cfg, holdout_idx=calib_idx)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    save_head(model, str(model_path), temperatures=temperatures)
    print(f"  saved to {model_path}  (best val_mean_acc={best_val_acc:.3f})")
    return model


def encode_labels(states, question_spec: dict) -> dict[str, np.ndarray]:
    """Convert TextState gold labels into per-question int class arrays.

    choice → index into spec options; score → level index; noul → 0/1.
    """
    import numpy as _np

    labels_by_qid: dict[str, list] = {qid: [] for qid in question_spec}
    for s in states:
        for qid in question_spec:
            labels_by_qid[qid].append(s.labels[qid])

    encoded = {}
    for qid, spec in question_spec.items():
        raw = labels_by_qid[qid]
        kind = spec["type"]
        if kind == "choice":
            key_to_idx = {opt: i for i, opt in enumerate(spec["options"])}
            encoded[qid] = _np.array([key_to_idx[v] for v in raw], dtype=np.int64)
        elif kind == "score":
            encoded[qid] = _np.array([int(v) for v in raw], dtype=np.int64)
        elif kind == "noul":
            encoded[qid] = _np.array([int(bool(v)) for v in raw], dtype=np.int64)
        else:
            raise ValueError(f"unknown question type '{kind}'")
    return encoded