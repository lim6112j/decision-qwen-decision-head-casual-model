"""Feature extraction from llama-server embeddings, cached to .npz.

Two cache formats:
- features_*.npz: one pooled (input_dim,) vector per state (v1; typed head
  and legacy dynamic head).
- features_*_fields.npz: ragged field-set cache — ``features``
  (ΣM_i, input_dim) stacked + ``field_counts`` (N,) so state i owns rows
  [offset_i : offset_i + M_i]. Field splitting goes through
  states/fields.state_fields, the single source of truth shared with
  inference.
"""

from pathlib import Path
from typing import Sequence

import numpy as np

from decision_lab.config import Config


def extract_features(
    data_path: Path,
    cache_path: Path,
    server,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract embeddings for all states in dataset, cache to .npz.

    Args:
        data_path: JSONL dataset file.
        cache_path: where to save/load .npz cache.
        server: LlamaServer instance (must be running).
        batch_size: texts to batch per embedding request.
    """
    from decision_lab.states.dataset import load_dataset

    fingerprint = _cache_fingerprint(data_path, include_summary_field=False)
    if cache_path.exists():
        data = np.load(cache_path)
        stored = str(data["fingerprint"]) if "fingerprint" in data else ""
        if stored == fingerprint:
            return data["features"]
        print(f"  cache stale (source changed) — re-extracting {cache_path.name}")

    states = load_dataset(data_path)
    texts = [s.render() for s in states]
    all_embeds = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeds = server.embed(batch)
        all_embeds.extend(embeds)

    features = np.array(all_embeds, dtype=np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, features=features, fingerprint=fingerprint)
    return features


def _cache_fingerprint(data_path: Path, include_summary_field: bool) -> str:
    """Fingerprint tying a feature cache to its source dataset + settings.

    A stale cache silently returning embeddings for a DIFFERENT dataset is
    catastrophic (training pairs embeddings of one text set with labels of
    another) and nearly invisible — the counts barely differ. Fingerprint
    the source file bytes plus every setting that changes the output.
    """
    import hashlib

    h = hashlib.sha256()
    h.update(data_path.read_bytes())
    h.update(f"|include_summary_field={include_summary_field}".encode())
    return h.hexdigest()


def extract_field_features(
    data_path: Path,
    cache_path: Path,
    server,
    batch_size: int = 32,
    include_summary_field: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract field-set embeddings for all states, cache to ragged .npz.

    Field splitting goes through ``state_field_set`` — the same function
    inference uses, so training-time chunking always matches run-time
    chunking.

    The cache carries a fingerprint of the source dataset; a mismatch
    (dataset regenerated, setting changed) triggers re-extraction instead
    of silently serving stale embeddings.

    Returns:
        (features, field_counts): features is (ΣM_i, input_dim) stacked
        row-major; field_counts is (N,) with state i owning rows
        [offset_i : offset_i + M_i].
    """
    from decision_lab.states.dataset import load_dataset
    from decision_lab.states.fields import state_field_set

    fingerprint = _cache_fingerprint(data_path, include_summary_field)
    if cache_path.exists():
        data = np.load(cache_path)
        stored = str(data["fingerprint"]) if "fingerprint" in data else ""
        if stored == fingerprint:
            return data["features"], data["field_counts"]
        print(f"  cache stale (source changed) — re-extracting {cache_path.name}")

    states = load_dataset(data_path)

    # Gather each state's fields, dedupe texts across the whole split
    field_sets = [
        state_field_set(state, include_summary_field) for state in states
    ]
    unique_texts: dict[str, int] = {}
    for fields in field_sets:
        for text in fields:
            if text not in unique_texts:
                unique_texts[text] = len(unique_texts)

    # Embed unique texts in batches
    text_list = list(unique_texts)
    emb_by_text: dict[str, np.ndarray] = {}
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i : i + batch_size]
        embeds = server.embed(batch)
        for text, emb in zip(batch, embeds):
            emb_by_text[text] = np.asarray(emb, dtype=np.float32)

    # Stack ragged: state i's rows are contiguous
    chunks = []
    field_counts = np.zeros(len(states), dtype=np.int64)
    for i, fields in enumerate(field_sets):
        emb = np.stack([emb_by_text[t] for t in fields])
        chunks.append(emb)
        field_counts[i] = len(fields)

    features = np.concatenate(chunks, axis=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path, features=features, field_counts=field_counts,
        fingerprint=fingerprint,
    )
    print(f"  {len(states)} states, {int(field_counts.sum())} fields "
          f"({len(unique_texts)} unique texts)")
    return features, field_counts


def load_field_features(cache_path: Path) -> tuple[list[np.ndarray], np.ndarray]:
    """Load a ragged field cache → (per-state (M_i, input_dim) arrays, field_counts)."""
    data = np.load(cache_path)
    features = data["features"]
    field_counts = data["field_counts"]
    field_sets = []
    offset = 0
    for count in field_counts:
        field_sets.append(features[offset : offset + int(count)])
        offset += int(count)
    return field_sets, field_counts


def collate_field_batch(
    field_sets: Sequence[np.ndarray],
    device,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Pad a list of (M_i, input_dim) arrays → (B, M_max, D) + (B, M_max) mask."""
    import torch

    batch = len(field_sets)
    max_m = max(int(fs.shape[0]) for fs in field_sets)
    dim = field_sets[0].shape[-1]
    xb = torch.zeros(batch, max_m, dim, dtype=torch.float32)
    mask = torch.zeros(batch, max_m, dtype=torch.bool)
    for i, fs in enumerate(field_sets):
        m = int(fs.shape[0])
        xb[i, :m] = torch.from_numpy(np.asarray(fs, dtype=np.float32))
        mask[i, :m] = True
    return xb.to(device), mask.to(device)


def embed_options(server, option_texts: list[str], batch_size: int = 64) -> dict[str, np.ndarray]:
    """Embed option/value texts via the frozen backbone.

    Args:
        server: LlamaServer instance (must be running).
        option_texts: list of unique option text strings.
        batch_size: texts to batch per embedding request.

    Returns:
        {text: (input_dim,) float32 embedding vector}
    """
    result = {}
    for i in range(0, len(option_texts), batch_size):
        batch = option_texts[i : i + batch_size]
        embs = server.embed(batch)
        for text, emb in zip(batch, embs):
            result[text] = np.array(emb, dtype=np.float32)
    return result