"""Feature extraction from llama-server embeddings, cached to .npz."""

from pathlib import Path

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

    if cache_path.exists():
        data = np.load(cache_path)
        return data["features"]

    states = load_dataset(data_path)
    texts = [s.render() for s in states]
    all_embeds = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeds = server.embed(batch)
        all_embeds.extend(embeds)

    features = np.array(all_embeds, dtype=np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, features=features)
    return features


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