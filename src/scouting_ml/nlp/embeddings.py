"""Lightweight player embedding utilities used for similarity search.

This module intentionally avoids Hugging Face dependencies. Embeddings are
derived from basic structured attributes to support local FAISS indexing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

import faiss
import numpy as np

logger = logging.getLogger(__name__)

# Common positional groups to anchor deterministic vectors.
_POSITION_VOCAB = ("GK", "CB", "FB", "DM", "CM", "WM", "AM", "FW")


def _one_hot(value: str, vocab: Iterable[str]) -> List[float]:
    """Return a one-hot list for a value within the provided vocab."""
    mapping = {v: i for i, v in enumerate(vocab)}
    vec = [0.0] * len(mapping)
    if value in mapping:
        vec[mapping[value]] = 1.0
    return vec


def _safe_float(value: Any) -> float:
    """Best-effort float parsing with zero fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def encode_player_profile(player: Dict[str, Any]) -> np.ndarray:
    """Generate a deterministic numeric embedding from basic player attributes."""
    position = str(player.get("position", "")).upper()
    age = _safe_float(player.get("age")) / 40.0
    minutes = _safe_float(player.get("minutes") or player.get("minutes_played")) / 5000.0
    goals = _safe_float(player.get("goals")) / 30.0
    assists = _safe_float(player.get("assists")) / 30.0

    pos_vec = _one_hot(position, _POSITION_VOCAB)
    numeric_vec = [age, minutes, goals, assists]
    vector = np.asarray(numeric_vec + pos_vec, dtype="float32")
    return vector


def build_faiss_index(embeddings: np.ndarray, metric: str = "cosine") -> faiss.Index:
    """Build a FAISS index for the given embeddings."""
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")

    vectors = embeddings.astype("float32", copy=False)
    dim = vectors.shape[1]

    if metric == "cosine":
        # Normalize for inner-product cosine similarity.
        faiss.normalize_L2(vectors)
        index: faiss.Index = faiss.IndexFlatIP(dim)
    elif metric in {"l2", "euclidean"}:
        index = faiss.IndexFlatL2(dim)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    index.add(vectors)
    return index


def save_faiss_index(index: faiss.Index, path: Path) -> None:
    """Persist a FAISS index to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))
    logger.info("Saved FAISS index to %s", path)


__all__ = ["encode_player_profile", "build_faiss_index", "save_faiss_index"]
