"""Stateless player similarity search backed by FAISS.

Environment variables define resource locations to avoid hard-coded paths:
- PLAYER_FAISS_INDEX_PATH
- PLAYER_EMBEDDINGS_PATH (.npy or .npz)
- PLAYER_METADATA_PATH (.json, .jsonl, or .csv)

Assumptions: metadata rows align with embedding rows; FAISS index was built on
the same embedding matrix. Scores are returned as provided by FAISS (distance
or similarity depending on index type)."""

from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import faiss
except ImportError as exc:  # pragma: no cover - defensive guard
    raise ImportError(
        "faiss is required for player similarity search. Please install faiss."
    ) from exc

logger = logging.getLogger(__name__)

INDEX_ENV = "PLAYER_FAISS_INDEX_PATH"
EMBEDDINGS_ENV = "PLAYER_EMBEDDINGS_PATH"
METADATA_ENV = "PLAYER_METADATA_PATH"

IDENTIFIER_KEYS = ("player_id", "id", "uuid")


def _env_path(env_var: str) -> Path:
    """Resolve a required environment variable to a Path."""
    value = os.getenv(env_var)
    if not value:
        raise RuntimeError(f"Environment variable {env_var} is required but not set.")
    return Path(value)


def _load_embeddings(path: Path) -> np.ndarray:
    """Load embeddings from .npy or .npz."""
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found at {path}")
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        first_key = next(iter(data.files))
        return data[first_key]
    raise ValueError(f"Unsupported embeddings format: {path.suffix}")


def _load_metadata(path: Path) -> List[Dict[str, Any]]:
    """Load metadata records from json, jsonl, or csv."""
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found at {path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            if not isinstance(data, list):
                raise ValueError("JSON metadata must be a list of objects.")
            return [dict(item) for item in data]
    if suffix == ".jsonl":
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                records.append(json.loads(line))
        return records
    if suffix == ".csv":
        with path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            return [dict(row) for row in reader]
    raise ValueError(f"Unsupported metadata format: {path.suffix}")


def _extract_id(record: Dict[str, Any]) -> str:
    """Extract a player identifier from metadata."""
    for key in IDENTIFIER_KEYS:
        value = record.get(key)
        if value:
            return str(value)
    raise ValueError("Metadata record missing a player identifier.")


def _load_resources() -> Tuple[faiss.Index, np.ndarray, List[Dict[str, Any]], Dict[str, int]]:
    """Load FAISS index, embeddings, metadata, and identifier lookup."""
    index_path = _env_path(INDEX_ENV)
    embeddings_path = _env_path(EMBEDDINGS_ENV)
    metadata_path = _env_path(METADATA_ENV)

    logger.info(
        "Loading similarity resources",
        extra={"index_path": str(index_path), "embeddings_path": str(embeddings_path), "metadata_path": str(metadata_path)},
    )

    index = faiss.read_index(str(index_path))
    embeddings = _load_embeddings(embeddings_path).astype("float32", copy=False)
    metadata = _load_metadata(metadata_path)

    if embeddings.shape[0] != len(metadata):
        raise ValueError(
            f"Embeddings count ({embeddings.shape[0]}) does not match metadata rows ({len(metadata)})."
        )
    if index.d != embeddings.shape[1]:
        raise ValueError(
            f"FAISS index dimension ({index.d}) does not match embedding dimension ({embeddings.shape[1]})."
        )

    id_lookup: Dict[str, int] = {}
    for row_idx, record in enumerate(metadata):
        pid = _extract_id(record)
        if pid in id_lookup:
            raise ValueError(f"Duplicate player identifier detected: {pid}")
        id_lookup[pid] = row_idx

    return index, embeddings, metadata, id_lookup


def _build_justification(source: Dict[str, Any], target: Dict[str, Any]) -> str:
    """Construct a deterministic justification from shared attributes."""
    reasons: List[str] = []
    if source.get("team") and source.get("team") == target.get("team"):
        reasons.append(f"same team ({source['team']})")
    if source.get("league") and source.get("league") == target.get("league"):
        reasons.append(f"compete in {source['league']}")
    if source.get("position") and source.get("position") == target.get("position"):
        reasons.append(f"play as {source['position']}")
    if source.get("nationality") and source.get("nationality") == target.get("nationality"):
        reasons.append(f"share nationality ({source['nationality']})")

    def _age(value: Any) -> int | None:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    age_src = _age(source.get("age"))
    age_tgt = _age(target.get("age"))
    if age_src is not None and age_tgt is not None and abs(age_src - age_tgt) <= 1:
        reasons.append("similar age band")

    if source.get("foot") and source.get("foot") == target.get("foot"):
        reasons.append(f"prefer {source['foot']}-foot")

    if reasons:
        return "; ".join(reasons)
    return "Closest match based on embedding similarity with limited shared metadata."


def find_similar_players(player_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Return a list of similar players with scores and deterministic justifications."""
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    index, embeddings, metadata, id_lookup = _load_resources()
    if player_id not in id_lookup:
        raise ValueError(f"Player ID {player_id!r} not found in metadata.")

    logger.info("Finding similar players", extra={"player_id": player_id, "top_k": top_k})

    query_idx = id_lookup[player_id]
    query_vector = embeddings[query_idx].astype("float32")[None, :]
    scores, indices = index.search(query_vector, min(top_k + 1, len(metadata)))

    results: List[Dict[str, Any]] = []
    for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
        if idx == -1:
            continue
        candidate_id = _extract_id(metadata[idx])
        if candidate_id == player_id:
            continue
        justification = _build_justification(metadata[query_idx], metadata[idx])
        results.append(
            {
                "player_id": candidate_id,
                "score": float(score),
                "justification": justification,
            }
        )
        if len(results) >= top_k:
            break

    return results


__all__ = ["find_similar_players"]
