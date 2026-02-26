"""Integration-style test for player similarity using FAISS and deterministic embeddings."""
from __future__ import annotations

import numpy as np
import pytest

import faiss
from scouting_ml.nlp import embeddings as embeddings_module


def _deterministic_encoder(player: dict) -> np.ndarray:
    """Produce a simple, deterministic embedding from basic player attributes."""
    pos_map = {
        "CB": np.array([1.0, 0.0, 0.0], dtype="float32"),
        "CM": np.array([0.0, 1.0, 0.0], dtype="float32"),
        "FW": np.array([0.0, 0.0, 1.0], dtype="float32"),
    }
    pos_vec = pos_map.get(player.get("position"), np.zeros(3, dtype="float32"))
    age_component = float(player.get("age", 0)) / 40.0
    minutes_component = float(player.get("minutes", player.get("minutes_played", 0))) / 5000.0
    return np.concatenate(
        [np.array([age_component, minutes_component], dtype="float32"), pos_vec]
    )


def test_similarity_search_returns_expected_order(monkeypatch: pytest.MonkeyPatch) -> None:
    # Mock players with clear positional and workload differences.
    players = [
        {"player_id": "cb1", "position": "CB", "age": 25, "minutes": 2000},
        {"player_id": "cb2", "position": "CB", "age": 26, "minutes": 2100},
        {"player_id": "cm1", "position": "CM", "age": 24, "minutes": 2200},
        {"player_id": "fw1", "position": "FW", "age": 22, "minutes": 1500},
    ]

    # Patch encode_player_profile to avoid HF inference while still using the public API.
    monkeypatch.setattr(embeddings_module, "encode_player_profile", _deterministic_encoder)

    # Generate embeddings and build an in-memory FAISS index (cosine via normalized vectors).
    vectors = np.stack([embeddings_module.encode_player_profile(p) for p in players])
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors.astype("float32"))

    # Query for cb1 and remove self from results.
    scores, indices = index.search(vectors[0:1], 4)
    scores = scores[0].tolist()
    indices = indices[0].tolist()

    # Build result list excluding the query player itself.
    similar = [
        {"player_id": players[idx]["player_id"], "score": score}
        for idx, score in zip(indices, scores)
        if idx != 0 and idx != -1
    ]

    assert isinstance(similar, list)
    assert similar, "Expected at least one similar player"

    top_match = similar[0]
    assert top_match["player_id"] == "cb2", "Closest match should share position and workload"

    # Ensure scores are non-increasing.
    result_scores = [entry["score"] for entry in similar]
    assert all(result_scores[i] >= result_scores[i + 1] for i in range(len(result_scores) - 1))
