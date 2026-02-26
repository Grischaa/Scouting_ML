"""Build and persist a FAISS index for player similarity search.

This script reuses existing embedding utilities and is intended to be run
manually or in CI (not within the API layer).
"""
from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from scouting_ml.nlp.embeddings import (
    build_faiss_index,
    encode_player_profile,
    save_faiss_index,
)

logger = logging.getLogger(__name__)

IDENTIFIER_KEYS = ("player_id", "id", "uuid", "name")


def load_player_records() -> List[Dict[str, Any]]:
    """Load player data records.

    By default, attempts to read PLAYER_DATA_PATH (json, jsonl, csv, parquet).
    Replace or extend this loader with your canonical data source.
    """
    data_path = os.getenv("PLAYER_DATA_PATH")
    if not data_path:
        logger.info("PLAYER_DATA_PATH not set; using stubbed player records.")
        return [
            {"player_id": "stub_cb", "position": "CB", "age": 25, "minutes": 2100},
            {"player_id": "stub_cm", "position": "CM", "age": 24, "minutes": 2400},
            {"player_id": "stub_fw", "position": "FW", "age": 22, "minutes": 1900},
        ]

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"PLAYER_DATA_PATH points to missing file: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError("JSON player data must be a list of objects.")
        return [dict(item) for item in data]

    if suffix == ".jsonl":
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    records.append(json.loads(line))
        return records

    if suffix == ".csv":
        with path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            return [dict(row) for row in reader]

    if suffix in {".parquet", ".pq"}:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - defensive
            raise ImportError("pandas is required to read parquet player data.") from exc
        df = pd.read_parquet(path)
        return df.to_dict(orient="records")

    raise ValueError(f"Unsupported player data format: {suffix}")


def _resolve_player_id(record: Dict[str, Any], idx: int) -> str:
    """Extract a stable player identifier from the record."""
    for key in IDENTIFIER_KEYS:
        value = record.get(key)
        if value:
            return str(value)
    raise ValueError(f"Player record at index {idx} lacks an identifier ({IDENTIFIER_KEYS}).")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    players = load_player_records()
    if not players:
        raise RuntimeError("No player records found; cannot build embedding index.")

    logger.info("Loaded %d player records", len(players))
    embeddings: List[np.ndarray] = []
    id_to_idx: Dict[str, int] = {}

    logger.info("Encoding player profiles...")
    for idx, player in enumerate(players):
        player_id = _resolve_player_id(player, idx)
        embedding = encode_player_profile(player)
        embeddings.append(np.asarray(embedding, dtype="float32"))
        id_to_idx[player_id] = idx

    embedding_matrix = np.stack(embeddings, axis=0)
    logger.info("Building FAISS index with cosine similarity (vectors: %s)", embedding_matrix.shape)
    index = build_faiss_index(embedding_matrix, metric="cosine")

    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "data" / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / "player_index.faiss"
    embeddings_path = output_dir / "player_embeddings.npy"
    mapping_path = output_dir / "player_index_mapping.json"

    logger.info("Saving FAISS index to %s", index_path)
    save_faiss_index(index, index_path)

    logger.info("Saving embedding matrix to %s", embeddings_path)
    np.save(embeddings_path, embedding_matrix.astype("float32", copy=False))

    logger.info("Saving player-id mapping to %s", mapping_path)
    mapping_path.write_text(json.dumps(id_to_idx, indent=2), encoding="utf-8")

    logger.info("Done. Indexed %d players.", len(players))


if __name__ == "__main__":
    main()
