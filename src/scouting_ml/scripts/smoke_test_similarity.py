"""Quick smoke test for player similarity search."""

from __future__ import annotations

import argparse
import os
import sys

from scouting_ml.services.player_similarity_service import find_similar_players


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test player similarity search.")
    parser.add_argument(
        "player_id",
        nargs="?",
        help="Player ID to query (falls back to PLAYER_ID env var if omitted).",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of neighbours to return.")
    args = parser.parse_args()

    player_id = args.player_id or os.getenv("PLAYER_ID")
    if not player_id:
        sys.exit("Please provide a player_id argument or set PLAYER_ID.")

    print("FAISS index path:", os.getenv("PLAYER_FAISS_INDEX_PATH"))
    print("Metadata path:", os.getenv("PLAYER_METADATA_PATH"))
    print("Query player_id:", player_id)

    results = find_similar_players(player_id, top_k=args.top_k)

    print("\nTop similar players:")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
