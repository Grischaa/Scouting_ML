from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from scouting_ml.providers.statsbomb import aggregate_player_season_features


def _parse_csv_ints(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    values: list[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    return values or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate StatsBomb open-data events into player-season external features.")
    parser.add_argument("--open-data-root", required=True, help="Local root of the StatsBomb open-data repository.")
    parser.add_argument("--output", default="data/external/statsbomb_player_season_features.csv")
    parser.add_argument("--player-links", default="data/external/player_provider_links.csv")
    parser.add_argument("--competition-ids", default="")
    parser.add_argument("--season-ids", default="")
    parser.add_argument("--snapshot-date", default=datetime.now(timezone.utc).date().isoformat())
    args = parser.parse_args()

    out = aggregate_player_season_features(
        args.open_data_root,
        competition_ids=_parse_csv_ints(args.competition_ids),
        season_ids=_parse_csv_ints(args.season_ids),
        player_links_path=args.player_links,
        snapshot_date=args.snapshot_date,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"[statsbomb] wrote {len(out):,} rows -> {output_path}")


if __name__ == "__main__":
    main()
