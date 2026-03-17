from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from scouting_ml.providers.odds import OddsApiClient, build_market_context, normalize_odds_events


def _load_json_files(paths: list[str]) -> list[object]:
    return [json.loads(Path(raw).read_text(encoding="utf-8")) for raw in paths]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build season-level market context from The Odds API snapshots.")
    parser.add_argument("--input-json", nargs="*", default=[])
    parser.add_argument("--api-url", default="", help="Optional full API URL to fetch instead of local snapshots.")
    parser.add_argument("--league", required=True)
    parser.add_argument("--season", required=True)
    parser.add_argument("--club-links", default="data/external/club_provider_links.csv")
    parser.add_argument("--output", default="data/external/market_context.csv")
    parser.add_argument("--snapshot-date", default=datetime.now(timezone.utc).date().isoformat())
    args = parser.parse_args()

    payloads = _load_json_files(args.input_json)
    if args.api_url:
        payloads.append(OddsApiClient().get_json(args.api_url))
    frames = [normalize_odds_events(payload, season=args.season, league=args.league) for payload in payloads]
    odds_events = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True, sort=False) if frames else pd.DataFrame()
    out = build_market_context(
        odds_events,
        club_links_path=args.club_links,
        snapshot_date=args.snapshot_date,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"[market-context] wrote {len(out):,} rows -> {output_path}")


if __name__ == "__main__":
    main()
