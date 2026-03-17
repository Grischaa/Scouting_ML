from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from scouting_ml.providers.football_api import ApiFootballClient, SportmonksClient, build_fixture_context, normalize_fixtures


def _load_json_files(paths: list[str]) -> list[object]:
    payloads: list[object] = []
    for raw in paths:
        path = Path(raw)
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    return payloads


def _fetch_live_payload(provider: str, api_url: str) -> object:
    if provider == "sportmonks":
        return SportmonksClient().get_json(api_url)
    return ApiFootballClient().get_json(api_url)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build season-level fixture context from Sportmonks or API-Football snapshots.")
    parser.add_argument("--provider", choices=["sportmonks", "api-football"], required=True)
    parser.add_argument("--input-json", nargs="*", default=[])
    parser.add_argument("--api-url", default="", help="Optional full API URL to fetch instead of local snapshots.")
    parser.add_argument("--club-links", default="data/external/club_provider_links.csv")
    parser.add_argument("--output", default="data/external/fixture_context.csv")
    parser.add_argument("--snapshot-date", default=datetime.now(timezone.utc).date().isoformat())
    args = parser.parse_args()

    payloads = _load_json_files(args.input_json)
    if args.api_url:
        payloads.append(_fetch_live_payload(args.provider, args.api_url))
    frames = [normalize_fixtures(payload, provider=args.provider) for payload in payloads]
    fixtures = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True, sort=False) if frames else pd.DataFrame()
    out = build_fixture_context(
        fixtures,
        provider=args.provider,
        club_links_path=args.club_links,
        snapshot_date=args.snapshot_date,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"[fixture-context] wrote {len(out):,} rows -> {output_path}")


if __name__ == "__main__":
    main()
