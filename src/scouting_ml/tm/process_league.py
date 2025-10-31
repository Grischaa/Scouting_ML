"""
Process every club in a league in a club-scoped, idempotent way.

Usage:
  python -m scouting_ml.process_league \
    --league-url "https://www.transfermarkt.com/bundesliga/startseite/wettbewerb/A1/saison_id/2025" \
    --league-name "Austrian Bundesliga" \
    --season "2025/26" \
    --sleep 4
"""

from __future__ import annotations
from scouting_ml.utils.import_guard import *  # noqa: F403
import argparse
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional
import re

import pandas as pd
from bs4 import BeautifulSoup

from scouting_ml.paths import ensure_dirs
from scouting_ml.tm_scraper import fetch  # robust fetch with retries/backoff


# -------------------------
# Helpers
# -------------------------

def _slug(s: str) -> str:
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^a-zA-Z0-9_.-]", "-", s)
    return s.lower()


@dataclass
class Club:
    name: str
    id: str
    squad_url: str
    slug: str


def get_club_list(league_url: str, season_id: Optional[int] = None) -> List[Club]:
    """
    Parse a Transfermarkt league 'startseite' page and return club names + squad URLs.
    """
    html = fetch(league_url)
    soup = BeautifulSoup(html, "lxml")
    clubs: List[Club] = []

    for row in soup.select("table.items tbody tr"):
        a = row.select_one("td.hauptlink a[title][href*='/startseite/verein/']")
        if not a:
            continue
        name = a.get("title", "").strip()
        href = a.get("href", "").strip()
        if not name or not href:
            continue
        if not href.startswith("http"):
            href = f"https://www.transfermarkt.com{href}"
        if season_id and "saison_id" not in href:
            sep = "&" if "?" in href else "?"
            href = f"{href}{sep}saison_id={season_id}"
        cid = href.rstrip("/").split("/")[-1]
        clubs.append(Club(name=name, id=cid, squad_url=href, slug=_slug(name)))
    return clubs


def _run(cmd: List[str]) -> int:
    return subprocess.call(cmd)


# -------------------------
# Per-club processing
# -------------------------

def process_club(
    club: Club,
    *,
    league_name: str,
    season_label: str,
    polite_sleep: float = 3.0,
    skip_existing: bool = False,
    force: bool = False,
    ) -> Dict[str, str]:

    ensure_dirs()

    out_name = f"{club.slug}_team.html"
    team_processed_csv = Path("data/processed") / f"{club.slug}_team_players.csv"

    # 🟢 short-circuit
    if skip_existing and team_processed_csv.exists() and not force:
        print(f"[skip] {club.name} already processed -> {team_processed_csv.name}")
        return {
            "processed_csv": str(team_processed_csv),
            "normalized_csv": str(Path("data/processed") / f"{club.slug}_players_normalised.csv"),
            "stats_csv": str(Path("data/interim/tm") / f"{club.slug}_team_stats.csv"),
        }
    # A) run pipeline
    out_name = f"{club.slug}_team.html"
    pipeline_cmd = [
        "python", "-m", "scouting_ml.pipeline_tm",
        "--url", club.squad_url,
        "--out-name", out_name,
        "--club", club.name,
        "--league", league_name,
        "--season", season_label,
        "--format", "csv",
    ]
    print(f"\n[pipeline] {club.name} -> {club.squad_url}")
    rc = _run(pipeline_cmd)
    if rc != 0:
        print(f"[pipeline] ERROR (rc={rc}) for {club.name}; skipping.")
        return {}

    team_processed_csv = Path("data/processed") / f"{Path(out_name).stem}_players.csv"
    if not team_processed_csv.exists():
        print(f"[pipeline] Expected output missing: {team_processed_csv} — skipping club.")
        return {}

    time.sleep(polite_sleep)

    # B) normalize
    norm_csv = Path("data/processed") / f"{club.slug}_players_normalised.csv"
    normalize_cmd = [
        "python", "-m", "scouting_ml.normalize_tm",
        "--in", str(team_processed_csv),
        "--out", str(norm_csv),
    ]
    print(f"[normalize] {club.name} -> {norm_csv.name}")
    rc = _run(normalize_cmd)
    if rc != 0:
        print(f"[normalize] ERROR (rc={rc}) for {club.name}; continuing.")
    time.sleep(polite_sleep)

    # C) stats
    stats_csv = Path("data/interim/tm") / f"{club.slug}_team_stats.csv"
    stats_csv.parent.mkdir(parents=True, exist_ok=True)
    stats_cmd = [
        "python", "-m", "scouting_ml.build_tm_stats",
        "--in_players", str(team_processed_csv),
        "--out", str(stats_csv),
        "--concurrency", "4",
        "--verbose",
    ]
    print(f"[stats] {club.name} -> {stats_csv.name}")
    rc = _run(stats_cmd)
    if rc != 0:
        print(f"[stats] ERROR (rc={rc}) for {club.name}; continuing.")
    time.sleep(polite_sleep)

    return {
        "processed_csv": str(team_processed_csv),
        "normalized_csv": str(norm_csv),
        "stats_csv": str(stats_csv),
    }


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Process all clubs from a Transfermarkt league page (club-scoped).")
    ap.add_argument("--league-url", required=True)
    ap.add_argument("--league-name", default="Austrian Bundesliga")
    ap.add_argument("--season", default="2025/26")
    ap.add_argument("--season-id", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=3.0)
    ap.add_argument("--merge-stats", action="store_true")
    ap.add_argument("--skip-existing",action="store_true",help="Skip clubs that already have a processed CSV.")
    ap.add_argument("--force-club",action="append",default=[],help="Club slug(s) to force reprocess even if they exist (can repeat).")
    args = ap.parse_args()

    ensure_dirs()
    clubs = get_club_list(args.league_url, season_id=args.season_id)
    print(f"[league] Found {len(clubs)} clubs")

    for club in clubs:
        paths = process_club(
            club,
            league_name=args.league_name,
            season_label=args.season,
            polite_sleep=args.sleep,
            skip_existing=args.skip_existing,
            force=(club.slug in args.force_club),
        )


    # -------------------------
# Optional: merge stats
# -------------------------

    if args.merge_stats:
        stats_dir = Path("data/interim/tm")
        all_stats = list(stats_dir.glob("*_team_stats.csv"))
        if not all_stats:
            print("[merge] No stats files found to merge in data/interim/tm/")
        else:
            dfs = [pd.read_csv(p) for p in all_stats]
            merged = pd.concat(dfs, ignore_index=True)
            out = stats_dir / f"{_slug(args.league_name)}_{_slug(args.season)}_full_stats.csv"
            merged.to_csv(out, index=False)
            print(f"[merge] Wrote league stats -> {out.resolve()}")

# -------------------------
# ✅ NEW: Schema validation
# -------------------------

    try:
        from scouting_ml.validate_schema import validate_schema
        import pandas as pd

        final_path = Path("data/processed") / f"{_slug(args.league_name)}_{_slug(args.season)}_clean.csv"
        if final_path.exists():
            print(f"\n[validate] Checking schema for {final_path.name} ...")
            df = pd.read_csv(final_path)
            validate_schema(df)
        else:
            # Fallback: validate merged feature file if clean file not yet created
            merged_features = Path("data/processed") / f"{_slug(args.league_name)}_{_slug(args.season)}_features.csv"
            if merged_features.exists():
                print(f"\n[validate] Checking schema for {merged_features.name} ...")
                df = pd.read_csv(merged_features)
                validate_schema(df)
            else:
                print("[validate ⚠️] No merged or clean file found to validate.")
    except Exception as e:
        print(f"[validate ❌] Schema validation failed to run: {e}")



if __name__ == "__main__":
    main()
