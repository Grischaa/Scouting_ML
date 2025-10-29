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

import argparse
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional
import re

import pandas as pd
import requests
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

    # Typical league table rows
    for row in soup.select("table.items tbody tr"):
        a = row.select_one("td.hauptlink a[title][href*='/startseite/verein/']")
        if not a:
            continue
        name = a.get("title", "").strip()
        href = a.get("href", "").strip()
        if not name or not href:
            continue
        # absolute url
        if not href.startswith("http"):
            href = f"https://www.transfermarkt.com{href}"
        # ensure we keep the same page type; inject season_id if requested
        if season_id and "saison_id" not in href:
            sep = "&" if "?" in href else "?"
            href = f"{href}{sep}saison_id={season_id}"
        # crude id from tail
        cid = href.rstrip("/").split("/")[-1]
        clubs.append(Club(name=name, id=cid, squad_url=href, slug=_slug(name)))
    return clubs


def _run(cmd: List[str]) -> int:
    """
    Run a command and stream output; return exit code.
    """
    return subprocess.call(cmd)


# -------------------------
# Per-club processing
# -------------------------

def process_club(club: Club, *, league_name: str, season_label: str, polite_sleep: float = 3.0) -> Dict[str, str]:
    """
    For one club:
      1) Run pipeline_tm to fetch+parse+enrich the team page
      2) Normalize that team CSV (club-scoped path)
      3) Build per-player season stats (club-scoped path)
    Returns paths for downstream aggregation.
    """

    ensure_dirs()

    # A) Fetch & parse team page, save club-scoped CSV
    #    pipeline_tm writes processed CSV as: data/processed/{stem}_players.csv
    #    where stem = Path(out_name).stem
    out_name = f"{club.slug}_team.html"
    pipeline_cmd = [
        "python", "-m", "scouting_ml.pipeline_tm",
        "--url", club.squad_url,
        "--out-name", out_name,
        "--club", club.name,
        "--league", league_name,
        "--season", season_label,
        "--format", "csv",  # csv is enough; parquet optional
    ]
    print(f"\n[pipeline] {club.name} -> {club.squad_url}")
    rc = _run(pipeline_cmd)
    if rc != 0:
        print(f"[pipeline] ERROR (rc={rc}) for {club.name}; skipping.")
        return {}

    # pipeline_tm saves here:
    #   data/processed/{club.slug}_team_players.csv
    team_processed_csv = Path("data/processed") / f"{Path(out_name).stem}_players.csv"

    if not team_processed_csv.exists():
        print(f"[pipeline] Expected output missing: {team_processed_csv} — skipping club.")
        return {}

    time.sleep(polite_sleep)

    # B) Normalize to a club-scoped Normalized CSV
    norm_csv = Path("data/processed") / f"{club.slug}_players_normalised.csv"
    normalize_cmd = [
        "python", "-m", "scouting_ml.normalize_tm",
        "--in", str(team_processed_csv),
        "--out", str(norm_csv),
    ]
    print(f"[normalize] {club.name} -> {norm_csv.name}")
    rc = _run(normalize_cmd)
    if rc != 0:
        print(f"[normalize] ERROR (rc={rc}) for {club.name}; continuing without normalized file.")
    time.sleep(polite_sleep)

    # C) Build per-season stats (uses Name/player_id from processed CSV)
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
        print(f"[stats] ERROR (rc={rc}) for {club.name}; continuing pipeline.")
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
    ap.add_argument("--league-url", required=True, help="League 'startseite' URL on Transfermarkt.")
    ap.add_argument("--league-name", default="Austrian Bundesliga", help="Label to stamp in outputs.")
    ap.add_argument("--season", default="2025/26", help="Season label to stamp in outputs (e.g., 2025/26).")
    ap.add_argument("--season-id", type=int, default=None, help="Optional saison_id to enforce in club URLs (e.g., 2025).")
    ap.add_argument("--sleep", type=float, default=3.0, help="Polite sleep (seconds) between steps per club.")
    ap.add_argument("--merge-stats", action="store_true", help="Concatenate all club stats into one league CSV at the end.")
    args = ap.parse_args()

    ensure_dirs()
    clubs = get_club_list(args.league_url, season_id=args.season_id)
    print(f"[league] Found {len(clubs)} clubs")

    artifacts: List[Dict[str, str]] = []
    for club in clubs:
        paths = process_club(
            club,
            league_name=args.league_name,
            season_label=args.season,
            polite_sleep=args.sleep,
        )
        if paths:
            artifacts.append(paths)

    # Optional: merge stats
    if args.merge_stats and artifacts:
        dfs = []
        for p in artifacts:
            sc = p.get("stats_csv")
            if sc and Path(sc).exists():
                df = pd.read_csv(sc)
                dfs.append(df)
        if dfs:
            merged = pd.concat(dfs, ignore_index=True)
            out = Path("data/interim/tm") / f"{_slug(args.league_name)}_{_slug(args.season)}_full_stats.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            merged.to_csv(out, index=False)
            print(f"[merge] Wrote league stats -> {out.resolve()}")
        else:
            print("[merge] No stats files found to merge.")

if __name__ == "__main__":
    main()
