# src/scouting_ml/fbref_scraper.py
from __future__ import annotations

import re
import time
import random
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

from scouting_ml.paths import ensure_dirs  # Assuming this is your module for dir creation

FBREF_BASE = "https://fbref.com"

# Example default; can override with args
DEFAULT_LEAGUE_URL = "https://fbref.com/en/comps/12/Austrian-Bundesliga-Stats"


def _get(url: str) -> str:
    for attempt in range(3):
        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            print(f"[fbref] Retry {attempt+1}/3 for {url}: {e}")
            time.sleep(random.uniform(5, 10))  # Increased sleep for robustness
    raise ValueError(f"[fbref] Failed to fetch {url} after 3 attempts")


def _extract_team_links(league_html: str) -> List[Dict[str, str]]:
    teams: List[Dict[str, str]] = []
    soup = BeautifulSoup(league_html, "lxml")
    for a in soup.select("table a[href*='/en/squads/']"):
        name = a.get_text(strip=True)
        href = a["href"]
        if not name:
            continue
        if not href.startswith("http"):
            href = FBREF_BASE + href
        teams.append({"name": name, "url": href})
    # Deduplicate
    seen = set()
    uniq = []
    for t in teams:
        if t["url"] in seen:
            continue
        seen.add(t["url"])
        uniq.append(t)
    return uniq


def _fetch_team_stats(team_url: str, team_name: str) -> Optional[pd.DataFrame]:
    html = _get(team_url)
    soup = BeautifulSoup(html, "lxml")

    # Categories as before
    categories = {
        "standard": "Standard Stats",
        "shooting": "Shooting",
        "passing": "Passing",
        "pass_types": "Pass Types",
        "gca": "Goal and Shot Creation",
        "defense": "Defensive Actions",
        "possession": "Possession",
        "playing_time": "Playing Time",
        "misc": "Miscellaneous Stats",
        "goalkeeping": "Goalkeeping",
        "adv_goalkeeping": "Advanced Goalkeeping",
    }

    dfs: Dict[str, pd.DataFrame] = {}
    for cat, match_str in categories.items():
        try:
            tables = pd.read_html(html, match=match_str, flavor="lxml")
            if not tables:
                continue
            df = tables[0]
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ["_".join([str(c).strip() for c in col if str(c).strip()]) for col in df.columns]
            if "Player" not in df.columns:
                continue
            df = df[df["Player"].notna() & ~df["Player"].isin(["Squad Total", "Opponent Total"])]
            if "Rk" in df.columns:
                df = df[df["Rk"].str.isdigit()]
            df["team"] = team_name
            key_cols = ["Player", "Nation", "Pos", "Age", "team"]
            prefix_cols = [col for col in df.columns if col not in key_cols]
            df = df.rename(columns={col: f"{cat}_{col}" for col in prefix_cols})
            dfs[cat] = df
        except Exception as e:
            print(f"[fbref] Failed {match_str} for {team_name}: {e}")

    if not dfs or "standard" not in dfs:
        return None

    out = dfs["standard"]
    for cat, df in dfs.items():
        if cat == "standard":
            continue
        out = pd.merge(out, df, on=["Player", "Nation", "Pos", "Age", "team"], how="left", suffixes=("", f"_{cat}_dup"))
    out = out.loc[:, ~out.columns.duplicated(keep="first")]
    return out


def _extract_season_links(history_html: str) -> List[Dict[str, str]]:
    seasons: List[Dict[str, str]] = []
    soup = BeautifulSoup(history_html, "lxml")
    for a in soup.select("table a[href*='/en/comps/']"):
        season_text = a.get_text(strip=True)
        href = a["href"]
        if re.match(r"\d{4}-\d{4}", season_text):  # e.g., "2023-2024"
            if not href.startswith("http"):
                href = FBREF_BASE + href
            seasons.append({"season": season_text, "url": href})
    # Reverse to get most recent first, deduplicate
    seasons = seasons[::-1]
    seen = set()
    uniq = []
    for s in seasons:
        if s["url"] in seen:
            continue
        seen.add(s["url"])
        uniq.append(s)
    return uniq


def scrape_league(league_url: str, position_filter: Optional[str] = None, scrape_history: bool = False, num_seasons: int = 5) -> pd.DataFrame:
    # Parse league name and comp_id from URL
    parts = league_url.split('/')
    comp_id = parts[parts.index('comps') + 1]
    league_slug = league_url.split('/')[-1].replace('-Stats', '')
    league_name = league_slug.replace('-', ' ')
    print(f"[fbref] Processing league: {league_name} ({league_url})")

    frames: List[pd.DataFrame] = []
    target_urls = [ {"season": "current", "url": league_url} ]

    if scrape_history:
        history_url = f"{FBREF_BASE}/en/comps/{comp_id}/history/{league_slug}-Seasons"
        print(f"[fbref] Fetching history: {history_url}")
        history_html = _get(history_url)
        season_links = _extract_season_links(history_html)
        print(f"[fbref] Found {len(season_links)} historical seasons")
        target_urls = season_links[-num_seasons:]  # Last N seasons (most recent)

    for s_idx, s in enumerate(target_urls):
        season = s["season"]
        url = s["url"]
        print(f"[fbref] Fetching season {season}: {url}")
        try:
            league_html = _get(url)
            teams = _extract_team_links(league_html)
            print(f"[fbref] Found {len(teams)} teams for {season}")
            for t_idx, t in enumerate(teams):
                print(f"[fbref] Team {t_idx+1}/{len(teams)} in {season} -> {t['name']} :: {t['url']}")
                df_team = _fetch_team_stats(t["url"], t["name"])
                if df_team is None:
                    continue
                df_team["season"] = season
                df_team["league"] = league_name
                if position_filter:
                    pos_list = [p.strip() for p in position_filter.split(',')]
                    df_team = df_team[df_team["Pos"].isin(pos_list)]
                frames.append(df_team)
                if t_idx < len(teams) - 1:
                    time.sleep(random.uniform(5, 10))
        except Exception as e:
            print(f"[fbref] Failed season {season}: {e}")
        if s_idx < len(target_urls) - 1:
            time.sleep(random.uniform(10, 15))  # Longer sleep between seasons

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Scrape FBref leagues/teams/players (all stats categories)")
    ap.add_argument("--league-urls", default=DEFAULT_LEAGUE_URL, help="Comma-separated FBref league stats URLs")
    ap.add_argument("--scrape-history", action="store_true", help="Scrape historical seasons for each league")
    ap.add_argument("--num-seasons", type=int, default=5, help="Number of historical seasons to scrape (most recent)")
    ap.add_argument("--position-filter", default=None, help="Filter by position (e.g., 'GK' or 'FW,MF')")
    ap.add_argument("--outfile", default="data/processed/fbref_players_all_stats.csv")
    args = ap.parse_args()

    ensure_dirs()
    league_urls = [url.strip() for url in args.league_urls.split(',')]
    all_frames: List[pd.DataFrame] = []
    for l_idx, league_url in enumerate(league_urls):
        df_league = scrape_league(
            league_url,
            position_filter=args.position_filter,
            scrape_history=args.scrape_history,
            num_seasons=args.num_seasons
        )
        if not df_league.empty:
            all_frames.append(df_league)
        if l_idx < len(league_urls) - 1:
            time.sleep(random.uniform(15, 20))  # Sleep between leagues

    if not all_frames:
        print("[fbref] No data scraped")
        return
    out_df = pd.concat(all_frames, ignore_index=True)
    out_path = Path(args.outfile)
    if args.position_filter:
        pos_suffix = "_" + args.position_filter.replace(',', '_')
        out_path = out_path.with_stem(out_path.stem + pos_suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[fbref] Wrote {len(out_df)} rows -> {out_path.resolve()}")


if __name__ == "__main__":
    main()