# src/scouting_ml/fbref_scraper.py
from __future__ import annotations

import re
import time
import random
from pathlib import Path
from typing import List, Dict, Optional
from scouting_ml.utils.import_guard import *  # noqa: F403
import pandas as pd
import requests
from bs4 import BeautifulSoup

from scouting_ml.paths import ensure_dirs

FBREF_BASE = "https://fbref.com"

DEFAULT_LEAGUE_URL = "https://fbref.com/en/comps/12/Austrian-Bundesliga-Stats"


def _get(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
        "Referer": "https://fbref.com/",
        "Connection": "keep-alive",
        "DNT": "1",
        "Upgrade-Insecure-Requests": "1",
    }

    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 403:
                Path("data/raw/fbref_403.html").write_text(resp.text, encoding="utf-8")
                print(f"[fbref] ❌ 403 Forbidden. Saved HTML to data/raw/fbref_403.html for debugging.")
                raise requests.HTTPError("403 Forbidden")
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            print(f"[fbref] Retry {attempt+1}/3 for {url}: {e}")
            time.sleep(random.uniform(5, 10))
    raise ValueError(f"[fbref] Failed to fetch {url} after retries")



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
    seen = set()
    uniq = []
    for t in teams:
        if t["url"] not in seen:
            seen.add(t["url"])
            uniq.append(t)
    return uniq


def _fetch_team_stats(team_url: str, team_name: str) -> Optional[pd.DataFrame]:
    html = _get(team_url)
    soup = BeautifulSoup(html, "lxml")
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
            # Add player_id from links
            player_links = [a['href'] for a in soup.select("table#stats_standard a[href*='/en/players/']") if a.get_text(strip=True) in df["Player"].values]
            if len(player_links) == len(df):
                df["player_url"] = player_links
            key_cols = ["Player", "Nation", "Pos", "Age", "team", "player_url"]
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
        out = pd.merge(out, df, on=["Player", "Nation", "Pos", "Age", "team"], how="left")
    out = out.loc[:, ~out.columns.duplicated()]
    return out


def _extract_season_links(history_html: str) -> List[Dict[str, str]]:
    seasons: List[Dict[str, str]] = []
    soup = BeautifulSoup(history_html, "lxml")
    for a in soup.select("table a[href*='/en/comps/']"):
        season_text = a.get_text(strip=True)
        href = a["href"]
        if re.match(r"\d{4}-\d{4}", season_text):
            if not href.startswith("http"):
                href = FBREF_BASE + href
            seasons.append({"season": season_text, "url": href})
    seasons = seasons[::-1]  # Recent first
    seen = set()
    uniq = []
    for s in seasons:
        if s["url"] not in seen:
            seen.add(s["url"])
            uniq.append(s)
    return uniq


def scrape_league(league_url: str, position_filter: Optional[str] = None, min_minutes: int = 0, scrape_history: bool = False, num_seasons: int = 5) -> pd.DataFrame:
    parts = league_url.split('/')
    comp_id = parts[parts.index('comps') + 1]
    league_slug = league_url.split('/')[-1].replace('-Stats', '')
    league_name = league_slug.replace('-', ' ')
    print(f"[fbref] Processing league: {league_name}")
    frames = []
    target_urls = [{"season": "current", "url": league_url}]
    if scrape_history:
        history_url = f"{FBREF_BASE}/en/comps/{comp_id}/history/{league_slug}-Seasons"
        history_html = _get(history_url)
        season_links = _extract_season_links(history_html)
        target_urls = season_links[-num_seasons:]
    for s in target_urls:
        league_html = _get(s["url"])
        teams = _extract_team_links(league_html)
        for t in teams:
            df_team = _fetch_team_stats(t["url"], t["name"])
            if df_team is None:
                continue
            df_team["season"] = s["season"]
            df_team["league"] = league_name
            if "playing_time_Min" in df_team.columns:
                df_team = df_team[df_team["playing_time_Min"] >= min_minutes]
            if position_filter:
                pos_list = [p.strip() for p in position_filter.split(',')]
                df_team = df_team[df_team["Pos"].isin(pos_list)]
            frames.append(df_team)
            time.sleep(random.uniform(5, 10))
        time.sleep(random.uniform(10, 15))  # Between seasons
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Robust FBref scraper for player stats")
    ap.add_argument("--league-urls", default=DEFAULT_LEAGUE_URL, help="Comma-separated league URLs")
    ap.add_argument("--scrape-history", action="store_true")
    ap.add_argument("--num-seasons", type=int, default=5)
    ap.add_argument("--position-filter", default=None)
    ap.add_argument("--min-minutes", type=int, default=0, help="Filter players with at least this many minutes")
    ap.add_argument("--outfile", default="data/processed/fbref_players_all_stats.csv")
    args = ap.parse_args()
    ensure_dirs()
    league_urls = [url.strip() for url in args.league_urls.split(',')]
    all_frames = []
    for league_url in league_urls:
        df = scrape_league(league_url, args.position_filter, args.min_minutes, args.scrape_history, args.num_seasons)
        if not df.empty:
            all_frames.append(df)
        time.sleep(random.uniform(15, 20))  # Between leagues
    if not all_frames:
        print("[fbref] No data")
        return
    out_df = pd.concat(all_frames, ignore_index=True)
    out_path = Path(args.outfile)
    if args.position_filter:
        out_path = out_path.with_stem(out_path.stem + "_" + args.position_filter.replace(',', '_'))
    out_df.to_csv(out_path, index=False)
    print(f"[fbref] Wrote {len(out_df)} rows to {out_path}")


if __name__ == "__main__":
    main()