# src/scouting_ml/fbref_scraper.py
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import requests

from scouting_ml.paths import ensure_dirs

FBREF_BASE = "https://fbref.com"

# Austrian Bundesliga example:
# go to https://fbref.com/en/comps and click through to your league
# you'll get something like /en/comps/12/Austrian-Bundesliga-Stats
DEFAULT_LEAGUE_URL = "https://fbref.com/en/comps/12/Austrian-Bundesliga-Stats"


def _get(url: str) -> str:
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    return resp.text


def _extract_team_links(league_html: str) -> List[Dict[str, str]]:
    """
    From the league stats page, find all squad links.
    FBref puts them in a table with squad names linking to /en/squads/{teamid}/...
    """
    dfs = pd.read_html(league_html)
    teams: List[Dict[str, str]] = []
    # usually the first table is the league table with squad names
    # but we can also parse directly from the HTML with regex
    # easier: use pandas to find a column called 'Squad'
    for df in dfs:
        if "Squad" in df.columns:
            # we need the actual links -> we re-parse with read_html + match links
            break
    else:
        return teams  # nothing found

    # manual HTML scan for links
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(league_html, "lxml")
    for a in soup.select("table a[href*='/en/squads/']"):
        name = a.get_text(strip=True)
        href = a["href"]
        if not name:
            continue
        if not href.startswith("http"):
            href = FBREF_BASE + href
        teams.append({"name": name, "url": href})
    # deduplicate
    seen = set()
    uniq = []
    for t in teams:
        if t["url"] in seen:
            continue
        seen.add(t["url"])
        uniq.append(t)
    return uniq


def _fetch_team_standard(team_url: str) -> Optional[pd.DataFrame]:
    """
    On a team season page, FBref has multiple tables.
    We want the player 'Standard' table -> id="stats_standard"
    """
    html = _get(team_url)
    # fbref sometimes hides real tables in <!-- --> comments, so read_html with match
    tables = pd.read_html(html, match="Standard", flavor="lxml")
    if not tables:
        return None
    df = tables[0]
    # filter out "Squad Total" / "Opponent" rows
    if "Player" not in df.columns:
        return None
    df = df[df["Player"].notna()].copy()
    # keep only player rows
    # many fbref tables have a "Rk" column
    if "Rk" in df.columns:
        df = df[df["Rk"].apply(lambda x: str(x).isdigit())]
    # add team
    # try to get team name from page title
    m = re.search(r"<title>(.*?)Stats.*?</title>", html, re.I | re.S)
    team_name = m.group(1).strip() if m else None
    df["team"] = team_name
    return df


def scrape_league(league_url: str = DEFAULT_LEAGUE_URL) -> pd.DataFrame:
    print(f"[fbref] fetching league page: {league_url}")
    league_html = _get(league_url)
    teams = _extract_team_links(league_html)
    print(f"[fbref] found {len(teams)} teams")

    frames: List[pd.DataFrame] = []
    for t in teams:
        print(f"[fbref] team -> {t['name']} :: {t['url']}")
        df_team = _fetch_team_standard(t["url"])
        if df_team is None:
            print(f"[fbref]   !! no standard table")
            continue
        frames.append(df_team)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Scrape FBref league -> team -> players (standard stats)")
    ap.add_argument("--league-url", default=DEFAULT_LEAGUE_URL, help="FBref league stats page")
    ap.add_argument("--outfile", default="data/processed/fbref_players.csv")
    args = ap.parse_args()

    ensure_dirs()
    df = scrape_league(args.league_url)
    if df.empty:
        print("[fbref] no data scraped")
        return
    out_path = Path(args.outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[fbref] wrote {len(df)} rows -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
