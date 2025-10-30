"""
Transfermarkt Stats Parser — extend player-level scraping to capture per-season and per-competition statistics.

Usage (CLI):
    python -m scouting_ml.build_tm_stats \
        --in_players data/interim/tm/sturm_graz_team_parsed.csv \
        --out data/interim/tm/sturm_graz_team_stats.csv \
        --concurrency 4 --verbose
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import dataclasses as dc
import re
import time
import os
from typing import Dict, List, Optional
from io import StringIO

import pandas as pd
import requests
from bs4 import BeautifulSoup
from scouting_ml.logging import get_logger

logger = get_logger()

# -----------------------------
# Config
# -----------------------------
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Connection": "keep-alive",
}
REQ_TIMEOUT = 20
RETRY = 3
SLEEP_BETWEEN = 1.0  # polite crawling
VERBOSE = False

# Regex helpers
RE_MINUTES = re.compile(r"([\d\.']*)'$")  # e.g., "1.234'"
RE_INT = re.compile(r"\d+")
RE_FLOAT = re.compile(r"\d+[\.,]?\d*")
RE_SEASON = re.compile(r"(\d{4}|\d{2})/(\d{2})")

# Columns mapping (DE / EN -> canonical)
COLMAP: Dict[str, str] = {
    # Appearances & minutes
    "apps": "apps",
    "appearances": "apps",
    "einsätze": "apps",
    "einsatz(e)": "apps",
    "minutes": "minutes",
    "min": "minutes",
    "spielminuten": "minutes",
    "starting eleven": "starts",
    "startelf": "starts",
    "substituted in": "sub_on",
    "eingewechselt": "sub_on",
    "substituted off": "sub_off",
    "ausgewechselt": "sub_off",
    # Scoring
    "goals": "goals",
    "tore": "goals",
    "assists": "assists",
    "vorlagen": "assists",
    # Cards
    "yellow cards": "yc",
    "gelbe karten": "yc",
    "second yellow cards": "y2c",
    "gelb-rote karten": "y2c",
    "red cards": "rc",
    "rote karten": "rc",
    # Others frequently present
    "clean sheets": "clean_sheets",
    "zu-null-spiele": "clean_sheets",
    "goals conceded": "goals_conceded",
    "gegentore": "goals_conceded",
    "points per match": "ppm",
    "punkte pro spiel": "ppm",
    # Context
    "season": "season",
    "wettbewerb": "competition",
    "competition": "competition",
}

@dc.dataclass
class PlayerTask:
    player_id: str
    player_name: str
    profile_url: Optional[str]


def _get(url: str) -> Optional[requests.Response]:
    for i in range(RETRY):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQ_TIMEOUT)
            if resp.status_code == 200:
                return resp
        except requests.RequestException as e:
            if VERBOSE:
                logger.warning(f"Retry {i+1}/{RETRY} for {url}: {e}")
        time.sleep(SLEEP_BETWEEN * (i + 1))
    return None

def candidate_stats_urls(player_id: str, profile_url: Optional[str]) -> List[str]:
    pid = RE_INT.search(player_id).group(0) if RE_INT.search(player_id) else player_id
    bases = ["https://www.transfermarkt.com", "https://www.transfermarkt.de"]
    urls = []

    # If profile_url provided, derive exact stats URLs from it
    if profile_url:
        # Replace /profil/ with /leistungsdatendetails/ or /detaillierte-leistungsdaten/
        stats_url = profile_url.replace("/profil/", "/leistungsdatendetails/")
        detailed_url = profile_url.replace("/profil/", "/detaillierte-leistungsdaten/")
        urls.extend([detailed_url, stats_url])  # Prioritize detailed first

    # Fallback constructions if no profile_url or derivations fail
    for base in bases:
        urls.extend([
            f"{base}/detaillierte-leistungsdaten/spieler/{pid}",  # Detailed stats
            f"{base}/leistungsdatendetails/spieler/{pid}",  # Compact stats
            f"{base}/leistungsdaten/spieler/{pid}/saison/ges/",  # General fallback
        ])
    return list(dict.fromkeys(urls))  # Unique, ordered

def discover_stats_url(profile_url: str) -> Optional[str]:
    resp = _get(profile_url)
    if not resp:
        return None
    soup = BeautifulSoup(resp.text, "lxml")
    for a in soup.select("a[href]"):
        href = a.get("href", "")
        if "leistungsdaten" in href or "detaillierte-leistungsdaten" in href or "stats" in href or "performance" in href:
            if not href.startswith("http"):
                m = re.match(r"(https?://[^/]+)", profile_url)
                if m:
                    href = m.group(1) + href
            return href
    return None

def normalize_season(s: str) -> str:
    m = RE_SEASON.match(s)
    if not m:
        return s
    y1, y2 = m.groups()
    if len(y1) == 2:
        y1 = "20" + y1 if int(y1) < 50 else "19" + y1
    if len(y2) == 2:
        y2 = "20" + y2 if int(y2) < 50 else "19" + y2
    return f"{y1}/{y2}"

def normalize_stats_df(df: pd.DataFrame) -> pd.DataFrame:
    # Rename columns
    rename = {}
    for c in df.columns:
        cl = str(c).lower().strip()
        for k, v in COLMAP.items():
            if k in cl:
                rename[c] = v
                break
    df = df.rename(columns=rename)

    # Normalize season
    if "season" in df:
        df["season"] = df["season"].astype(str).apply(normalize_season)

    # Minutes: extract digits, remove '
    if "minutes" in df:
        df["minutes"] = df["minutes"].astype(str).apply(lambda x: ''.join(c for c in x if c.isdigit() or c == '.') if "'" in x else x)

    # Numeric columns
    num_cols = ["apps", "minutes", "starts", "sub_on", "sub_off", "goals", "assists", "yc", "y2c", "rc", "clean_sheets", "goals_conceded", "ppm"]
    for col in num_cols:
        if col in df:
            # Replace non-numeric indicators with NaN
            df[col] = df[col].replace("-", pd.NA).replace("", pd.NA)
            # Handle thousands/decimal separators (assuming . thousands, , decimal in some locales)
            s = df[col].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(s, errors='coerce').fillna(0)

    # Drop totals/empty rows
    if "season" in df:
        df = df[~df["season"].str.contains("total", na=False, case=False)]
        df = df[df["season"].notna()]

    return df

def parse_stats_table(html: str) -> Optional[pd.DataFrame]:
    try:
        tables = pd.read_html(StringIO(html), flavor="lxml")
        best: Optional[pd.DataFrame] = None
        best_n = -1
        for t in tables:
            cols_lower = [str(c).lower() for c in t.columns]
            if any("season" in c or "wettbewerb" in c or "competition" in c for c in cols_lower):
                n = t.shape[0] * t.shape[1]
                if n > best_n:
                    best, best_n = t, n
        if best is None:
            return None
        return normalize_stats_df(best)
    except ValueError as e:
        if VERBOSE:
            logger.error(f"Parse error: {e}")
        return None

def fetch_player_stats(task: PlayerTask) -> Optional[pd.DataFrame]:
    candidates = candidate_stats_urls(task.player_id, task.profile_url)
    if task.profile_url:  # If profile given, try discovering additional if needed
        tab = discover_stats_url(task.profile_url)
        if tab and tab not in candidates:
            candidates.insert(0, tab)
    seen = set()
    for url in candidates:
        if not url or url in seen:
            continue
        seen.add(url)
        if VERBOSE:
            logger.info(f"Trying direct fetch for {task.player_name} ({task.player_id}) -> {url}")
        resp = _get(url)
        if not resp:
            continue
        if VERBOSE:
            logger.info(f"Status {resp.status_code} for {url}")
        try:
            os.makedirs("data/raw/tm/stats", exist_ok=True)
            with open(f"data/raw/tm/stats/stats_{task.player_id}.html", "w", encoding="utf-8") as f:
                f.write(resp.text)
        except Exception as ex:
            if VERBOSE:
                logger.warning(f"Failed to save HTML: {ex}")
        df = parse_stats_table(resp.text)
        if df is None or df.empty:
            continue
        df.insert(0, "player_id", task.player_id)
        df.insert(1, "player_name", task.player_name)
        df.insert(2, "stats_url", url)
        return df
    if VERBOSE:
        logger.warning(f"No valid stats for {task.player_name} ({task.player_id})")
    return None

def load_player_tasks(path: str) -> List[PlayerTask]:
    df = pd.read_csv(path)
    if len(df.columns) < 2:
        raise ValueError(f"CSV file has only {len(df.columns)} columns; it needs at least player_id and name. Check if commas are missing or file is malformed.")
    low = {c.lower(): c for c in df.columns}
    pid_col = low.get("playerid") or low.get("player_id") or low.get("tm_id") or low.get("transfermarkt_id") or low.get("id") or df.columns[0]
    name_col = low.get("name") or low.get("player_name") or (df.columns[1] if len(df.columns) > 1 else None)
    if name_col is None:
        raise ValueError("No name column found; check CSV headers or add 'Name' or 'player_name' column.")
    url_col = None
    for key in ["profile_url", "tm_url", "player_url", "url", "profil", "profil_url"]:
        if key in low:
            url_col = low[key]
            break
    def to_pid(val: object) -> str:
        s = "" if pd.isna(val) else str(val)
        m = RE_INT.search(s)
        return m.group(0) if m else s
    tasks = []
    for _, r in df.iterrows():
        tasks.append(PlayerTask(player_id=to_pid(r[pid_col]), player_name=str(r[name_col]), profile_url=(str(r[url_col]) if url_col and pd.notna(r[url_col]) else None)))
    return tasks

def run(in_players: str, out_path: str, concurrency: int = 4, debug_first: Optional[int] = None) -> None:
    tasks = load_player_tasks(in_players)
    if debug_first:
        tasks = tasks[:debug_first]
    rows = []
    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(fetch_player_stats, t) for t in tasks]
        for f in cf.as_completed(futures):
            try:
                df = f.result()
                if df is not None and not df.empty:
                    rows.append(df)
            except Exception as e:
                logger.warning(f"Exception in task: {e}")
            time.sleep(SLEEP_BETWEEN / concurrency)  # Distribute sleep
    if rows:
        out = pd.concat(rows, ignore_index=True)
        out.to_csv(out_path, index=False)
        logger.success(f"Wrote {len(out)} rows -> {out_path}")
    else:
        logger.warning("No stats rows parsed; output skipped")

def main():
    ap = argparse.ArgumentParser(description="Build Transfermarkt stats table from player parsed CSV")
    ap.add_argument("--in_players", required=True, help="CSV from build_tm_interim_from_raw stage")
    ap.add_argument("--out", required=True, help="Output CSV to write team/player stats to (e.g. data/interim/tm/sturm_graz_team_stats.csv)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--debug_first", type=int, default=None)
    ap.add_argument("--verbose", action="store_true", help="Print debug info and save HTML")
    args = ap.parse_args()

    global VERBOSE
    VERBOSE = bool(args.verbose)

    # 👇 THIS was the bug: args.out_path -> args.out
    run(args.in_players, args.out, args.concurrency, args.debug_first)


if __name__ == "__main__":
    main()