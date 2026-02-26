# src/scouting_ml/fbref/fbref_api_scraper.py
from __future__ import annotations

import argparse
import io
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import re

import httpx
import pandas as pd
from bs4 import BeautifulSoup, Comment

# A simple fallback for the ensure_dirs function if the original import fails.
def ensure_dirs():
    """Ensures necessary data directories exist."""
    Path("data/processed").mkdir(parents=True, exist_ok=True)


FBREF_BASE = "https://fbref.com"
# This page aggregates Big 5 links; for direct leagues pass a league page like
# https://fbref.com/en/comps/9/Premier-League-Stats or its season page.
DEFAULT_LEAGUE_URL = "https://fbref.com/en/comps/Big5/Big-5-European-Leagues-Stats"
DEFAULT_OUT = "data/processed/fbref_big5_league.csv"


# -------------------------------------------------
# helpers
# -------------------------------------------------
def _uncomment_fbref(html: str) -> str:
    """
    FBref hides tables in HTML comments. This function pulls them out.
    """
    # The "lxml" parser is essential here for performance and correctness
    soup = BeautifulSoup(html, "lxml")
    for element in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment_text = str(element)
        if "<table" in comment_text:
            # Replace the comment with the HTML it contains
            element.replace_with(BeautifulSoup(comment_text, "lxml"))
    return str(soup)


def _flatten_columns(cols) -> List[str]:
    """Flatten FBref MultiIndex columns preferring the leaf label.

    Removes 'Unnamed' placeholders and non-alphanumerics; returns simple names.
    """
    if not isinstance(cols, pd.MultiIndex):
        return [str(c) for c in cols]
    out: List[str] = []
    for col in cols.values:
        parts = [str(p).strip() for p in col if str(p).strip() and not str(p).startswith("Unnamed")]
        name = parts[-1] if parts else str(col[-1])
        out.append(name)
    # Clean
    return [re.sub(r"[^A-Za-z0-9_]+", "", c) for c in out]


def _build_http_client() -> httpx.Client:
    """Create a persistent HTTP client with sensible defaults."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/128.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }
    # HTTP/2 helps with performance; timeouts keep things sane
    return httpx.Client(
        headers=headers,
        http2=True,
        timeout=httpx.Timeout(30.0, connect=15.0, read=30.0),
        follow_redirects=True,
    )


def _polite_delay(min_s: float = 1.0, max_s: float = 2.5) -> None:
    time.sleep(random.uniform(min_s, max_s))


def _http_get_text(client: httpx.Client, url: str, *, retries: int = 3) -> Optional[str]:
    """Fetch URL and return response text with basic retry/backoff."""
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.get(url)
            # Handle rate limiting politely
            if resp.status_code == 429:
                wait = random.uniform(20, 35)
                print(f"[fbref-scraper] 429 for {url}. Backing off {wait:.1f}s...")
                time.sleep(wait)
                continue
            if 500 <= resp.status_code < 600:
                raise httpx.HTTPStatusError(
                    f"server error: {resp.status_code}", request=resp.request, response=resp
                )
            return resp.text
        except Exception as e:
            last_error = e
            print(f"[fbref-scraper] GET failed (attempt {attempt}/{retries}) for {url}: {e}")
            time.sleep(random.uniform(5, 10))
    print(f"[fbref-scraper] Failed to fetch {url}: {last_error}")
    return None


def _extract_team_links(league_html: str, max_teams: Optional[int] = None) -> List[Dict[str, str]]:
    """
    Returns a list of dictionaries {'name': ..., 'url': ...} for squad pages.
    This function uses an updated and more reliable CSS selector.
    """
    league_html = _uncomment_fbref(league_html)
    soup = BeautifulSoup(league_html, "lxml")

    teams: List[Dict[str, str]] = []

    # Primary: league squad table
    for link in soup.select('td[data-stat="squad"] a[href*="/squads/"]'):
        href = link.get("href")
        if href:
            full_url = FBREF_BASE + href if not href.startswith("http") else href
            teams.append({"name": link.get_text(strip=True), "url": full_url})

    # Fallback: scan for any squad links present on the page
    if not teams:
        # Prefer season-specific squad links when available
        season_links: Dict[str, Dict[str, str]] = {}
        generic_links: Dict[str, Dict[str, str]] = {}
        for a in soup.select('a[href^="/en/squads/"]'):
            href = a.get("href") or ""
            m = re.match(r"^/en/squads/([a-f0-9]{8})(/.*)?$", href)
            if not m or "-Stats" not in href:
                continue
            team_id = m.group(1)
            name = a.get_text(strip=True) or Path(href).name.replace("-Stats", "")
            full_url = FBREF_BASE + href
            if re.search(r"/\d{4}-\d{4}/", href):
                season_links.setdefault(team_id, {"name": name, "url": full_url})
            else:
                generic_links.setdefault(team_id, {"name": name, "url": full_url})

        chosen = season_links or generic_links
        teams.extend(chosen.values())

    # Only return true squad links; handling of aggregate pages is done separately.

    # Deduplicate the list of teams based on URL
    seen_urls = set()
    unique_teams = []
    for team in teams:
        if team["url"] not in seen_urls:
            unique_teams.append(team)
            seen_urls.add(team["url"])

    return unique_teams[:max_teams] if max_teams else unique_teams


def _fetch_team_stats(client: httpx.Client, team_url: str, season: Optional[str]) -> Optional[pd.DataFrame]:
    """
    Fetches all player statistics tables for a single team.
    Constructs a season-specific URL if a season is provided.
    """
    # Construct a season-specific URL if applicable. If pattern doesn't match, fall back to given URL.
    if season and "/squads/" in team_url:
        try:
            parts = team_url.split("/squads/")
            team_id = parts[1].split("/")[0]
            tail = parts[1].split("/")[-1]
            team_url = f"{parts[0]}/squads/{team_id}/{season}/{tail}"
        except Exception:
            pass

    html = _http_get_text(client, team_url)
    if not html:
        print(f"[fbref-scraper] Failed to load page: {team_url}")
        return None

    html = _uncomment_fbref(html)
    soup = BeautifulSoup(html, "lxml")
    h1 = soup.select_one("h1")
    h1_text = h1.get_text(strip=True) if h1 else ""
    team_name = h1_text.split("Stats")[0].strip() if h1_text else "Unknown"
    # Try to detect season from the H1 (e.g., "2024-2025 Arsenal Stats")
    detected_season = None
    m_season = re.search(r"(\d{4}-\d{4})", h1_text)
    if m_season:
        detected_season = m_season.group(1)

    # If we landed on a generic team page without tables, try to follow to a season page
    if "Standard Stats" not in html:
        m = re.search(r"/en/squads/([a-f0-9]{8})", team_url)
        team_id = m.group(1) if m else None
        if team_id:
            candidates = []
            for a in soup.select(f'a[href^="/en/squads/{team_id}/"]'):
                href = a.get("href") or ""
                if "-Stats" not in href:
                    continue
                if season and f"/{season}/" not in href:
                    continue
                if not season and re.search(r"/\d{4}-\d{4}/", href):
                    candidates.append(href)
                elif season:
                    candidates.append(href)
            if candidates:
                chosen = sorted(set(candidates), key=len)[0]
                new_url = FBREF_BASE + chosen if not chosen.startswith("http") else chosen
                alt_html = _http_get_text(client, new_url)
                if alt_html:
                    html = _uncomment_fbref(alt_html)
                    soup = BeautifulSoup(html, "lxml")
                    h1 = soup.select_one("h1")
                    h1_text = h1.get_text(strip=True) if h1 else h1_text
                    team_name = h1_text.split("Stats")[0].strip() if h1_text else team_name
                    m_season = re.search(r"(\d{4}-\d{4})", h1_text or "")
                    if m_season:
                        detected_season = detected_season or m_season.group(1)
            # Look for dropdown/option values listing seasons
            if "Standard Stats" not in html and team_id:
                option_candidates = []
                for opt in soup.select("option[value]"):
                    val = opt.get("value") or ""
                    if val.startswith(f"/en/squads/{team_id}/") and val.endswith("-Stats"):
                        ms = re.search(r"/(\d{4}-\d{4})/", val)
                        season_str = ms.group(1) if ms else None
                        option_candidates.append((season_str, val))
                # Prefer latest season (lexicographically descending works for YYYY-YYYY)
                option_candidates = [x for x in option_candidates if x[0]]
                if option_candidates:
                    option_candidates.sort(key=lambda x: x[0], reverse=True)
                    for _, chosen_val in option_candidates:
                        new_url = FBREF_BASE + chosen_val
                        alt_html = _http_get_text(client, new_url)
                        if not alt_html:
                            continue
                        trial_html = _uncomment_fbref(alt_html)
                        trial_soup = BeautifulSoup(trial_html, "lxml")
                        if trial_soup.select_one('table[id^="stats_standard"]') is None:
                            continue
                        html = trial_html
                        soup = trial_soup
                        h1 = soup.select_one("h1")
                        h1_text = h1.get_text(strip=True) if h1 else h1_text
                        team_name = h1_text.split("Stats")[0].strip() if h1_text else team_name
                        m_season = re.search(r"(\d{4}-\d{4})", h1_text or "")
                        if m_season:
                            detected_season = detected_season or m_season.group(1)
                        break
            # If still nothing, try constructing a season URL using detected season
            if "Standard Stats" not in html and detected_season:
                try:
                    parts = team_url.split("/squads/")
                    team_id2 = parts[1].split("/")[0]
                    tail = parts[1].split("/")[-1]
                    constructed = f"{parts[0]}/squads/{team_id2}/{detected_season}/{tail}"
                    alt_html = _http_get_text(client, constructed)
                    if alt_html:
                        html = _uncomment_fbref(alt_html)
                        soup = BeautifulSoup(html, "lxml")
                except Exception:
                    pass

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

    # Known table id prefixes on FBref
    table_id_prefixes = {
        "standard": ["stats_standard"],
        "shooting": ["stats_shooting"],
        "passing": ["stats_passing"],
        "pass_types": ["stats_passing_types", "stats_pass_types"],
        "gca": ["stats_gca"],
        "defense": ["stats_defense", "stats_def_actions"],
        "possession": ["stats_possession"],
        "playing_time": ["stats_playing_time"],
        "misc": ["stats_misc"],
        "goalkeeping": ["stats_keeper"],
        "adv_goalkeeping": ["stats_keeper_adv"],
    }

    dfs: Dict[str, pd.DataFrame] = {}
    effective_season = season or detected_season or "current"

    for cat, match_str in categories.items():
        try:
            # Prefer selecting the table element by id prefix for robustness
            table_html = None
            for pref in table_id_prefixes.get(cat, []):
                tbl = soup.select_one(f'table[id^="{pref}"]')
                if tbl is not None:
                    table_html = str(tbl)
                    break
            if table_html is not None:
                tables = pd.read_html(io.StringIO(table_html), flavor="lxml")
            else:
                # Fallback: try matching by caption text
                tables = pd.read_html(io.StringIO(html), match=match_str, flavor="lxml")
            if not tables:
                continue
            df = tables[0].copy()

            # Robustly flatten columns and clean names
            df.columns = _flatten_columns(df.columns)

            if "Player" not in df.columns:
                continue

            # Filter out non-player rows
            df = df[df["Player"].notna() & ~df["Player"].isin(["Squad Total", "Opponent Total"])]
            if "Rk" in df.columns:
                df = df[df["Rk"].astype(str).str.isdigit()]

            df["team"] = team_name
            df["season"] = effective_season

            key_cols = ["Player", "Nation", "Pos", "Age", "team", "season"]
            prefix_cols = [c for c in df.columns if c not in key_cols]
            df = df.rename(columns={c: f"{cat}_{c}" for c in prefix_cols})
            dfs[cat] = df
        except Exception as e:
            # This helps debug if a specific table fails to parse
            print(f"[fbref-scraper] Could not parse '{match_str}' table for {team_name}: {e}")

    if "standard" not in dfs:
        # Fallback: try to locate any plausible player table
        std_tbl = soup.select_one('table[id*="stats_standard"], table[id^="stats_standard"]')
        fallback_df: Optional[pd.DataFrame] = None
        try:
            if std_tbl is not None:
                tmp = pd.read_html(io.StringIO(str(std_tbl)), flavor="lxml")
                if tmp:
                    fallback_df = tmp[0]
            else:
                # As a last resort, scan all tables and pick one with a Player column
                for tbl in soup.find_all("table"):
                    tmp = pd.read_html(io.StringIO(str(tbl)), flavor="lxml")
                    if not tmp:
                        continue
                    cand = tmp[0]
                    if "Player" in list(cand.columns):
                        fallback_df = cand
                        break
        except Exception:
            fallback_df = None

        if fallback_df is None or fallback_df.empty:
            print(f"[fbref-scraper] No standard stats found for {team_name} at {team_url}")
            return None

        df = fallback_df.copy()
        df.columns = _flatten_columns(df.columns)
        if "Player" not in df.columns:
            print(f"[fbref-scraper] Fallback table lacks Player column for {team_name}")
            return None
        df = df[df["Player"].notna() & ~df["Player"].isin(["Squad Total", "Opponent Total"])]
        if "Rk" in df.columns:
            try:
                df = df[df["Rk"].astype(str).str.isdigit()]
            except Exception:
                pass
        df["team"] = team_name
        df["season"] = effective_season
        dfs["standard"] = df

    # Merge all dataframes into one, starting with standard stats
    final_df = dfs["standard"]
    for cat, df_to_merge in dfs.items():
        if cat == "standard":
            continue
        final_df = pd.merge(
            final_df,
            df_to_merge,
            on=["Player", "Nation", "Pos", "Age", "team", "season"],
            how="left",
            suffixes=("", f"_{cat}_dup"), # Add suffix to avoid column clashes
        )
    
    # Remove any fully duplicated columns that may arise from merges
    final_df = final_df.loc[:, ~final_df.columns.duplicated(keep="first")]
    return final_df


def _seasonize_league_url(url: str, season: Optional[str]) -> str:
    """Attempt to coerce a league URL to a season-specific URL.

    Examples:
      /en/comps/9/Premier-League-Stats -> /en/comps/9/2024-2025/2024-2025-Premier-League-Stats
      /en/comps/12/La-Liga-Stats      -> /en/comps/12/2024-2025/2024-2025-La-Liga-Stats
    """
    if not season:
        return url

    try:
        # Normalize to path portion only for regex and rebuild with domain afterwards
        path = url
        if url.startswith("http"):
            path = "/" + url.split("/", 3)[3]

        m = re.match(r"^(/en/comps/\d+)(?:/\d{4}-\d{4})?/(.+-Stats)$", path)
        if not m:
            return url
        base, slug = m.groups()
        if slug.startswith(season):
            new_path = f"{base}/{season}/{slug}"
        else:
            new_path = f"{base}/{season}/{season}-{slug}"
        return FBREF_BASE + new_path if not url.startswith("http") else FBREF_BASE + new_path
    except Exception:
        return url


def _extract_league_pages(aggregate_html: str, season: Optional[str] = None) -> List[Dict[str, str]]:
    """Extract per-league stats pages from an aggregate page (e.g., Big 5).

    Returns list of dicts with name and URL. If a season is provided, prefer
    links that include that season; otherwise attempt to seasonize generic links.
    """
    aggregate_html = _uncomment_fbref(aggregate_html)
    soup = BeautifulSoup(aggregate_html, "lxml")

    # Collect candidate links grouped by competition id
    by_comp: Dict[str, List[str]] = {}
    for a in soup.select('a[href^="/en/comps/"]'):
        href = (a.get("href") or "").strip()
        text = a.get_text(strip=True)
        if not href or "-Stats" not in href or not text.endswith("Stats"):
            continue
        # Skip the aggregate Big-5 page itself
        if "Big-5-European-Leagues-Stats" in href:
            continue
        m = re.match(r"^/en/comps/(\d+)(/.*)?/(.+-Stats)$", href)
        if not m:
            continue
        comp_id = m.group(1)
        by_comp.setdefault(comp_id, []).append(href)

    league_pages: List[Dict[str, str]] = []
    for comp_id, hrefs in by_comp.items():
        # Prefer season-specific link when requested
        chosen: Optional[str] = None
        if season:
            for h in hrefs:
                if f"/{season}/" in h:
                    chosen = h
                    break
        # Fall back to the first generic stats link
        if not chosen and hrefs:
            chosen = sorted(hrefs, key=len)[0]

        if not chosen:
            continue

        full = FBREF_BASE + chosen if not chosen.startswith("http") else chosen
        name = Path(chosen).name.replace("-Stats", "")
        if season and f"/{season}/" not in full:
            full = _seasonize_league_url(full, season)
        league_pages.append({"name": name, "url": full})

    if not league_pages:
        # Heuristic fallback for the Big 5 aggregate landing page
        if "Big-5-European-Leagues-Stats" in aggregate_html:
            comp_slug = {
                "9": "Premier-League-Stats",
                "12": "La-Liga-Stats",
                "20": "Bundesliga-Stats",
                "11": "Serie-A-Stats",
                "13": "Ligue-1-Stats",
            }
            for comp_id, slug in comp_slug.items():
                base = f"{FBREF_BASE}/en/comps/{comp_id}/{slug}"
                url = _seasonize_league_url(base, season) if season else base
                league_pages.append({"name": slug.replace("-Stats", ""), "url": url})

    return league_pages



def scrape_league(
    league_url: str,
    season: Optional[str] = None,
    max_teams: Optional[int] = None,
) -> pd.DataFrame:
    """Main function to orchestrate the scraping of an entire league."""
    with _build_http_client() as client:
        print(f"[fbref-scraper] Loading league page: {league_url}")
        league_html = _http_get_text(client, league_url)
        if not league_html:
            print("[fbref-scraper] Could not fetch league page.")
            return pd.DataFrame()
        _polite_delay(2.5, 5.0)

        teams = _extract_team_links(league_html, max_teams=max_teams)
        if not teams:
            print("[fbref-scraper] Could not find any team links on page."
                  " If this is an aggregate page, try --all-leagues.")
            return pd.DataFrame()

        print(f"[fbref-scraper] Found {len(teams)} teams to scrape.")

        all_frames: List[pd.DataFrame] = []
        for i, team in enumerate(teams):
            print(f"[fbref-scraper] ({i + 1}/{len(teams)}) Scraping: {team['name']}")
            team_df = _fetch_team_stats(client, team["url"], season)
            if team_df is not None:
                all_frames.append(team_df)
            # Be respectful with delays between team requests
            _polite_delay(6.0, 12.0)

        if not all_frames:
            return pd.DataFrame()

        return pd.concat(all_frames, ignore_index=True)


def scrape_all_leagues(
    aggregate_url: str = DEFAULT_LEAGUE_URL,
    season: Optional[str] = None,
    max_teams: Optional[int] = None,
) -> pd.DataFrame:
    """Scrape every league linked from an aggregate page (e.g., Big 5).

    For each league page discovered, fetch team squad links and scrape player
    stats per squad, concatenating all into a single DataFrame.
    """
    with _build_http_client() as client:
        print(f"[fbref-scraper] Loading aggregate page: {aggregate_url}")
        agg_html = _http_get_text(client, aggregate_url)
        if not agg_html:
            print("[fbref-scraper] Could not fetch aggregate page.")
            return pd.DataFrame()
        _polite_delay(2.5, 5.0)

        leagues = _extract_league_pages(agg_html, season=season)
        if not leagues:
            print("[fbref-scraper] No leagues found on aggregate page.")
            return pd.DataFrame()

        print(f"[fbref-scraper] Found {len(leagues)} leagues. Scraping squads for each...")

        all_frames: List[pd.DataFrame] = []
        for i, league in enumerate(leagues):
            print(f"[fbref-scraper] League ({i + 1}/{len(leagues)}): {league['name']}")
            league_html = _http_get_text(client, league["url"]) or ""
            if not league_html:
                print(f"[fbref-scraper] Skipping league (failed to fetch): {league['url']}")
                continue
            _polite_delay(2.0, 4.0)

            # If the league page is generic, try to follow the latest season option to get season-specific squad links
            if league_html and (season is None or season not in league["url"]):
                soup = BeautifulSoup(_uncomment_fbref(league_html), "lxml")
                comp_id_match = re.search(r"/en/comps/(\d+)", league["url"])
                comp_id = comp_id_match.group(1) if comp_id_match else None
                option_candidates = []
                if comp_id:
                    for opt in soup.select("option[value]"):
                        val = opt.get("value") or ""
                        if val.startswith(f"/en/comps/{comp_id}/") and val.endswith("-Stats"):
                            ms = re.search(r"/(\d{4}-\d{4})/", val)
                            season_str = ms.group(1) if ms else None
                            option_candidates.append((season_str, val))
                option_candidates = [x for x in option_candidates if x[0]]
                if option_candidates:
                    option_candidates.sort(key=lambda x: x[0], reverse=True)
                    for _, chosen_val in option_candidates:
                        new_url = FBREF_BASE + chosen_val
                        alt_html = _http_get_text(client, new_url)
                        if not alt_html:
                            continue
                        league_html = alt_html
                        break

            teams = _extract_team_links(league_html, max_teams=max_teams)
            print(f"[fbref-scraper]  - Found {len(teams)} teams")

            for j, team in enumerate(teams):
                print(f"[fbref-scraper]    ({j + 1}/{len(teams)}) {team['name']}")
                team_df = _fetch_team_stats(client, team["url"], season)
                if team_df is not None:
                    all_frames.append(team_df)
                _polite_delay(6.0, 12.0)

        if not all_frames:
            return pd.DataFrame()

        return pd.concat(all_frames, ignore_index=True)

def main():
    """Main entry point for the command-line script."""
    parser = argparse.ArgumentParser(description="Scrape player stats from FBref.com")
    parser.add_argument(
        "--league-url",
        default=DEFAULT_LEAGUE_URL,
        help=(
            "URL of a league page (scrapes that league) or an aggregate page like"
            " Big 5 (use with --all-leagues)."
        ),
    )
    parser.add_argument("--season", default=None, help="Season in YYYY-YYYY format (e.g., '2023-2024').")
    parser.add_argument("--outfile", default=DEFAULT_OUT, help="Path to save the output CSV file.")
    parser.add_argument("--max-teams", type=int, default=None, help="Limit the number of teams to scrape (for testing).")
    parser.add_argument(
        "--all-leagues",
        action="store_true",
        help="Treat --league-url as an aggregate page and scrape every linked league.",
    )
    args = parser.parse_args()

    ensure_dirs()
    if args.all_leagues:
        df = scrape_all_leagues(args.league_url, args.season, args.max_teams)
    else:
        df = scrape_league(args.league_url, args.season, args.max_teams)

    if df.empty:
        print("[fbref-scraper] Scraping finished with no data. Check the league URL and season format.")
        sys.exit(1)

    out_path = Path(args.outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n[fbref-scraper] Success! Wrote {len(df)} rows to {out_path.resolve()}")


if __name__ == "__main__":
    main()
