# src/scouting_ml/fbref/fbref_api_scraper.py
from __future__ import annotations

import argparse
import io
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from bs4 import BeautifulSoup, Comment
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from stealth import stealth
from webdriver_manager.chrome import ChromeDriverManager

# A simple fallback for the ensure_dirs function if the original import fails.
def ensure_dirs():
    """Ensures necessary data directories exist."""
    Path("data/processed").mkdir(parents=True, exist_ok=True)


FBREF_BASE = "https://fbref.com"
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


def setup_driver() -> webdriver.Chrome:
    """Sets up a headless Chrome WebDriver with anti-detection measures."""
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    )
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    # Use webdriver-manager to handle driver installation
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    # Apply selenium-stealth patches
    stealth(
        driver,
        languages=["en-US", "en"],
        vendor="Google Inc.",
        platform="Win32",
        webgl_vendor="Intel Inc.",
        renderer="Intel Iris OpenGL Engine",
        fix_hairline=True,
    )

    return driver


def _extract_team_links(league_html: str, max_teams: Optional[int] = None) -> List[Dict[str, str]]:
    """
    Returns a list of dictionaries {'name': ..., 'url': ...} for squad pages.
    This function uses an updated and more reliable CSS selector.
    """
    league_html = _uncomment_fbref(league_html)
    soup = BeautifulSoup(league_html, "lxml")

    teams: List[Dict[str, str]] = []

    # Updated selector for the main league table. This is more stable.
    # It targets the `<a>` tag within the table cell (`<td>`) with `data-stat="squad"`.
    for link in soup.select('td[data-stat="squad"] a[href*="/squads/"]'):
        href = link.get("href")
        if href:
            full_url = FBREF_BASE + href if not href.startswith("http") else href
            teams.append({"name": link.get_text(strip=True), "url": full_url})

    # Deduplicate the list of teams based on URL
    seen_urls = set()
    unique_teams = []
    for team in teams:
        if team["url"] not in seen_urls:
            unique_teams.append(team)
            seen_urls.add(team["url"])

    return unique_teams[:max_teams] if max_teams else unique_teams


def _fetch_team_stats(driver: webdriver.Chrome, team_url: str, season: Optional[str]) -> Optional[pd.DataFrame]:
    """
    Fetches all player statistics tables for a single team.
    Constructs a season-specific URL if a season is provided.
    """
    # Construct a season-specific URL if applicable
    if season and "/squads/" in team_url:
        parts = team_url.split("/squads/")
        # Example: https://fbref.com/en/squads/361ca564/2023-2024/Manchester-City-Stats
        team_url = f"{parts[0]}/squads/{parts[1].split('/')[0]}/{season}/{parts[1].split('/')[-1]}"

    html = None
    # Retry mechanism for fetching the page
    for attempt in range(3):
        try:
            driver.get(team_url)
            time.sleep(random.uniform(3, 6))  # Wait for the page to load
            html = driver.page_source
            if "Too Many Requests" not in html:
                break
            print(f"[fbref-scraper] Rate limit hit for {team_url}. Retrying in 20-30s...")
            time.sleep(random.uniform(20, 30))
        except WebDriverException as e:
            print(f"[fbref-scraper] Retry {attempt + 1}/3 for {team_url}: {e}")
            time.sleep(random.uniform(10, 20))
    
    if not html:
        print(f"[fbref-scraper] Failed to load page: {team_url}")
        return None

    html = _uncomment_fbref(html)
    soup = BeautifulSoup(html, "lxml")
    h1 = soup.select_one("h1")
    team_name = h1.get_text(strip=True).split("Stats")[0].strip() if h1 else "Unknown"

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
            tables = pd.read_html(io.StringIO(html), match=match_str, flavor="lxml")
            if not tables:
                continue
            df = tables[0].copy()

            # Robustly flatten multi-index columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ["_".join(col).strip() for col in df.columns.values]

            # Clean up remaining non-ASCII characters or extra spaces in column names
            df.columns = df.columns.str.replace(r"[^A-Za-z0-9_]+", "", regex=True)

            if "Player" not in df.columns:
                continue

            # Filter out non-player rows
            df = df[df["Player"].notna() & ~df["Player"].isin(["Squad Total", "Opponent Total"])]
            if "Rk" in df.columns:
                df = df[df["Rk"].astype(str).str.isdigit()]

            df["team"] = team_name
            df["season"] = season or "current"

            key_cols = ["Player", "Nation", "Pos", "Age", "team", "season"]
            prefix_cols = [c for c in df.columns if c not in key_cols]
            df = df.rename(columns={c: f"{cat}_{c}" for c in prefix_cols})
            dfs[cat] = df
        except Exception as e:
            # This helps debug if a specific table fails to parse
            print(f"[fbref-scraper] Could not parse '{match_str}' table for {team_name}: {e}")

    if "standard" not in dfs:
        print(f"[fbref-scraper] No standard stats found for {team_name} at {team_url}")
        return None

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


def scrape_league(
    league_url: str,
    season: Optional[str] = None,
    max_teams: Optional[int] = None,
) -> pd.DataFrame:
    """Main function to orchestrate the scraping of an entire league."""
    driver = setup_driver()
    try:
        print(f"[fbref-scraper] Loading league page: {league_url}")
        driver.get(league_url)
        time.sleep(random.uniform(5, 10))
        league_html = driver.page_source

        teams = _extract_team_links(league_html, max_teams=max_teams)
        if not teams:
            print("[fbref-scraper] Could not find any team links. The website structure may have changed.")
            return pd.DataFrame()

        print(f"[fbref-scraper] Found {len(teams)} teams to scrape.")

        all_frames: List[pd.DataFrame] = []
        for i, team in enumerate(teams):
            print(f"[fbref-scraper] ({i + 1}/{len(teams)}) Scraping: {team['name']}")
            team_df = _fetch_team_stats(driver, team["url"], season)
            if team_df is not None:
                all_frames.append(team_df)
            # Be respectful with delays
            time.sleep(random.uniform(8, 15))

        if not all_frames:
            return pd.DataFrame()

        return pd.concat(all_frames, ignore_index=True)

    finally:
        driver.quit()


def main():
    """Main entry point for the command-line script."""
    parser = argparse.ArgumentParser(description="Scrape player stats for a league from FBref.com")
    parser.add_argument("--league-url", default=DEFAULT_LEAGUE_URL, help="Full URL of the FBref league page.")
    parser.add_argument("--season", default=None, help="Season in YYYY-YYYY format (e.g., '2023-2024').")
    parser.add_argument("--outfile", default=DEFAULT_OUT, help="Path to save the output CSV file.")
    parser.add_argument("--max-teams", type=int, default=None, help="Limit the number of teams to scrape (for testing).")
    args = parser.parse_args()

    ensure_dirs()
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