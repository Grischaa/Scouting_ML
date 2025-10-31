# src/scouting_ml/fbref_scraper.py
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests

# if you moved helper stuff into src/scouting_ml/core, adapt this:
try:
    from scouting_ml.core.paths import RAW_DIR  # new layout
except ImportError:
    # fallback if you didn't move things yet
    RAW_DIR = Path("data/raw")

FBREF_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": FBREF_UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
        "Referer": "https://fbref.com/",
        "Connection": "keep-alive",
    }
)


def _get(url: str, *, retries: int = 2, sleep: float = 1.5) -> str:
    """GET with browser-like headers + a couple retries."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 2):
        try:
            resp = SESSION.get(url, timeout=25)
            if resp.status_code == 403:
                # write for debugging
                _save_debug_html("fbref_403.html", resp.text)
                raise requests.HTTPError(f"403 from FBref on {url}")
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            last_exc = e
            if attempt <= retries:
                time.sleep(sleep)
                continue
            raise
    # should not get here
    raise last_exc  # type: ignore


def _save_debug_html(name: str, html: str) -> None:
    out_dir = RAW_DIR / "fbref"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / name).write_text(html, encoding="utf-8")


def _read_fbref_table(html: str) -> pd.DataFrame:
    """
    FBref tables are normal <table> elements, so pandas.read_html works.
    We'll just take the first non-empty table for now.
    """
    tables = pd.read_html(html)
    for t in tables:
        if not t.empty:
            return t
    return pd.DataFrame()


def scrape_league(league_url: str) -> pd.DataFrame:
    print(f"[fbref] fetching league page: {league_url}")
    html = _get(league_url)
    _save_debug_html("last_league.html", html)

    df = _read_fbref_table(html)
    if df.empty:
        print("[fbref] ⚠ no tables found on page (maybe JS / protected?)")
    return df


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser("FBref league scraper")
    ap.add_argument("--league-url", required=True)
    ap.add_argument("--outfile", required=True)
    args = ap.parse_args()

    df = scrape_league(args.league_url)
    if df.empty:
        print("[fbref] no data scraped – check data/raw/fbref/last_league.html")
        return

    out_path = Path(args.outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[fbref] wrote {len(df)} rows -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
