"""
Minimal Transfermarkt fetcher.
Fetches ONE url and saves the raw HTML under data/raw/tm/.

Usage (PowerShell from project root):
  $env:PYTHONPATH = "$PWD\src"
  python -m scouting_ml.tm.tm_scraper --url "https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query=sturm%20graz" --out "sturm_graz_search.html"
"""

from pathlib import Path
from typing import Optional
import time
import httpx
import typer
from scouting_ml.utils.import_guard import *  # noqa: F403
from scouting_ml.paths import ensure_dirs, tm_html

app = typer.Typer(add_completion=False)

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

def fetch(url: str, timeout: float = 20.0, retries: int = 2, sleep_between: float = 1.5) -> bytes:
    headers = {
        "User-Agent": DEFAULT_UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    last_err = None
    for attempt in range(retries + 1):
        try:
            with httpx.Client(follow_redirects=True, headers=headers, timeout=timeout) as client:
                r = client.get(url)
                r.raise_for_status()
                return r.content
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(sleep_between)
            else:
                raise
    # Should never reach here
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")

def safe_name(name: str) -> str:
    # Very simple filename sanitizer
    s = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in name)
    return s[:150]  # keep it short-ish

@app.command()
def save(
    url: str = typer.Option(..., "--url", help="Full Transfermarkt URL to download"),
    out: Optional[str] = typer.Option(None, "--out", help="Output filename (defaults to auto-generated)"),
) -> None:
    """Download one URL and write it to data/raw/tm/<filename>.html"""
    ensure_dirs()
    html = fetch(url)

    if out is None:
        # auto name from path/query
        from urllib.parse import urlparse
        parsed = urlparse(url)
        auto = safe_name((parsed.path.strip("/") or "index") + ("_" + parsed.query if parsed.query else "")) + ".html"
        out = auto

    dest: Path = tm_html(out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(html)
    print(f"Saved {len(html):,} bytes → {dest}")

if __name__ == "__main__":
    app()
