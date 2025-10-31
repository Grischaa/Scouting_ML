# src/scouting_ml/tm_scraper.py
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from scouting_ml.utils.import_guard import *  # noqa: F403
import httpx
import requests

from scouting_ml.logging import get_logger
from scouting_ml.paths import ensure_dirs, tm_html

logger = get_logger()

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Connection": "keep-alive",
}

# Keep it simple (HTTP/1.1) to avoid extra deps; flip to True if you installed httpx[http2]
HTTP2 = False

CLIENT = httpx.Client(
    headers=HEADERS,
    http2=HTTP2,
    follow_redirects=True,
    timeout=httpx.Timeout(connect=12.0, read=40.0, write=20.0, pool=40.0),
)

# Be conservative with these markers; many legit pages mention "cookie" in inline JS.
CONSENT_MARKERS = (
    b">Privacy settings<",
    b"<title>Privacy settings</title>",
    b"enable javascript",
    b"bot protection",
)

def _looks_like_consent(content: bytes) -> bool:
    low = content[:16000].lower()
    return any(m in low for m in CONSENT_MARKERS)

def _httpx_get(url: str) -> bytes:
    resp = CLIENT.get(url)
    # retry-worthy statuses will be handled in fetch()
    resp.raise_for_status()
    return resp.content

def _requests_get(url: str) -> bytes:
    r = requests.get(url, headers=HEADERS, allow_redirects=True, timeout=40)
    r.raise_for_status()
    return r.content

def fetch(url: str, *, retries: int = 5, backoff: float = 0.9) -> bytes:
    """
    Robust fetch with httpx primary + requests fallback.
    On the last attempt, returns the body even if it still looks like a consent page.
    """
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            # 1) primary: httpx
            try:
                content = _httpx_get(url)
            except Exception as e_httpx:
                # 2) fallback: requests
                content = _requests_get(url)

            # status ok and we have bytes; decide about "consent"
            if _looks_like_consent(content):
                if attempt < retries:
                    delay = backoff * attempt + random.uniform(0.05, 0.4)
                    time.sleep(delay)
                    continue
                else:
                    # final attempt: return anyway so caller can inspect/parse
                    return content

            return content

        except (httpx.ReadTimeout, httpx.ConnectError, httpx.HTTPStatusError, requests.RequestException) as e:
            last_exc = e
            delay = backoff * attempt + random.uniform(0.1, 0.6)
            time.sleep(delay)
            continue

    # All retries exhausted: dump a tiny debug to help diagnose
    dbg = Path("data/raw/tm/_last_fetch_debug.txt")
    try:
        dbg.parent.mkdir(parents=True, exist_ok=True)
        dbg.write_text(f"fetch failed for URL: {url}\nlast_exc: {repr(last_exc)}\n", encoding="utf-8")
    except Exception:
        pass

    if last_exc:
        raise last_exc
    raise RuntimeError("fetch failed unexpectedly")


def main():
    parser = argparse.ArgumentParser(description="Download one URL and save to data/raw/tm/")
    parser.add_argument("--url", required=True, help="Full URL to download")
    parser.add_argument("--out", default="page.html", help="Output filename (under data/raw/tm/)")
    args = parser.parse_args()

    ensure_dirs()
    html = fetch(args.url)
    dest: Path = tm_html(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(html)
    logger.success(f"Saved file to {dest.resolve()} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
