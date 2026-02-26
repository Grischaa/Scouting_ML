import sys
from pathlib import Path

import httpx


def main(url: str, out_path: str) -> None:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/128.0.0.0 Safari/537.36"
        )
    }
    with httpx.Client(http2=True, follow_redirects=True, headers=headers, timeout=30.0) as c:
        r = c.get(url)
        r.raise_for_status()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(r.text, encoding="utf-8")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

