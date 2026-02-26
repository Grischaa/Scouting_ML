# src/scouting_ml/sofa_rapid.py
from __future__ import annotations
import os
import json
from pathlib import Path
from scouting_ml.utils.import_guard import *  # noqa: F403
import requests

from scouting_ml.paths import RAW_DIR, ensure_dirs

SOFA_ROOT = RAW_DIR / "sofascore"
PROBE_DIR = SOFA_ROOT / "probe"

BASE_URL = "https://sofascore.p.rapidapi.com"
HOST = "sofascore.p.rapidapi.com"


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def probe_tvchannels(match_id: int = 13157877) -> Path:
    """
    Just call the endpoint we KNOW exists from RapidAPI UI:
      GET /tvchannels/get-available-countries?matchId=...
    and save it, so we prove our setup works.
    """
    api_key = os.environ.get("RAPIDAPI_KEY")
    if not api_key:
        raise RuntimeError("RAPIDAPI_KEY not set")

    ensure_dirs()

    url = f"{BASE_URL}/tvchannels/get-available-countries"
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": HOST,
    }
    params = {"matchId": str(match_id)}

    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    out_path = PROBE_DIR / f"tvchannels_{match_id}.json"
    _write_json(out_path, data)
    print(f"[rapid-sofa] wrote -> {out_path}")
    return out_path


def main() -> None:
    # call the probe for now
    probe_tvchannels(13157877)


if __name__ == "__main__":
    main()
