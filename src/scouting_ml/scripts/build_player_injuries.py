from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence
from urllib.parse import urlparse, urlunparse

import pandas as pd
from bs4 import BeautifulSoup

from scouting_ml.tm.tm_scraper import fetch

TM_ID_RE = re.compile(r"/spieler/(\d+)")
INT_RE = re.compile(r"(\d+)")


def normalize_season_label(value: str | float | int | None) -> str | None:
    if value is None:
        return None
    season = str(value).strip().replace("\\", "/")
    if not season:
        return None
    if "-" in season and "/" not in season:
        season = season.replace("-", "/")

    m_full = re.match(r"^(\d{4})/(\d{2}|\d{4})$", season)
    if m_full:
        start = int(m_full.group(1))
        end = m_full.group(2)
        end2 = end[-2:] if len(end) == 4 else end
        return f"{start}/{end2}"

    m_short = re.match(r"^(\d{2})/(\d{2})$", season)
    if m_short:
        start2 = int(m_short.group(1))
        start = 2000 + start2 if start2 <= 69 else 1900 + start2
        return f"{start}/{m_short.group(2)}"

    m_year = re.match(r"^(\d{4})$", season)
    if m_year:
        year = int(m_year.group(1))
        return f"{year-1}/{str(year)[-2:]}"
    return season


def season_start_year(season: str | None) -> int | None:
    if season is None:
        return None
    m = re.match(r"^(\d{4})/\d{2}$", season)
    if not m:
        return None
    return int(m.group(1))


def in_season_range(season: str, start_season: str | None, end_season: str | None) -> bool:
    y = season_start_year(season)
    if y is None:
        return False
    if start_season:
        y0 = season_start_year(start_season)
        if y0 is not None and y < y0:
            return False
    if end_season:
        y1 = season_start_year(end_season)
        if y1 is not None and y > y1:
            return False
    return True


def extract_transfermarkt_id(link: str | None) -> str | None:
    if not link:
        return None
    m = TM_ID_RE.search(link)
    return m.group(1) if m else None


def build_injuries_url(profile_link: str, transfermarkt_id: str) -> str:
    parsed = urlparse(profile_link)
    path = parsed.path or ""
    if "/profil/spieler/" in path:
        injury_path = path.replace("/profil/spieler/", "/verletzungen/spieler/")
    else:
        injury_path = f"/-/verletzungen/spieler/{transfermarkt_id}"
    return urlunparse((parsed.scheme or "https", parsed.netloc or "www.transfermarkt.com", injury_path, "", "", ""))


def parse_int(text: str) -> int:
    if not text:
        return 0
    m = INT_RE.search(text.replace(",", ""))
    return int(m.group(1)) if m else 0


def _detect_column_index(headers: Sequence[str], tokens: Sequence[str], fallback: int | None = None) -> int | None:
    for i, header in enumerate(headers):
        h = header.lower()
        if all(tok in h for tok in tokens):
            return i
    for i, header in enumerate(headers):
        h = header.lower()
        if any(tok in h for tok in tokens):
            return i
    return fallback


def parse_injury_rows(html: bytes) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    out: list[dict] = []

    for table in soup.select("table.items"):
        header_cells = table.select("thead th")
        headers = [c.get_text(" ", strip=True).lower() for c in header_cells]
        season_idx = _detect_column_index(headers, ("season",), fallback=0)
        if season_idx is None:
            season_idx = _detect_column_index(headers, ("saison",), fallback=0)
        injury_idx = _detect_column_index(headers, ("injury",), fallback=1)
        if injury_idx is None:
            injury_idx = _detect_column_index(headers, ("verletzung",), fallback=1)
        days_idx = _detect_column_index(headers, ("day",), fallback=4)
        if days_idx is None:
            days_idx = _detect_column_index(headers, ("tage",), fallback=4)
        games_idx = _detect_column_index(headers, ("games", "miss"), fallback=5)
        if games_idx is None:
            games_idx = _detect_column_index(headers, ("spiele",), fallback=5)

        for row in table.select("tbody tr"):
            cells = row.find_all("td")
            if len(cells) < 2:
                continue
            values = [c.get_text(" ", strip=True) for c in cells]
            if season_idx >= len(values):
                continue
            season = normalize_season_label(values[season_idx])
            if not season:
                continue

            injury_name = values[injury_idx] if injury_idx is not None and injury_idx < len(values) else ""
            days = parse_int(values[days_idx]) if days_idx is not None and days_idx < len(values) else 0
            games = parse_int(values[games_idx]) if games_idx is not None and games_idx < len(values) else 0
            out.append(
                {
                    "season": season,
                    "injury_name": injury_name,
                    "days_missed": days,
                    "games_missed": games,
                }
            )
    return out


def _read_players_from_file(path: Path) -> pd.DataFrame:
    use_cols = ["player_id", "transfermarkt_id", "name", "dob", "season", "link"]
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
        cols = [c for c in use_cols if c in df.columns]
        return df[cols].copy()
    return pd.read_csv(path, usecols=lambda c: c in set(use_cols))


def load_player_index(players_source: Path) -> pd.DataFrame:
    if players_source.is_dir():
        files = sorted(players_source.glob("*_with_sofa.csv"))
        if not files:
            raise ValueError(f"No *_with_sofa.csv files found in {players_source}")
        frames = [_read_players_from_file(p) for p in files]
        base = pd.concat(frames, ignore_index=True, sort=False)
    else:
        base = _read_players_from_file(players_source)

    if base.empty:
        raise ValueError("No players found in source.")

    base["season"] = base["season"].apply(normalize_season_label)
    if "transfermarkt_id" not in base.columns:
        base["transfermarkt_id"] = pd.NA

    # Primary fallback: many datasets store Transfermarkt id in `player_id`.
    if "player_id" in base.columns:
        miss_tm = base["transfermarkt_id"].isna()
        base.loc[miss_tm, "transfermarkt_id"] = base.loc[miss_tm, "player_id"]

    base["transfermarkt_id"] = (
        base["transfermarkt_id"]
        .astype(str)
        .str.strip()
        .replace(
            {
                "": pd.NA,
                "nan": pd.NA,
                "NaN": pd.NA,
                "None": pd.NA,
                "<NA>": pd.NA,
                "N/A": pd.NA,
            }
        )
    )
    if "link" in base.columns:
        missing = base["transfermarkt_id"].isna()
        base.loc[missing, "transfermarkt_id"] = base.loc[missing, "link"].apply(extract_transfermarkt_id)

    base = base[base["transfermarkt_id"].notna()].copy()
    base["transfermarkt_id"] = base["transfermarkt_id"].astype(str)

    if "link" not in base.columns:
        base["link"] = "https://www.transfermarkt.com/-/profil/spieler/" + base["transfermarkt_id"]
    base["link"] = base["link"].astype(str)

    grouped_rows: list[dict] = []
    for tmid, grp in base.groupby("transfermarkt_id", dropna=False):
        seasons = sorted({s for s in grp["season"].dropna().astype(str)})
        row = {
            "transfermarkt_id": str(tmid),
            "player_id": grp["player_id"].dropna().astype(str).iloc[0] if "player_id" in grp.columns and grp["player_id"].notna().any() else "",
            "name": grp["name"].dropna().astype(str).iloc[0] if "name" in grp.columns and grp["name"].notna().any() else "",
            "dob": grp["dob"].dropna().astype(str).iloc[0] if "dob" in grp.columns and grp["dob"].notna().any() else "",
            "profile_link": grp["link"].dropna().astype(str).iloc[0],
            "target_seasons": seasons,
        }
        grouped_rows.append(row)
    return pd.DataFrame(grouped_rows)


def aggregate_player_injuries(
    rows: Iterable[dict],
    target_seasons: Sequence[str],
) -> Dict[str, dict]:
    agg: Dict[str, dict] = {
        s: {"injury_days_missed": 0, "injury_games_missed": 0, "injury_count": 0}
        for s in target_seasons
    }
    for row in rows:
        season = normalize_season_label(row.get("season"))
        if season not in agg:
            continue
        agg[season]["injury_days_missed"] += int(row.get("days_missed", 0) or 0)
        agg[season]["injury_games_missed"] += int(row.get("games_missed", 0) or 0)
        agg[season]["injury_count"] += 1
    return agg


def build_player_injuries(
    players_source: str,
    output: str,
    sleep_seconds: float = 2.5,
    max_players: int | None = None,
    start_season: str | None = None,
    end_season: str | None = None,
    include_failed: bool = True,
    overwrite_cache: bool = False,
) -> None:
    players_df = load_player_index(Path(players_source))
    if max_players is not None and max_players > 0:
        players_df = players_df.head(max_players).copy()

    cache_dir = Path("data/raw/tm/injuries")
    cache_dir.mkdir(parents=True, exist_ok=True)

    start_season_norm = normalize_season_label(start_season) if start_season else None
    end_season_norm = normalize_season_label(end_season) if end_season else None

    out_rows: list[dict] = []
    total = len(players_df)
    for idx, player in players_df.iterrows():
        tmid = str(player["transfermarkt_id"])
        profile_link = str(player["profile_link"])
        target_seasons = [s for s in player["target_seasons"] if in_season_range(s, start_season_norm, end_season_norm)]
        if not target_seasons:
            continue

        injury_url = build_injuries_url(profile_link, tmid)
        cache_path = cache_dir / f"{tmid}.html"

        scrape_success = 1
        injury_rows: list[dict] = []
        try:
            if overwrite_cache or (not cache_path.exists()):
                body = fetch(injury_url)
                cache_path.write_bytes(body)
                time.sleep(max(sleep_seconds, 0.0))
            else:
                body = cache_path.read_bytes()
            injury_rows = parse_injury_rows(body)
        except Exception as exc:
            scrape_success = 0
            print(f"[injuries] failed {tmid} ({player.get('name','')}): {exc}")

        if scrape_success == 0 and not include_failed:
            continue

        aggregated = aggregate_player_injuries(injury_rows, target_seasons=target_seasons)
        for season in target_seasons:
            base = {
                "player_id": player["player_id"],
                "transfermarkt_id": tmid,
                "name": player["name"],
                "dob": player["dob"],
                "season": season,
                "injury_source_url": injury_url,
                "injury_scrape_success": scrape_success,
                "injury_rows_parsed": len(injury_rows),
            }
            if scrape_success == 0:
                base.update(
                    {
                        "days_missed": pd.NA,
                        "games_missed": pd.NA,
                        "injury_count": pd.NA,
                        "major_injury_flag": pd.NA,
                    }
                )
            else:
                days = aggregated.get(season, {}).get("injury_days_missed", 0)
                games = aggregated.get(season, {}).get("injury_games_missed", 0)
                cnt = aggregated.get(season, {}).get("injury_count", 0)
                base.update(
                    {
                        "days_missed": int(days),
                        "games_missed": int(games),
                        "injury_count": int(cnt),
                        "major_injury_flag": int(days >= 60),
                    }
                )
            out_rows.append(base)

        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            print(f"[injuries] processed {idx+1:,}/{total:,} players")

    out_df = pd.DataFrame(out_rows)
    if out_df.empty:
        raise ValueError("No injury rows produced.")

    # Keep only one row per player-season (latest scrape if duplicates happened).
    out_df = out_df.sort_values(["transfermarkt_id", "season", "injury_scrape_success"], ascending=[True, True, False])
    out_df = out_df.drop_duplicates(subset=["transfermarkt_id", "season"], keep="first")

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    success_rate = float((out_df["injury_scrape_success"] == 1).mean() * 100.0)
    print(f"[injuries] wrote {len(out_df):,} rows -> {out_path}")
    print(f"[injuries] scrape success rate: {success_rate:,.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build season-level player injury features from Transfermarkt injury history pages."
    )
    parser.add_argument(
        "--players-source",
        default="data/model/big5_players.parquet",
        help="Input player table (CSV/Parquet) or directory with *_with_sofa.csv files.",
    )
    parser.add_argument(
        "--output",
        default="data/external/player_injuries.csv",
        help="Output CSV path.",
    )
    parser.add_argument("--sleep", type=float, default=2.5, help="Sleep seconds between web fetches.")
    parser.add_argument("--max-players", type=int, default=None, help="Optional cap for test runs.")
    parser.add_argument("--start-season", default=None, help="Optional season lower bound, e.g. 2019/20")
    parser.add_argument("--end-season", default=None, help="Optional season upper bound, e.g. 2024/25")
    parser.add_argument(
        "--no-failed-rows",
        action="store_true",
        help="Drop failed player-season rows instead of writing NaNs.",
    )
    parser.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="Refetch and overwrite cached injury HTML files.",
    )
    args = parser.parse_args()

    build_player_injuries(
        players_source=args.players_source,
        output=args.output,
        sleep_seconds=args.sleep,
        max_players=args.max_players,
        start_season=args.start_season,
        end_season=args.end_season,
        include_failed=not args.no_failed_rows,
        overwrite_cache=args.overwrite_cache,
    )


if __name__ == "__main__":
    main()
