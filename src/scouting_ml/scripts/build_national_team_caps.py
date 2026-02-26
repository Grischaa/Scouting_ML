from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Dict, List, Sequence
from urllib.parse import urlparse, urlunparse

import pandas as pd
from bs4 import BeautifulSoup

from scouting_ml.tm.tm_scraper import fetch

TM_ID_RE = re.compile(r"/spieler/(\d+)")
INT_RE = re.compile(r"(\d+)")
CAPS_GOALS_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


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
    return int(m.group(1)) if m else None


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


def select_target_seasons(
    seasons: Sequence[str],
    start_season: str | None,
    end_season: str | None,
    latest_only: bool,
) -> List[str]:
    filtered = [s for s in seasons if in_season_range(s, start_season, end_season)]
    if not filtered:
        return []
    if latest_only:
        return [max(filtered)]
    return sorted(filtered)


def extract_transfermarkt_id(link: str | None) -> str | None:
    if not link:
        return None
    m = TM_ID_RE.search(link)
    return m.group(1) if m else None


def _numeric_id_from_key(tm_key: str | None) -> str | None:
    if not tm_key:
        return None
    s = str(tm_key).strip()
    if s.isdigit():
        return s
    m = re.search(r"(\d+)$", s)
    return m.group(1) if m else None


def _slug_from_profile_link(profile_link: str | None) -> str | None:
    if not profile_link:
        return None
    m = re.search(r"transfermarkt\.[^/]+/([^/]+)/profil/spieler/\d+", profile_link)
    if not m:
        return None
    return m.group(1).strip().replace("-", "_")


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
    if "player_id" in base.columns:
        miss_tm = base["transfermarkt_id"].isna()
        base.loc[miss_tm, "transfermarkt_id"] = base.loc[miss_tm, "player_id"]

    base["transfermarkt_id"] = (
        base["transfermarkt_id"]
        .astype(str)
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    )
    if "link" in base.columns:
        miss = base["transfermarkt_id"].isna()
        base.loc[miss, "transfermarkt_id"] = base.loc[miss, "link"].apply(extract_transfermarkt_id)

    base = base[base["transfermarkt_id"].notna()].copy()
    base["transfermarkt_id"] = base["transfermarkt_id"].astype(str)
    if "link" not in base.columns:
        base["link"] = "https://www.transfermarkt.com/-/profil/spieler/" + base["transfermarkt_id"]
    base["link"] = base["link"].astype(str)

    rows: List[dict] = []
    for tmid, grp in base.groupby("transfermarkt_id", dropna=False):
        seasons = sorted({s for s in grp["season"].dropna().astype(str)})
        rows.append(
            {
                "transfermarkt_id": str(tmid),
                "player_id": (
                    grp["player_id"].dropna().astype(str).iloc[0]
                    if "player_id" in grp.columns and grp["player_id"].notna().any()
                    else ""
                ),
                "name": (
                    grp["name"].dropna().astype(str).iloc[0]
                    if "name" in grp.columns and grp["name"].notna().any()
                    else ""
                ),
                "dob": (
                    grp["dob"].dropna().astype(str).iloc[0]
                    if "dob" in grp.columns and grp["dob"].notna().any()
                    else ""
                ),
                "profile_link": grp["link"].dropna().astype(str).iloc[0],
                "target_seasons": seasons,
            }
        )
    return pd.DataFrame(rows)


def resolve_cached_profile_html(tm_key: str, profile_link: str | None = None) -> Path | None:
    players_dir = Path("data/raw/tm/players")
    exact = players_dir / f"{tm_key}.html"
    if exact.exists():
        return exact

    slug = _slug_from_profile_link(profile_link)
    numeric_id = _numeric_id_from_key(tm_key) or extract_transfermarkt_id(profile_link)
    if slug and numeric_id:
        slug_file = players_dir / f"{slug}_{numeric_id}.html"
        if slug_file.exists():
            return slug_file

    if numeric_id:
        matches = sorted(players_dir.glob(f"*_{numeric_id}.html"))
        if matches:
            return matches[0]
    return None


def build_profile_url_from_id(tm_key: str) -> str:
    numeric_id = _numeric_id_from_key(tm_key)
    if numeric_id:
        return f"https://www.transfermarkt.com/-/profil/spieler/{numeric_id}"
    return f"https://www.transfermarkt.com/-/profil/spieler/{tm_key}"


def build_national_team_url(profile_link: str, transfermarkt_id: str) -> str:
    parsed = urlparse(profile_link)
    path = parsed.path or ""
    if "/profil/spieler/" in path:
        nt_path = path.replace("/profil/spieler/", "/nationalmannschaft/spieler/")
    else:
        nt_path = f"/-/nationalmannschaft/spieler/{transfermarkt_id}"
    return urlunparse(
        (
            parsed.scheme or "https",
            parsed.netloc or "www.transfermarkt.com",
            nt_path,
            "",
            "",
            "",
        )
    )


def _parse_int(text: str) -> int:
    if not text:
        return 0
    m = INT_RE.search(text.replace(",", ""))
    return int(m.group(1)) if m else 0


def _clean_label(label: str) -> str:
    return re.sub(r"\s+", " ", label).strip().lower().rstrip(":")


def extract_label_pairs(soup: BeautifulSoup) -> Dict[str, str]:
    pairs: Dict[str, str] = {}

    for li in soup.select("li.data-header__label"):
        value_el = li.find("span", class_=re.compile(r"data-header__content"))
        value = value_el.get_text(" ", strip=True) if value_el else ""
        text = li.get_text(" ", strip=True)
        label = text
        if value and value in text:
            label = text.replace(value, "", 1)
        elif ":" in text:
            label = text.split(":", 1)[0]
        label = _clean_label(label)
        if label and value and label not in pairs:
            pairs[label] = value

    regulars = soup.select("span.info-table__content--regular")
    for reg in regulars:
        label = _clean_label(reg.get_text(" ", strip=True))
        value = ""
        sib = reg.find_next_sibling("span")
        while sib is not None:
            cls = " ".join(sib.get("class", []))
            if "info-table__content--bold" in cls:
                value = sib.get_text(" ", strip=True)
                break
            sib = sib.find_next_sibling("span")
        if label and value and label not in pairs:
            pairs[label] = value

    return pairs


def _parse_caps_goals_value(text: str) -> tuple[int | None, int | None]:
    m = CAPS_GOALS_RE.search(text or "")
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def parse_profile_national_fields(html: bytes) -> dict:
    soup = BeautifulSoup(html, "lxml")
    pairs = extract_label_pairs(soup)

    caps_raw = ""
    for key, value in pairs.items():
        if "caps/goals" in key or "länderspieltore" in key:
            caps_raw = value
            break
    if not caps_raw:
        for li in soup.select("li.data-header__label"):
            text = li.get_text(" ", strip=True)
            if "Caps/Goals" in text or "Länderspiele/Tore" in text:
                span = li.find("span", class_=re.compile(r"data-header__content"))
                caps_raw = span.get_text(" ", strip=True) if span else text
                break

    caps, goals = _parse_caps_goals_value(caps_raw)

    current_team = ""
    for key, value in pairs.items():
        if "current international" in key or "aktueller nationalspieler" in key:
            current_team = value
            break

    return {
        "caps_raw": caps_raw if caps_raw else pd.NA,
        "senior_caps": caps if caps is not None else pd.NA,
        "senior_goals": goals if goals is not None else pd.NA,
        "current_team": current_team if current_team else pd.NA,
    }


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


def parse_national_team_rows(html: bytes) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    rows: list[dict] = []
    for table in soup.select("table.items"):
        headers = [h.get_text(" ", strip=True).lower() for h in table.select("thead th")]
        team_idx = _detect_column_index(headers, ("national", "team"), fallback=1)
        if team_idx is None:
            team_idx = _detect_column_index(headers, ("nationalmannschaft",), fallback=1)
        matches_idx = _detect_column_index(headers, ("matches",), fallback=None)
        if matches_idx is None:
            matches_idx = _detect_column_index(headers, ("spiele",), fallback=None)
        goals_idx = _detect_column_index(headers, ("goals",), fallback=None)
        if goals_idx is None:
            goals_idx = _detect_column_index(headers, ("tore",), fallback=None)
        if team_idx is None or matches_idx is None:
            continue

        for tr in table.select("tbody tr"):
            cells = tr.find_all("td")
            if not cells:
                continue
            values = [c.get_text(" ", strip=True) for c in cells]
            if team_idx >= len(values) or matches_idx >= len(values):
                continue
            team = values[team_idx].strip()
            if not team:
                continue
            matches = _parse_int(values[matches_idx])
            goals = _parse_int(values[goals_idx]) if goals_idx is not None and goals_idx < len(values) else 0
            rows.append(
                {
                    "team": team,
                    "matches": matches,
                    "goals": goals,
                    "is_youth": int(bool(re.search(r"\bU[- ]?\d{2}\b", team, flags=re.IGNORECASE))),
                }
            )

    if not rows:
        return rows
    out = pd.DataFrame(rows).drop_duplicates(subset=["team", "matches", "goals"], keep="first")
    return out.to_dict(orient="records")


def aggregate_national_rows(rows: Sequence[dict]) -> dict:
    if not rows:
        return {
            "senior_caps": pd.NA,
            "senior_goals": pd.NA,
            "youth_caps": pd.NA,
            "youth_goals": pd.NA,
        }

    frame = pd.DataFrame(rows)
    frame["matches"] = pd.to_numeric(frame["matches"], errors="coerce").fillna(0)
    frame["goals"] = pd.to_numeric(frame["goals"], errors="coerce").fillna(0)
    frame["is_youth"] = pd.to_numeric(frame["is_youth"], errors="coerce").fillna(0).astype(int)

    senior = frame[frame["is_youth"] == 0]
    youth = frame[frame["is_youth"] == 1]

    return {
        "senior_caps": int(senior["matches"].sum()) if not senior.empty else 0,
        "senior_goals": int(senior["goals"].sum()) if not senior.empty else 0,
        "youth_caps": int(youth["matches"].sum()) if not youth.empty else 0,
        "youth_goals": int(youth["goals"].sum()) if not youth.empty else 0,
    }


def build_player_national_caps(
    players_source: str,
    output: str,
    start_season: str | None = None,
    end_season: str | None = None,
    latest_only: bool = True,
    max_players: int | None = None,
    fetch_missing_profiles: bool = False,
    fetch_national_page: bool = False,
    sleep_seconds: float = 2.0,
    overwrite_cache: bool = False,
    include_failed: bool = True,
) -> None:
    players_df = load_player_index(Path(players_source))
    if max_players is not None and max_players > 0:
        players_df = players_df.head(max_players).copy()

    start_norm = normalize_season_label(start_season) if start_season else None
    end_norm = normalize_season_label(end_season) if end_season else None
    nt_cache_dir = Path("data/raw/tm/national_team")
    nt_cache_dir.mkdir(parents=True, exist_ok=True)

    out_rows: list[dict] = []
    total = len(players_df)
    for i, row in players_df.iterrows():
        tmid = str(row["transfermarkt_id"])
        seasons = select_target_seasons(row["target_seasons"], start_norm, end_norm, latest_only=latest_only)
        if not seasons:
            continue

        profile_link = str(row["profile_link"]) if row.get("profile_link") else build_profile_url_from_id(tmid)
        html_path = resolve_cached_profile_html(tmid, profile_link=profile_link)

        scrape_success = 1
        source_type = "cache_profile"
        profile_fields = {
            "caps_raw": pd.NA,
            "senior_caps": pd.NA,
            "senior_goals": pd.NA,
            "current_team": pd.NA,
        }
        detailed_rows: list[dict] = []
        nt_url = pd.NA

        try:
            if html_path is not None:
                body = html_path.read_bytes()
            elif fetch_missing_profiles:
                source_type = "fetched_profile"
                body = fetch(profile_link)
                cache_path = Path("data/raw/tm/players") / f"player_{tmid}.html"
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(body)
                time.sleep(max(sleep_seconds, 0.0))
            else:
                raise FileNotFoundError(f"no cached profile HTML for tmid={tmid}")
            profile_fields = parse_profile_national_fields(body)

            if fetch_national_page:
                nt_url = build_national_team_url(profile_link, tmid)
                nt_cache = nt_cache_dir / f"{tmid}.html"
                if overwrite_cache or (not nt_cache.exists()):
                    nt_body = fetch(str(nt_url))
                    nt_cache.write_bytes(nt_body)
                    time.sleep(max(sleep_seconds, 0.0))
                    source_type = "fetched_profile+national"
                else:
                    nt_body = nt_cache.read_bytes()
                detailed_rows = parse_national_team_rows(nt_body)
        except Exception as exc:
            scrape_success = 0
            print(f"[national] failed {tmid} ({row.get('name','')}): {exc}")

        if scrape_success == 0 and not include_failed:
            continue

        detailed = aggregate_national_rows(detailed_rows)
        senior_caps = detailed["senior_caps"]
        senior_goals = detailed["senior_goals"]
        youth_caps = detailed["youth_caps"]
        youth_goals = detailed["youth_goals"]

        # Fallback to profile header if detailed national page is unavailable.
        if pd.isna(senior_caps):
            senior_caps = profile_fields["senior_caps"]
        if pd.isna(senior_goals):
            senior_goals = profile_fields["senior_goals"]
        if pd.isna(youth_caps):
            youth_caps = 0 if not pd.isna(senior_caps) else pd.NA
        if pd.isna(youth_goals):
            youth_goals = 0 if not pd.isna(senior_goals) else pd.NA

        total_caps = pd.to_numeric(pd.Series([senior_caps]), errors="coerce").iloc[0]
        total_goals = pd.to_numeric(pd.Series([senior_goals]), errors="coerce").iloc[0]
        if pd.notna(youth_caps):
            total_caps = (0 if pd.isna(total_caps) else total_caps) + float(youth_caps)
        if pd.notna(youth_goals):
            total_goals = (0 if pd.isna(total_goals) else total_goals) + float(youth_goals)

        for season in seasons:
            out_rows.append(
                {
                    "player_id": row["player_id"],
                    "transfermarkt_id": tmid,
                    "name": row["name"],
                    "dob": row["dob"],
                    "season": season,
                    "senior_caps": senior_caps if scrape_success else pd.NA,
                    "senior_goals": senior_goals if scrape_success else pd.NA,
                    "youth_caps": youth_caps if scrape_success else pd.NA,
                    "youth_goals": youth_goals if scrape_success else pd.NA,
                    "total_caps": total_caps if scrape_success else pd.NA,
                    "total_goals": total_goals if scrape_success else pd.NA,
                    "is_full_international": int(pd.notna(senior_caps) and float(senior_caps) > 0)
                    if scrape_success
                    else pd.NA,
                    "current_national_team": profile_fields["current_team"] if scrape_success else pd.NA,
                    "caps_goals_raw": profile_fields["caps_raw"] if scrape_success else pd.NA,
                    "nt_source_url": nt_url if fetch_national_page else profile_link,
                    "nt_source_type": source_type,
                    "nt_scrape_success": scrape_success,
                    "nt_rows_parsed": len(detailed_rows),
                }
            )

        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"[national] processed {i+1:,}/{total:,} players")

    out_df = pd.DataFrame(out_rows)
    if out_df.empty:
        raise ValueError("No national-team rows produced.")

    out_df = out_df.sort_values(
        ["transfermarkt_id", "season", "nt_scrape_success"],
        ascending=[True, True, False],
    )
    out_df = out_df.drop_duplicates(subset=["transfermarkt_id", "season"], keep="first")

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    success_rate = float((out_df["nt_scrape_success"] == 1).mean() * 100.0)
    print(f"[national] wrote {len(out_df):,} rows -> {out_path}")
    print(f"[national] scrape success rate: {success_rate:,.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build season-level national-team cap features from Transfermarkt profiles."
    )
    parser.add_argument(
        "--players-source",
        default="data/model/big5_players.parquet",
        help="Input player table (CSV/Parquet) or directory with *_with_sofa.csv files.",
    )
    parser.add_argument("--output", default="data/external/national_team_caps.csv")
    parser.add_argument("--start-season", default=None, help="Optional season lower bound, e.g. 2019/20")
    parser.add_argument("--end-season", default=None, help="Optional season upper bound, e.g. 2024/25")
    parser.add_argument(
        "--all-seasons",
        action="store_true",
        help="Write values for all seasons in range (default: latest season only to reduce hindsight leakage).",
    )
    parser.add_argument("--max-players", type=int, default=None)
    parser.add_argument(
        "--fetch-missing-profiles",
        action="store_true",
        help="Fetch missing profile HTML if not cached in data/raw/tm/players.",
    )
    parser.add_argument(
        "--fetch-national-page",
        action="store_true",
        help="Also fetch /nationalmannschaft/ pages for separate youth/senior caps.",
    )
    parser.add_argument("--sleep", type=float, default=2.0, help="Sleep after web fetches.")
    parser.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="Refetch and overwrite cached national-team pages.",
    )
    parser.add_argument(
        "--no-failed-rows",
        action="store_true",
        help="Drop failed player-season rows instead of writing NaNs.",
    )
    args = parser.parse_args()

    build_player_national_caps(
        players_source=args.players_source,
        output=args.output,
        start_season=args.start_season,
        end_season=args.end_season,
        latest_only=not args.all_seasons,
        max_players=args.max_players,
        fetch_missing_profiles=args.fetch_missing_profiles,
        fetch_national_page=args.fetch_national_page,
        sleep_seconds=args.sleep,
        overwrite_cache=args.overwrite_cache,
        include_failed=not args.no_failed_rows,
    )


if __name__ == "__main__":
    main()
