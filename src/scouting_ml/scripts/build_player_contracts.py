from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
from bs4 import BeautifulSoup

from scouting_ml.tm.tm_scraper import fetch

TM_ID_RE = re.compile(r"/spieler/(\d+)")
DATE_DMY_RE = re.compile(r"(\d{2})/(\d{2})/(\d{4})")
MONEY_RE = re.compile(r"([€£$])\s*([0-9][0-9.,]*)\s*([A-Za-z]+)?")


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


def build_profile_url_from_id(tm_key: str) -> str:
    numeric_id = _numeric_id_from_key(tm_key)
    if numeric_id:
        return f"https://www.transfermarkt.com/-/profil/spieler/{numeric_id}"
    return f"https://www.transfermarkt.com/-/profil/spieler/{tm_key}"


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


def _normalize_number_string(num_text: str) -> float | None:
    txt = num_text.strip().replace(" ", "")
    if not txt:
        return None
    if "," in txt and "." in txt:
        if txt.rfind(".") > txt.rfind(","):
            txt = txt.replace(",", "")
        else:
            txt = txt.replace(".", "").replace(",", ".")
    elif "," in txt:
        txt = txt.replace(",", ".")
    try:
        return float(txt)
    except Exception:
        return None


def parse_money_to_eur(value: str) -> float | None:
    if not value:
        return None
    m = MONEY_RE.search(value)
    if not m:
        return None
    symbol, num_text, suffix = m.groups()
    if symbol != "€":
        return None
    base = _normalize_number_string(num_text)
    if base is None:
        return None
    mult = 1.0
    s = (suffix or "").lower()
    if s.startswith(("k", "tsd", "th")):
        mult = 1_000.0
    elif s.startswith(("m", "mio", "mill")):
        mult = 1_000_000.0
    elif s.startswith(("b", "bn")):
        mult = 1_000_000_000.0
    return base * mult


def parse_contract_year(date_text: str) -> int | None:
    if not date_text:
        return None
    m = DATE_DMY_RE.search(date_text)
    if not m:
        return None
    return int(m.group(3))


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


def parse_contract_fields(html: bytes) -> dict:
    soup = BeautifulSoup(html, "lxml")
    pairs = extract_label_pairs(soup)

    def find_value(label_variants: Sequence[str]) -> str:
        for key, value in pairs.items():
            for variant in label_variants:
                if variant in key:
                    return value
        return ""

    contract_until = find_value(["contract expires", "vertrag bis"])
    release_clause = find_value(["release clause", "ausstiegsklausel"])
    agent_name = find_value(["player agent", "agent", "berater"])
    loan_from = find_value(["on loan from", "leihe von"])

    return {
        "contract_until": contract_until if contract_until not in {"-", "—"} else "",
        "contract_until_year": parse_contract_year(contract_until),
        "release_clause": release_clause,
        "release_clause_eur": parse_money_to_eur(release_clause),
        "agent_name": agent_name,
        "loan_flag": int(bool(loan_from)),
    }


def resolve_cached_player_html(tm_key: str, profile_link: str | None = None) -> Path | None:
    players_dir = Path("data/raw/tm/players")

    # 1) Exact filename match (works for keys like aaron_hickey_591949)
    exact = players_dir / f"{tm_key}.html"
    if exact.exists():
        return exact

    # 2) Try slug + numeric id from profile URL
    slug = _slug_from_profile_link(profile_link)
    numeric_id = _numeric_id_from_key(tm_key) or extract_transfermarkt_id(profile_link)
    if slug and numeric_id:
        slug_file = players_dir / f"{slug}_{numeric_id}.html"
        if slug_file.exists():
            return slug_file

    # 3) Fallback glob by numeric id suffix
    if numeric_id:
        matches = sorted(players_dir.glob(f"*_{numeric_id}.html"))
        if matches:
            return matches[0]

    # 4) Last fallback glob containing key
    matches = sorted(players_dir.glob(f"*{tm_key}*.html"))
    if matches:
        return matches[0]
    return None


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


def build_player_contracts(
    players_source: str,
    output: str,
    start_season: str | None = None,
    end_season: str | None = None,
    latest_only: bool = True,
    max_players: int | None = None,
    fetch_missing: bool = False,
    sleep_seconds: float = 2.0,
) -> None:
    players_df = load_player_index(Path(players_source))
    if max_players is not None and max_players > 0:
        players_df = players_df.head(max_players).copy()

    start_norm = normalize_season_label(start_season) if start_season else None
    end_norm = normalize_season_label(end_season) if end_season else None

    out_rows: list[dict] = []
    total = len(players_df)
    for i, row in players_df.iterrows():
        tmid = str(row["transfermarkt_id"])
        seasons = select_target_seasons(row["target_seasons"], start_norm, end_norm, latest_only=latest_only)
        if not seasons:
            continue

        html_path = resolve_cached_player_html(tmid, profile_link=str(row.get("profile_link", "")))
        profile_link = str(row["profile_link"]) if row.get("profile_link") else build_profile_url_from_id(tmid)
        scrape_success = 1
        parsed = {
            "contract_until": "",
            "contract_until_year": None,
            "release_clause": "",
            "release_clause_eur": None,
            "agent_name": "",
            "loan_flag": 0,
        }
        source_type = "cache"

        try:
            if html_path is not None:
                body = html_path.read_bytes()
            elif fetch_missing:
                source_type = "fetched"
                body = fetch(profile_link)
                cache_path = Path("data/raw/tm/players") / f"player_{tmid}.html"
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(body)
                time.sleep(max(sleep_seconds, 0.0))
            else:
                raise FileNotFoundError(f"no cached profile HTML for tmid={tmid}")
            parsed = parse_contract_fields(body)
        except Exception as exc:
            scrape_success = 0
            print(f"[contracts] failed {tmid} ({row.get('name','')}): {exc}")

        for season in seasons:
            out_rows.append(
                {
                    "player_id": row["player_id"],
                    "transfermarkt_id": tmid,
                    "name": row["name"],
                    "dob": row["dob"],
                    "season": season,
                    "contract_until": parsed["contract_until"] if scrape_success else pd.NA,
                    "contract_until_year": parsed["contract_until_year"] if scrape_success else pd.NA,
                    "release_clause": parsed["release_clause"] if scrape_success else pd.NA,
                    "release_clause_eur": parsed["release_clause_eur"] if scrape_success else pd.NA,
                    "agent_name": parsed["agent_name"] if scrape_success else pd.NA,
                    "loan_flag": parsed["loan_flag"] if scrape_success else pd.NA,
                    "contract_source_url": profile_link,
                    "contract_source_type": source_type,
                    "contract_scrape_success": scrape_success,
                }
            )

        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"[contracts] processed {i+1:,}/{total:,} players")

    out_df = pd.DataFrame(out_rows)
    if out_df.empty:
        raise ValueError("No contract rows produced.")

    out_df = out_df.sort_values(
        ["transfermarkt_id", "season", "contract_scrape_success"],
        ascending=[True, True, False],
    )
    out_df = out_df.drop_duplicates(subset=["transfermarkt_id", "season"], keep="first")

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    success_rate = float((out_df["contract_scrape_success"] == 1).mean() * 100.0)
    print(f"[contracts] wrote {len(out_df):,} rows -> {out_path}")
    print(f"[contracts] scrape success rate: {success_rate:,.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build season-level player contract features from cached Transfermarkt profile HTML."
    )
    parser.add_argument(
        "--players-source",
        default="data/model/big5_players.parquet",
        help="Input player table (CSV/Parquet) or directory with *_with_sofa.csv files.",
    )
    parser.add_argument("--output", default="data/external/player_contracts.csv")
    parser.add_argument("--start-season", default=None, help="Optional season lower bound, e.g. 2019/20")
    parser.add_argument("--end-season", default=None, help="Optional season upper bound, e.g. 2024/25")
    parser.add_argument(
        "--all-seasons",
        action="store_true",
        help="Write contract values for all seasons in range (default: latest season only to reduce hindsight leakage).",
    )
    parser.add_argument("--max-players", type=int, default=None)
    parser.add_argument(
        "--fetch-missing",
        action="store_true",
        help="Fetch missing profile HTML if not present in data/raw/tm/players.",
    )
    parser.add_argument("--sleep", type=float, default=2.0, help="Sleep after fetching missing profiles.")
    args = parser.parse_args()

    build_player_contracts(
        players_source=args.players_source,
        output=args.output,
        start_season=args.start_season,
        end_season=args.end_season,
        latest_only=not args.all_seasons,
        max_players=args.max_players,
        fetch_missing=args.fetch_missing,
        sleep_seconds=args.sleep,
    )


if __name__ == "__main__":
    main()
