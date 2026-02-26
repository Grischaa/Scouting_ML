from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence
from urllib.parse import urlparse, urlunparse

import pandas as pd
from bs4 import BeautifulSoup

from scouting_ml.tm.tm_scraper import fetch

TM_ID_RE = re.compile(r"/spieler/(\d+)")
DATE_DMY_RE = re.compile(r"(\d{2})/(\d{2})/(\d{4})")
MONEY_RE = re.compile(r"([€£$])\s*([0-9][0-9.,]*)\s*([A-Za-z]+)?")
INT_RE = re.compile(r"(\d+)")
TM_TRANSFER_COMPONENT_MARKER = b"<tm-player-transfer-history"

# Coarse FX fallback for non-EUR fees so we can keep one numeric scale.
FX_TO_EUR = {
    "€": 1.00,
    "£": 1.17,
    "$": 0.92,
}


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


def build_transfers_url(profile_link: str, transfermarkt_id: str) -> str:
    parsed = urlparse(profile_link)
    path = parsed.path or ""
    if "/profil/spieler/" in path:
        transfer_path = path.replace("/profil/spieler/", "/transfers/spieler/")
    else:
        transfer_path = f"/-/transfers/spieler/{transfermarkt_id}"
    return urlunparse(
        (
            parsed.scheme or "https",
            parsed.netloc or "www.transfermarkt.com",
            transfer_path,
            "",
            "",
            "",
        )
    )


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
    fx = FX_TO_EUR.get(symbol, 1.0)
    return base * mult * fx


def parse_date_to_season(date_text: str) -> str | None:
    if not date_text:
        return None
    m = DATE_DMY_RE.search(date_text)
    if not m:
        return None
    day = int(m.group(1))
    month = int(m.group(2))
    year = int(m.group(3))
    if not (1 <= day <= 31 and 1 <= month <= 12):
        return None
    start_year = year if month >= 7 else year - 1
    return f"{start_year}/{str(start_year + 1)[-2:]}"


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


def _parse_transfer_row(values: Sequence[str], season_idx: int, date_idx: int | None, fee_idx: int | None) -> dict | None:
    if season_idx >= len(values):
        return None
    season_text = values[season_idx]
    date_text = values[date_idx] if date_idx is not None and date_idx < len(values) else ""
    fee_text = values[fee_idx] if fee_idx is not None and fee_idx < len(values) else (values[-1] if values else "")

    season = normalize_season_label(season_text) or parse_date_to_season(date_text)
    if not season:
        return None

    fee_text_norm = fee_text.strip()
    fee_eur = parse_money_to_eur(fee_text_norm)
    fee_low = fee_text_norm.lower()
    is_loan = int("loan" in fee_low or "leihe" in fee_low)
    is_free = int("free" in fee_low or "ablösefrei" in fee_low or fee_text_norm in {"-", "—"})

    return {
        "season": season,
        "date": date_text.strip(),
        "fee_text": fee_text_norm,
        "fee_eur": fee_eur,
        "is_loan": is_loan,
        "is_free": is_free,
    }


def _extract_transfer_fee_text(raw: object) -> str:
    if raw is None:
        return ""
    if isinstance(raw, dict):
        for key in ("displayValue", "value", "formatted", "text", "caption", "label"):
            value = raw.get(key)
            if value is not None:
                return str(value).strip()
        return ""
    if isinstance(raw, (int, float)):
        if pd.isna(raw):
            return ""
        return f"€{float(raw):,.0f}"
    return str(raw).strip()


def _extract_transfer_bool(record: dict, keys: Sequence[str]) -> int:
    for key in keys:
        if key not in record:
            continue
        value = record.get(key)
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)) and not pd.isna(value):
            return int(float(value) != 0.0)
        if isinstance(value, str):
            low = value.strip().casefold()
            if low in {"1", "true", "yes", "y", "loan"}:
                return 1
            if low in {"0", "false", "no", "n"}:
                return 0
    return 0


def _iter_transfer_records(payload: object) -> Iterable[dict]:
    if isinstance(payload, list):
        for item in payload:
            yield from _iter_transfer_records(item)
        return

    if not isinstance(payload, dict):
        return

    looks_like_record = any(
        key in payload
        for key in (
            "season",
            "seasonName",
            "transferSeason",
            "date",
            "transferDate",
            "fee",
            "transferFee",
        )
    )
    if looks_like_record:
        yield payload

    for key in (
        "items",
        "transfers",
        "transferHistory",
        "history",
        "rows",
        "data",
        "list",
        "career",
        "upcomingTransfers",
    ):
        value = payload.get(key)
        if isinstance(value, (list, dict)):
            yield from _iter_transfer_records(value)

    for value in payload.values():
        if isinstance(value, (list, dict)):
            yield from _iter_transfer_records(value)


def _parse_transfer_rows_from_json_payload(payload: object) -> list[dict]:
    out: list[dict] = []
    for record in _iter_transfer_records(payload):
        season_text = (
            record.get("season")
            or record.get("seasonName")
            or record.get("transferSeason")
            or record.get("season_label")
        )
        date_text = (
            record.get("date")
            or record.get("transferDate")
            or record.get("dateFormatted")
            or record.get("transfer_date")
            or ""
        )
        fee_text = _extract_transfer_fee_text(
            record.get("transferFee")
            or record.get("fee")
            or record.get("feeText")
            or record.get("transfer_fee")
        )
        fee_eur = parse_money_to_eur(fee_text)

        if fee_eur is None:
            numeric_fee = (
                record.get("feeEur")
                or record.get("fee_eur")
                or record.get("feeAmount")
                or record.get("transferFeeEur")
            )
            if isinstance(numeric_fee, (int, float)) and not pd.isna(numeric_fee):
                fee_eur = float(numeric_fee)

        season = normalize_season_label(str(season_text)) if season_text is not None else None
        if not season:
            season = parse_date_to_season(str(date_text))
        if not season:
            continue

        low_fee = fee_text.casefold()
        is_loan = _extract_transfer_bool(record, ("isLoan", "loan", "loanMove"))
        is_free = _extract_transfer_bool(record, ("isFreeTransfer", "freeTransfer", "free_move"))
        if not is_loan:
            is_loan = int("loan" in low_fee or "leihe" in low_fee)
        if not is_free:
            is_free = int("free" in low_fee or "ablösefrei" in low_fee or fee_text in {"-", "—"})

        out.append(
            {
                "season": season,
                "date": str(date_text).strip(),
                "fee_text": fee_text,
                "fee_eur": fee_eur,
                "is_loan": is_loan,
                "is_free": is_free,
            }
        )

    if not out:
        return []
    dedup = pd.DataFrame(out).drop_duplicates(subset=["season", "date", "fee_text"], keep="first")
    return dedup.to_dict(orient="records")


def _has_dynamic_transfer_component(html: bytes) -> bool:
    return TM_TRANSFER_COMPONENT_MARKER in html.lower()


def _build_dynamic_fallback_urls(transfer_url: str, transfermarkt_id: str) -> list[str]:
    parsed = urlparse(transfer_url)
    scheme = parsed.scheme or "https"
    host = parsed.netloc or "www.transfermarkt.com"
    base = f"{scheme}://{host}"
    path = parsed.path.rstrip("/")

    raw = [
        f"{base}{path}/plus/1",
        f"{base}{path}/ajax/yw1",
        f"{base}{path}?ajax=yw1",
        f"{base}/ceapi/transfer-history/{transfermarkt_id}",
        f"{base}/ceapi/transferHistory/{transfermarkt_id}",
    ]
    out: list[str] = []
    seen: set[str] = set()
    for url in raw:
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out


def parse_transfer_rows(html: bytes) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    out: list[dict] = []

    for table in soup.select("table.items"):
        header_cells = table.select("thead th")
        headers = [c.get_text(" ", strip=True).lower() for c in header_cells]

        season_idx = _detect_column_index(headers, ("season",), fallback=0)
        if season_idx is None:
            season_idx = _detect_column_index(headers, ("saison",), fallback=0)
        if season_idx is None:
            continue

        date_idx = _detect_column_index(headers, ("date",), fallback=None)
        if date_idx is None:
            date_idx = _detect_column_index(headers, ("datum",), fallback=None)
        fee_idx = _detect_column_index(headers, ("fee",), fallback=None)
        if fee_idx is None:
            fee_idx = _detect_column_index(headers, ("ablöse",), fallback=None)

        for row in table.select("tbody tr"):
            cells = row.find_all("td")
            if len(cells) < 2:
                continue
            values = [c.get_text(" ", strip=True) for c in cells]
            parsed = _parse_transfer_row(values, season_idx=season_idx, date_idx=date_idx, fee_idx=fee_idx)
            if parsed is None:
                continue
            out.append(parsed)

    if out:
        dedup = pd.DataFrame(out).drop_duplicates(subset=["season", "date", "fee_text"], keep="first")
        return dedup.to_dict(orient="records")

    for script in soup.select("script[type='application/json']"):
        raw = script.get_text(strip=True)
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        rows = _parse_transfer_rows_from_json_payload(payload)
        if rows:
            return rows

    transfer_component = soup.select_one("tm-player-transfer-history")
    if transfer_component is not None:
        for attr_name, attr_value in transfer_component.attrs.items():
            if attr_value is None:
                continue
            key = str(attr_name).casefold()
            if "transfer" not in key and "history" not in key and "data" not in key:
                continue
            raw_attr = str(attr_value).strip()
            if not raw_attr:
                continue
            try:
                payload = json.loads(raw_attr)
            except Exception:
                continue
            rows = _parse_transfer_rows_from_json_payload(payload)
            if rows:
                return rows

    raw_html = html.strip()
    if raw_html.startswith(b"{") or raw_html.startswith(b"["):
        try:
            payload = json.loads(raw_html.decode("utf-8", errors="ignore"))
        except Exception:
            payload = None
        if payload is not None:
            rows = _parse_transfer_rows_from_json_payload(payload)
            if rows:
                return rows

    return []


def _rows_to_frame(rows: Iterable[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    expected_cols = [
        "season",
        "date",
        "fee_text",
        "fee_eur",
        "is_loan",
        "is_free",
    ]
    for col in expected_cols:
        if col not in frame.columns:
            frame[col] = pd.NA
    if frame.empty:
        return frame[expected_cols]
    frame["season"] = frame["season"].apply(normalize_season_label)
    frame["season_start_year"] = frame["season"].apply(season_start_year)
    frame["fee_eur"] = pd.to_numeric(frame["fee_eur"], errors="coerce")
    frame["is_loan"] = pd.to_numeric(frame["is_loan"], errors="coerce").fillna(0).astype(int)
    frame["is_free"] = pd.to_numeric(frame["is_free"], errors="coerce").fillna(0).astype(int)
    frame["date_dt"] = pd.to_datetime(frame["date"], format="%d/%m/%Y", errors="coerce", dayfirst=True)
    frame = frame.sort_values(["season_start_year", "date_dt"], ascending=[True, True], na_position="last")
    return frame.reset_index(drop=True)


def aggregate_transfer_features(rows: Iterable[dict], target_seasons: Sequence[str]) -> Dict[str, dict]:
    frame = _rows_to_frame(rows)
    out: Dict[str, dict] = {}
    for season in target_seasons:
        season_norm = normalize_season_label(season)
        start_year = season_start_year(season_norm)
        if season_norm is None or start_year is None:
            continue

        if frame.empty:
            hist = frame
            recent = frame
        else:
            hist = frame[frame["season_start_year"] <= start_year]
            recent = hist[hist["season_start_year"] >= (start_year - 2)]

        paid_recent = pd.to_numeric(recent["fee_eur"], errors="coerce")
        paid_recent = paid_recent[paid_recent > 0]
        paid_hist = pd.to_numeric(hist["fee_eur"], errors="coerce")
        paid_hist = paid_hist[paid_hist > 0]

        if hist.empty:
            last_transfer_season = pd.NA
            last_transfer_fee_text = pd.NA
            last_transfer_fee_eur = pd.NA
            last_transfer_is_loan = pd.NA
        else:
            last = hist.iloc[-1]
            last_transfer_season = last.get("season", pd.NA)
            last_transfer_fee_text = last.get("fee_text", pd.NA)
            last_transfer_fee_eur = last.get("fee_eur", pd.NA)
            last_transfer_is_loan = int(last.get("is_loan", 0))

        out[season_norm] = {
            "last_transfer_season": last_transfer_season,
            "last_transfer_fee_text": last_transfer_fee_text,
            "last_transfer_fee_eur": last_transfer_fee_eur,
            "last_transfer_is_loan": last_transfer_is_loan,
            "transfer_count_career_to_date": int(len(hist)),
            "transfer_count_3y": int(len(recent)),
            "transfer_loans_3y": int(recent["is_loan"].sum()) if not recent.empty else 0,
            "transfer_free_moves_3y": int(recent["is_free"].sum()) if not recent.empty else 0,
            "transfer_paid_moves_3y": int((pd.to_numeric(recent["fee_eur"], errors="coerce") > 0).sum())
            if not recent.empty
            else 0,
            "transfer_total_fees_3y_eur": float(paid_recent.sum()) if not paid_recent.empty else 0.0,
            "transfer_avg_fee_3y_eur": float(paid_recent.mean()) if not paid_recent.empty else pd.NA,
            "transfer_max_fee_career_eur": float(paid_hist.max()) if not paid_hist.empty else pd.NA,
        }
    return out


def _resolve_cache_path(tmid: str, profile_link: str) -> Path:
    cache_dir = Path("data/raw/tm/transfers")
    cache_dir.mkdir(parents=True, exist_ok=True)

    numeric_id = _numeric_id_from_key(tmid) or extract_transfermarkt_id(profile_link)
    slug = _slug_from_profile_link(profile_link)
    if slug and numeric_id:
        return cache_dir / f"{slug}_{numeric_id}.html"
    if numeric_id:
        return cache_dir / f"player_{numeric_id}.html"
    return cache_dir / f"player_{tmid}.html"


def build_player_transfers(
    players_source: str,
    output: str,
    start_season: str | None = None,
    end_season: str | None = None,
    max_players: int | None = None,
    fetch_missing: bool = True,
    sleep_seconds: float = 2.5,
    overwrite_cache: bool = False,
    include_failed: bool = True,
    dynamic_fallback: bool = False,
    max_dynamic_fallback_attempts: int = 2,
) -> None:
    players_df = load_player_index(Path(players_source))
    if max_players is not None and max_players > 0:
        players_df = players_df.head(max_players).copy()

    start_norm = normalize_season_label(start_season) if start_season else None
    end_norm = normalize_season_label(end_season) if end_season else None

    out_rows: list[dict] = []
    total = len(players_df)
    dynamic_component_rows = 0
    dynamic_fallback_hits = 0

    for idx, row in players_df.iterrows():
        tmid = str(row["transfermarkt_id"])
        profile_link = str(row["profile_link"]) if row.get("profile_link") else ""
        target_seasons = [s for s in row["target_seasons"] if in_season_range(s, start_norm, end_norm)]
        if not target_seasons:
            continue

        transfer_url = build_transfers_url(profile_link, tmid)
        cache_path = _resolve_cache_path(tmid, profile_link)

        scrape_success = 1
        source_type = "cache"
        parser_mode = "legacy_table"
        transfer_rows: list[dict] = []
        try:
            if overwrite_cache or (not cache_path.exists()):
                if not fetch_missing:
                    raise FileNotFoundError(f"no cached transfer HTML for tmid={tmid}")
                source_type = "fetched"
                body = fetch(transfer_url)
                cache_path.write_bytes(body)
                time.sleep(max(sleep_seconds, 0.0))
            else:
                body = cache_path.read_bytes()
            transfer_rows = parse_transfer_rows(body)
            dynamic_page = _has_dynamic_transfer_component(body)
            if dynamic_page:
                dynamic_component_rows += 1
            if dynamic_page and not transfer_rows:
                parser_mode = "dynamic_component_no_rows"
            if transfer_rows and dynamic_page:
                parser_mode = "dynamic_component_parsed"

            if dynamic_fallback and dynamic_page and not transfer_rows:
                fallback_urls = _build_dynamic_fallback_urls(transfer_url=transfer_url, transfermarkt_id=tmid)
                for fallback_idx, fallback_url in enumerate(fallback_urls):
                    if fallback_idx >= max(max_dynamic_fallback_attempts, 0):
                        break
                    try:
                        fallback_body = fetch(fallback_url)
                        transfer_rows = parse_transfer_rows(fallback_body)
                        if transfer_rows:
                            parser_mode = "dynamic_fallback_parsed"
                            source_type = "dynamic_fallback"
                            dynamic_fallback_hits += 1
                            break
                    except Exception:
                        continue
        except Exception as exc:
            scrape_success = 0
            parser_mode = "error"
            print(f"[transfers] failed {tmid} ({row.get('name','')}): {exc}")

        if scrape_success == 0 and not include_failed:
            continue

        aggregated = aggregate_transfer_features(transfer_rows, target_seasons=target_seasons)
        for season in target_seasons:
            features = aggregated.get(normalize_season_label(season), {})
            base = {
                "player_id": row["player_id"],
                "transfermarkt_id": tmid,
                "name": row["name"],
                "dob": row["dob"],
                "season": season,
                "transfer_source_url": transfer_url,
                "transfer_source_type": source_type,
                "transfer_scrape_success": scrape_success,
                "transfer_rows_parsed": len(transfer_rows),
                "transfer_parser_mode": parser_mode,
            }
            if scrape_success == 0:
                base.update(
                    {
                        "last_transfer_season": pd.NA,
                        "last_transfer_fee_text": pd.NA,
                        "last_transfer_fee_eur": pd.NA,
                        "last_transfer_is_loan": pd.NA,
                        "transfer_count_career_to_date": pd.NA,
                        "transfer_count_3y": pd.NA,
                        "transfer_loans_3y": pd.NA,
                        "transfer_free_moves_3y": pd.NA,
                        "transfer_paid_moves_3y": pd.NA,
                        "transfer_total_fees_3y_eur": pd.NA,
                        "transfer_avg_fee_3y_eur": pd.NA,
                        "transfer_max_fee_career_eur": pd.NA,
                    }
                )
            else:
                base.update(features)
            out_rows.append(base)

        if (idx + 1) % 100 == 0 or (idx + 1) == total:
            print(f"[transfers] processed {idx+1:,}/{total:,} players")

    out_df = pd.DataFrame(out_rows)
    if out_df.empty:
        raise ValueError("No transfer rows produced.")

    out_df = out_df.sort_values(
        ["transfermarkt_id", "season", "transfer_scrape_success"],
        ascending=[True, True, False],
    )
    out_df = out_df.drop_duplicates(subset=["transfermarkt_id", "season"], keep="first")

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    success_rate = float((out_df["transfer_scrape_success"] == 1).mean() * 100.0)
    print(f"[transfers] wrote {len(out_df):,} rows -> {out_path}")
    print(f"[transfers] scrape success rate: {success_rate:,.1f}%")
    if dynamic_component_rows > 0:
        print(
            "[transfers] dynamic transfer component detected for "
            f"{dynamic_component_rows:,} players"
        )
    if dynamic_fallback:
        print(f"[transfers] dynamic fallback hits: {dynamic_fallback_hits:,}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build season-level player transfer features from Transfermarkt transfer history pages."
    )
    parser.add_argument(
        "--players-source",
        default="data/model/big5_players.parquet",
        help="Input player table (CSV/Parquet) or directory with *_with_sofa.csv files.",
    )
    parser.add_argument("--output", default="data/external/player_transfers.csv")
    parser.add_argument("--start-season", default=None, help="Optional season lower bound, e.g. 2019/20")
    parser.add_argument("--end-season", default=None, help="Optional season upper bound, e.g. 2024/25")
    parser.add_argument("--max-players", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=2.5, help="Sleep after fetching missing pages.")
    parser.add_argument(
        "--no-fetch-missing",
        action="store_true",
        help="Only use cached transfer pages from data/raw/tm/transfers.",
    )
    parser.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="Refetch and overwrite cached transfer history pages.",
    )
    parser.add_argument(
        "--no-failed-rows",
        action="store_true",
        help="Drop failed player-season rows instead of writing NaNs.",
    )
    parser.add_argument(
        "--dynamic-fallback",
        action="store_true",
        help="Try dynamic/JSON fallback URLs when Transfermarkt serves JS-only transfer history.",
    )
    parser.add_argument(
        "--max-dynamic-fallback-attempts",
        type=int,
        default=2,
        help="Max number of fallback URLs to try per player when --dynamic-fallback is enabled.",
    )
    args = parser.parse_args()

    build_player_transfers(
        players_source=args.players_source,
        output=args.output,
        start_season=args.start_season,
        end_season=args.end_season,
        max_players=args.max_players,
        fetch_missing=not args.no_fetch_missing,
        sleep_seconds=args.sleep,
        overwrite_cache=args.overwrite_cache,
        include_failed=not args.no_failed_rows,
        dynamic_fallback=args.dynamic_fallback,
        max_dynamic_fallback_attempts=args.max_dynamic_fallback_attempts,
    )


if __name__ == "__main__":
    main()
