# src/scouting_ml/pipeline_tm.py

from __future__ import annotations
from pathlib import Path
import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup  # add at top with other imports
import pandas as pd
import unicodedata

from scouting_ml.tm_scraper import fetch
from scouting_ml.paths import ensure_dirs, tm_html


def _parse_team_html(html_path: Path) -> pd.DataFrame:
    """Parse a Transfermarkt *team* page HTML into a DataFrame."""
    try:
        from scouting_ml.tm_parser import parse_tm_team  # preferred
        return parse_tm_team(html_path)
    except (ImportError, AttributeError):
        pass

    try:
        from scouting_ml.tm_parser import parse_tm_search  # fallback if compatible
        df = parse_tm_search(html_path)
        if df is None or (hasattr(df, "empty") and df.empty):
            raise ValueError(
                "parse_tm_search returned no rows for a team page. "
                "Please implement parse_tm_team(html_path) in scouting_ml.tm_parser."
            )
        return df
    except (ImportError, AttributeError):
        raise RuntimeError(
            "No suitable parser found. Implement parse_tm_team(html_path) in scouting_ml.tm_parser."
        )


# ---------- Cleaning helpers ----------

def _parse_market_value(s: str) -> float | None:
    if not isinstance(s, str):
        return None
    s = s.strip().lower().replace("€", "").replace(" ", "")
    if s in {"—", "-", ""}:
        return None
    mult = 1.0
    if s.endswith("m"):
        mult, s = 1_000_000, s[:-1]
    elif s.endswith("k"):
        mult, s = 1_000, s[:-1]
    try:
        return float(s.replace(",", ".")) * mult
    except ValueError:
        return None


def _split_dob_age(s: str) -> tuple[pd.Timestamp | None, float | None]:
    if not isinstance(s, str):
        return None, None
    m = re.match(r"(\d{2}/\d{2}/\d{4})\s*\((\d+)\)", s.strip())
    if not m:
        return None, None
    dob = pd.to_datetime(m.group(1), dayfirst=True, errors="coerce")
    age = float(m.group(2))
    return dob, age


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # dob/age
    if "dob_age" in df.columns:
        df[["dob", "age"]] = df["dob_age"].apply(lambda x: pd.Series(_split_dob_age(x)))
    # market value
    if "market_value" in df.columns:
        df["market_value_eur"] = df["market_value"].apply(_parse_market_value)
    # stable player_id from profile link if present
    if "link" in df.columns:
        id_part = df["link"].str.extract(r"/spieler/(\d+)")
        name_part = df["link"].str.extract(r"/([^/]+)/profil/")
        df["player_id"] = (
            name_part[0].fillna("").str.replace(r"[^a-z0-9_]+", "_", regex=True).str.strip("_")
            + "_"
            + id_part[0].fillna("")
        ).str.strip("_")
        df.loc[df["player_id"].eq(""), "player_id"] = id_part[0]
    # Column order
    preferred = [
        "player_id", "name", "position", "nationality",
        "dob", "age", "market_value_eur", "market_value", "link", "type"
    ]
    ordered = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[ordered]


# ---------- Position enrichment ----------

def _players_cache_path(player_id: str) -> Path:
    base = Path("data/raw/tm/players")
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{player_id}.html"


# --- replace the whole _extract_profile_facts with this version ---
def _extract_profile_facts(html: str) -> dict:
    """
    Parse a Transfermarkt player profile using BeautifulSoup.
    We look in two places:
      A) 'Detailposition' card  -> Hauptposition / Nebenposition
      B) 'Daten und Fakten'     -> Position:, Größe/Height:, Fuß/Foot:
    Returns keys: position_main, position_alt, height_cm, foot
    """
    if not html:
        return {}

    soup = BeautifulSoup(html, "lxml")

    def norm(s: str | None) -> str | None:
        if not s:
            return None
        s = unicodedata.normalize("NFKC", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s or None

    facts = {"position_main": None, "position_alt": None, "height_cm": None, "foot": None}

    # --- A) Detailposition card (most reliable for positions)
    # Headline usually contains 'Detailposition'/'Detailed position'
    detail = None
    for h in soup.find_all(["h2", "h3", "span", "div"]):
        t = (h.get_text(" ", strip=True) or "").lower()
        if "detailposition" in t or "detailed position" in t:
            # find the nearest container with labels/values inside
            detail = h.find_parent(class_=re.compile(r"content-box|box")) or h.parent
            break

    if detail:
        # labels look like: "Hauptposition:" / "Main position:"
        def grab_detail(lbl_pat: str) -> str | None:
            lab = detail.find(text=re.compile(lbl_pat, re.I))
            if not lab:
                return None
            # value usually in the next sibling span or same row
            row = lab.parent
            # search rightward siblings first
            for sib in row.find_all(["span", "div"], recursive=False):
                if sib is row:  # safety
                    continue
                val = norm(sib.get_text(" ", strip=True))
                if val and not val.endswith(":"):
                    return val
            # fallback – any following text node in row
            val = norm(row.get_text(" ", strip=True).split(":", 1)[-1])
            return val

        facts["position_main"] = norm(
            grab_detail(r"(Hauptposition|Main position)")
        ) or facts["position_main"]
        facts["position_alt"] = norm(
            grab_detail(r"(Nebenposition|Other position)")
        ) or facts["position_alt"]

    # --- B) Daten & Fakten table
    # In the info-table, labels have class ...--regular and values ...--bold
    # We iterate pairs and build a dict of label -> value.
    info_pairs = {}
    for box in soup.find_all("div", class_=re.compile(r"info-table")):
        regs = box.find_all("span", class_=re.compile(r"info-table__content--regular"))
        bolds = box.find_all("span", class_=re.compile(r"info-table__content--bold"))
        for lab, val in zip(regs, bolds):
            L = norm(lab.get_text(" ", strip=True))
            V = norm(val.get_text(" ", strip=True))
            if L and V:
                info_pairs[L.rstrip(":")] = V

    # Map DE/EN labels
    def pick(*keys):
        for k in keys:
            if k in info_pairs:
                return info_pairs[k]
        return None

    pos_line = pick("Position", "Positionen", "Position:", "Positionen:", "Positionen:", "Position:")  # TM varies
    height_line = pick("Größe", "Height")
    foot_line = pick("Fuß", "Foot")

    # Height -> cm
    if height_line:
        m = re.search(r"(\d+[.,]?\d*)\s*m", height_line)
        if m:
            facts["height_cm"] = int(round(float(m.group(1).replace(",", ".")) * 100))

    # Foot: just keep raw (e.g. "rechts", "links", "right", "left")
    if foot_line:
        facts["foot"] = foot_line

    # Position from info table (often "Mittelfeld - Defensives Mittelfeld")
    if pos_line and not facts["position_main"]:
        # take the most specific part (rightmost after '-')
        main = norm(pos_line.split("-")[-1])
        facts["position_main"] = main

    # Clean “Facts and data …” noise if it sneaks in
    for key in ("position_main", "position_alt"):
        val = facts.get(key)
        if val and "facts and data" in val.lower():
            facts[key] = None

    return facts





def _fetch_or_load_profile(link: str, player_id: str, sleep_s: float = 0.7, force: bool = False) -> str:
    """
    Load a player's profile HTML from cache if valid; otherwise fetch, save, and return.
    Detects cookie/consent or JS-stub pages and refetches them.
    """
    cache_path = _players_cache_path(player_id)

    # 1) Try cache
    text: str | None = None
    if not force and cache_path.exists() and cache_path.stat().st_size > 0:
        text = cache_path.read_text(encoding="utf-8", errors="ignore")
        low = text.lower()
        if ("consent" in low and "cookie" in low) or ("enable javascript" in low):
            # invalidate cached consent page; force a refetch below
            text = None

    # 2) Fetch if needed
    if text is None:
        time.sleep(sleep_s)  # be polite
        html_bytes = fetch(link)  # your fetch() should have proper headers/retries
        text = html_bytes.decode("utf-8", errors="ignore")
        cache_path.write_text(text, encoding="utf-8", errors="ignore")

    return text

    
def _clean_team_position(name: str, raw: str) -> str | None:
    """
    Given name and a raw 'position' string (which may contain the name),
    return a clean football position like 'Goalkeeper', 'Centre-Back', etc.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    s = raw.strip()

    # If the position text starts with the name, drop it
    if isinstance(name, str) and name and s.lower().startswith(name.lower()):
        s = s[len(name):].strip()

    # Trim common separators that may remain
    s = s.lstrip(" -:•\u00a0").strip()

    # Try to extract a known football position phrase
    # (covers EN/DE and common hyphen/space variants)
    pattern = re.compile(
        r"(Goalkeeper|Torwart|"
        r"(?:Centre|Center)[ -]?Back|"
        r"(?:Left|Right)[ -]Back|"
        r"(?:Left|Right)[ -]Wing(?:er)?|"
        r"(?:Wing|Full)[ -]?Back|"
        r"(?:Central|Defensive|Attacking)[ -]?Midfield(?:er)?|"
        r"Midfield(?:er)?|"
        r"(?:Centre|Center)[ -]?Forward|"
        r"Second[ -]?Striker|"
        r"Striker|Forward|Defender)"
        , re.IGNORECASE
    )
    m = pattern.search(s)
    if m:
        pos = m.group(1)
        # Normalize a bit
        pos = pos.replace("Center", "Centre")
        pos = pos.replace("Midfielder", "Midfield")
        # Title-case but keep hyphen casing nice
        pos = "-".join(p.capitalize() for p in pos.split("-"))
        # Fix common two-word capitalization
        pos = pos.replace("Left back", "Left Back").replace("Right back", "Right Back")
        pos = pos.replace("Left Winger", "Left Winger").replace("Right Winger", "Right Winger")
        pos = pos.replace("Second Striker", "Second Striker")
        pos = pos.replace("Full-back", "Full-Back").replace("Wing-back", "Wing-Back")
        pos = pos.replace("Centre forward", "Centre-Forward")
        pos = pos.replace("Centre back", "Centre-Back")
        return pos

    # Fallback: if the remaining text is short-ish, accept it
    if 2 <= len(s) <= 40:
        return s
    return None


def _enrich_positions_from_team_html(team_html_path: Path, df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract positions and nationality from the squad (team) page and merge into df.
    Primary key: numeric player_id from /spieler/<id>/ in the anchor href.
    Fallback: normalized name.
    """
    try:
        html = team_html_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        print("[pipeline] Team-table enrichment: could not read HTML")
        return df

    soup = BeautifulSoup(html, "lxml")

    def _normalize_name(s: str) -> str:
        if not isinstance(s, str):
            return ""
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
        s = re.sub(r"\s+", " ", s)
        return s.strip().lower()

    tables = soup.select("table.items")
    if not tables:
        print("[pipeline] Team-table enrichment: no <table class='items'> found")
        return df

    rows_found: list[tuple[str, str, str, str]] = []  # (player_id, name, position, nationality)

    for tbl in tables:
        for tr in tbl.select("tbody tr"):
            a = (tr.select_one("td a.spielprofil_tooltip") or
                 tr.select_one("td a[data-player]") or
                 tr.select_one('a[href*="/profil/spieler/"]'))
            if not a:
                continue

            name = a.get_text(strip=True)
            href = a.get("href") or ""
            m = re.search(r"/spieler/(\d+)", href)
            pid = m.group(1) if m else None
            if not name:
                continue

            # --- POSITION extraction ---
            pos = None
            pos_span = tr.select_one("td.posrela span.hauptposition")
            if pos_span:
                pos = pos_span.get_text(" ", strip=True)
            if not pos:
                pos_td = tr.find("td", attrs={"data-th": re.compile(r"^(Pos\.|Position)$", re.I)})
                if pos_td:
                    pos = pos_td.get_text(" ", strip=True)
            if not pos:
                pos_td = tr.select_one("td[class*=position]")
                if pos_td:
                    pos = pos_td.get_text(" ", strip=True)
            if not pos:
                for td in tr.find_all("td"):
                    txt = td.get_text(" ", strip=True)
                    if not txt:
                        continue
                    lo = txt.lower()
                    if (2 <= len(txt) <= 40) and any(k in lo for k in [
                        "goalkeeper", "torwart",
                        "defender", "verteidiger", "back", "left back", "right back",
                        "centre-back", "center-back",
                        "midfield", "mittelfeld", "defensive midfield", "attacking midfield",
                        "wing", "flügel", "wide",
                        "forward", "stürmer", "striker", "centre-forward", "center-forward"
                    ]):
                        pos = txt
                        break

            # --- NATIONALITY extraction ---
            nat = None
            nat_img = tr.select_one("td img[title]")
            if nat_img:
                nat = nat_img.get("title") or nat_img.get("alt")

            if pos:
                pos_clean = _clean_team_position(name, pos)
                if pos_clean:
                    rows_found.append((pid, name, pos_clean, nat))

    print(f"[pipeline] Team-table enrichment: candidate rows found = {len(rows_found)}")
    if rows_found:
        print("[pipeline] Sample from team page:", [(n, p) for _, n, p, _ in rows_found[:3]])

    if not rows_found:
        return df

    pos_df = pd.DataFrame(rows_found, columns=["player_id_raw", "name_raw", "position_team", "nationality_team"])
    pos_df["name_key"] = pos_df["name_raw"].apply(_normalize_name)
    pos_df = pos_df.drop_duplicates(subset=["player_id_raw", "name_key"])

    out = df.copy()

    # Ensure player_id column
    if "player_id" not in out.columns or out["player_id"].isna().any() or out["player_id"].eq("").any():
        if "link" in out.columns:
            pid_series = out["link"].str.extract(r"/spieler/(\d+)")[0]
            out["player_id"] = out.get("player_id", pd.Series(index=out.index, dtype="object"))
            out.loc[out["player_id"].isna() | out["player_id"].eq(""), "player_id"] = pid_series

    out["name_key"] = out["name"].apply(_normalize_name)

    if "position" not in out.columns:
        out["position"] = pd.NA
    if "nationality" not in out.columns:
        out["nationality"] = pd.NA

    before_missing = out["position"].isna().sum()

    # 1️⃣ Join by player_id (positions + nationality)
    right_id = pos_df[pos_df["player_id_raw"].notna()].rename(columns={"player_id_raw": "player_id"})
    merged = out.merge(
        right_id[["player_id", "position_team", "nationality_team"]],
        on="player_id",
        how="left",
        suffixes=("", "_byid"),
    )

    # Fill missing position
    mask_pos = merged["position"].isna() | merged["position"].astype(str).str.strip().eq("")
    merged.loc[mask_pos, "position"] = merged.loc[mask_pos, "position_team"]

    # Fill missing nationality
    mask_nat = merged["nationality"].isna() | merged["nationality"].astype(str).str.strip().eq("")
    merged.loc[mask_nat, "nationality"] = merged.loc[mask_nat, "nationality_team"]

    merged = merged.drop(columns=["position_team", "nationality_team"], errors="ignore")
    after_missing = merged["position"].isna().sum()
    filled_diff = max(0, before_missing - after_missing)
    print(f"[pipeline] Team-table enrichment: filled {filled_diff} positions from team page")

    merged = merged.drop(columns=["name_key"], errors="ignore")
    return merged




def _enrich_positions(df: pd.DataFrame, max_workers: int = 4) -> pd.DataFrame:
    # Ensure output cols exist
    for col in ["position", "position_main", "position_alt", "height_cm", "foot"]:
        if col not in df.columns:
            df[col] = pd.NA

    if not {"link"}.issubset(df.columns):
        return df

    df = df.copy()
    needs = df["position"].isna() | df["position"].astype(str).str.strip().eq("")
    todo = df[needs & df["link"].notna()]

    # ensure player_id for caching
    if "player_id" not in df.columns or df["player_id"].isna().any() or df["player_id"].eq("").any():
        id_part = df["link"].str.extract(r"/spieler/(\d+)")
        if "player_id" not in df.columns:
            df["player_id"] = pd.Series(index=df.index, dtype="object")
        df.loc[df["player_id"].isna() | df["player_id"].eq(""), "player_id"] = id_part[0].fillna("unknown")

    if todo.empty:
        print("[pipeline] Positions already present; no enrichment needed.")
        return df

    print(f"[pipeline] Enriching positions for {len(todo)} players...")

    def worker(idx_link_pid: tuple[int, str, str]) -> tuple[int, dict]:
        idx, link, pid = idx_link_pid
        try:
            html = _fetch_or_load_profile(link, pid)
            facts = _extract_profile_facts(html)
            return idx, facts
        except Exception:
            return idx, {}

    tasks = [(idx, row["link"], str(row["player_id"])) for idx, row in todo.iterrows()]

    results: list[tuple[int, dict]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, t) for t in tasks]
        for fut in as_completed(futures):
            results.append(fut.result())

    filled_positions = 0
    filled_total = 0
    for idx, facts in results:
        if not facts:
            continue
        # prefer position_main; also keep legacy 'position' field filled
        main = facts.get("position_main")
        alt = facts.get("position_alt")
        if isinstance(main, str) and main.strip():
            df.at[idx, "position_main"] = main.strip()
            if pd.isna(df.at[idx, "position"]) or str(df.at[idx, "position"]).strip() == "":
                df.at[idx, "position"] = main.strip()
            filled_positions += 1
        if isinstance(alt, str) and alt.strip():
            df.at[idx, "position_alt"] = alt.strip()
        # height / foot
        for k in ("height_cm", "foot"):
            v = facts.get(k)
            if v not in (None, ""):
                df.at[idx, k] = v
                filled_total += 1

    print(f"[pipeline] Profile enrichment filled main/alt for {filled_positions} players; "
          f"{filled_total} extra fields (height/foot).")
    return df


def _map_position_group(pos: str | None) -> str | None:
    if not isinstance(pos, str):
        return None
    p = pos.lower()
    if "torwart" in p or "goalkeeper" in p: return "GK"
    if any(k in p for k in ["innenverteidiger","verteidiger","centre-back","center-back","full-back","right-back","left-back","defender"]): return "DF"
    if any(k in p for k in ["mittelfeld","midfield","winger","flügel"]): return "MF"
    if any(k in p for k in ["sturm","stürmer","forward","striker","centre-forward","center-forward"]): return "FW"
    return None




# ---------- CLI ----------

def main():
    
    p = argparse.ArgumentParser(
        description="Fetch a Transfermarkt team page, parse players, optionally clean and enrich, and save CSV/Parquet."
    )
    p.add_argument("--url", "-u", required=True, help="Full Transfermarkt team page URL (squad page).")
    p.add_argument("--out-name", "-o", default="team.html",
                   help="Filename for saved HTML under data/raw/tm/ (e.g., sturm_graz_team.html).")
    p.add_argument("--clean", dest="clean", action="store_true", default=True,
                   help="Clean/normalize parsed data (default: on).")
    p.add_argument("--no-clean", dest="clean", action="store_false",
                   help="Disable cleaning step.")
    p.add_argument("--enrich-positions", dest="enrich_pos", action="store_true", default=True,
                   help="Fetch each player profile to fill missing positions (default: on).")
    p.add_argument("--no-enrich-positions", dest="enrich_pos", action="store_false",
                   help="Disable position enrichment.")
    p.add_argument("--club", default=None, help="Club name to stamp into rows.")
    p.add_argument("--league", default=None, help="League name to stamp into rows.")
    p.add_argument("--season", default=None, help="Season label to stamp into rows (e.g., 2025/26).")
    p.add_argument("--format", choices=["csv", "parquet", "both"], default="csv",
                   help="Output format (default: csv).")
    args = p.parse_args()

    ensure_dirs()

    # 1) Fetch
    print(f"[pipeline] Fetching: {args.url}")
    html_bytes = fetch(args.url)
    if not html_bytes:
        raise RuntimeError("Fetched empty response from the URL.")

    # 2) Save raw HTML
    dest_html = tm_html(args.out_name)
    dest_html.write_bytes(html_bytes)
    print(f"[pipeline] Saved raw HTML: {dest_html.resolve()} ({len(html_bytes):,} bytes)")

    
    # 3) Parse
    df = _parse_team_html(dest_html)
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Parser must return a pandas DataFrame.")
    print(f"[pipeline] Parsed {len(df)} rows, {len(df.columns)} columns")

    # 4) Clean
    if args.clean:
        df = _clean_df(df)
        print(f"[pipeline] Cleaned columns -> {len(df.columns)} total")
       

    # 4.5) Fallback: try to fill positions from team page table
    df = _enrich_positions_from_team_html(dest_html, df)

    # 5) Enrich positions
    if args.enrich_pos:
        df = _enrich_positions(df, max_workers=4)

    # 6) Position group (from main/position)
    if "position_main" in df.columns or "position" in df.columns:
        df["position_group"] = df.get("position_main", df.get("position")).fillna(df.get("position")).apply(_map_position_group)

    # 7) Stamp metadata
    meta = {}
    if args.club:
        meta["club"] = args.club
    if args.league:
        meta["league"] = args.league
    if args.season:
        meta["season"] = args.season
    if meta:
        for k, v in meta.items():
            df[k] = v

    # 8) Save processed
    stem = Path(args.out_name).stem + "_players"
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.format in {"csv", "both"}:
        out_csv = out_dir / f"{stem}.csv"
        df.to_csv(out_csv, index=False)
        print(f"[pipeline] ✅ Saved CSV: {out_csv.resolve()}")

    if args.format in {"parquet", "both"}:
        out_parquet = out_dir / f"{stem}.parquet"
        try:
            df.to_parquet(out_parquet, index=False)
            print(f"[pipeline] ✅ Saved Parquet: {out_parquet.resolve()}")
        except ImportError:
            print("[pipeline] ⚠️ Skipped Parquet (engine missing). Install: pip install pyarrow  (or fastparquet)")


if __name__ == "__main__":
    main()
