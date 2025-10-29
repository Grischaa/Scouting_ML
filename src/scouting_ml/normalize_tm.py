# scouting_ml/normalize_tm.py
"""
Normalize Transfermarkt team/player tables:
- Parse heights ("1,90 m" -> 190)
- Parse market values ("€3.5m" / "€350k" -> 3500000 / 350000)
- Standardize positions/feet/nationalities
- Derive age from DoB (if present)
- Tidy column names (snake_case) and dtypes
"""

from __future__ import annotations
import re
from datetime import date, datetime
from typing import Optional

import pandas as pd
import typer

app = typer.Typer(add_completion=False, help="Normalize parsed Transfermarkt data")

# ---- helpers ---------------------------------------------------------------

_MV_RE = re.compile(r"€\s*([0-9]+(?:[.,][0-9]+)?)\s*([kKmM])?")
# matches "184 cm" OR "1,84 m" / "1.84 m"
_HEIGHT_RE = re.compile(r"(?:(\d{3})\s*cm)|([1-2](?:[.,]\d{1,2})?)\s*m", re.I)

POSITION_MAP = {
    # granular -> buckets (your earlier convention)
    "Goalkeeper": "GK",
    "Centre-Back": "CB",
    "Left-Back": "LB",
    "Right-Back": "RB",
    "Defender": "DF",
    "Defensive Midfield": "DM",
    "Central Midfield": "CM",
    "Attacking Midfield": "AM",
    "Left Midfield": "LM",
    "Right Midfield": "RM",
    "Left Winger": "LW",
    "Right Winger": "RW",
    "Second Striker": "SS",
    "Centre-Forward": "CF",
    "Forward": "FW",
}

FOOT_MAP = {
    "right": "Right",
    "links": "Left",   # DE edge case
    "left": "Left",
    "both": "Both",
    "both feet": "Both",
}

COUNTRY_MAP = {
    # optional light remapping
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Côte d’Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "DR Congo": "Congo DR",
}

def _parse_height_cm(s: Optional[str]) -> Optional[int]:
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    m = _HEIGHT_RE.search(s)
    if not m:
        return None
    if m.group(1):  # "184 cm"
        return int(m.group(1))
    meters = float(m.group(2).replace(",", "."))
    return int(round(meters * 100))

def _parse_market_value_eur(s: Optional[str]) -> Optional[int]:
    if not s or not isinstance(s, str):
        return None
    s = s.replace(" ", "")
    m = _MV_RE.search(s)
    if not m:
        return None
    num = float(m.group(1).replace(",", "."))
    suffix = (m.group(2) or "").lower()
    if suffix == "m":
        num *= 1_000_000
    elif suffix == "k":
        num *= 1_000
    return int(round(num))

def _to_date(x: Optional[str]) -> Optional[date]:
    if x is None or pd.isna(x):
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d.%m.%Y", "%b %d, %Y"):
        try:
            return datetime.strptime(str(x), fmt).date()
        except Exception:
            continue
    return None

def _calc_age(d: Optional[date]) -> Optional[float]:
    if not d:
        return None
    today = date.today()
    return round((today - d).days / 365.25, 2)

def _norm_position(p: Optional[str]) -> Optional[str]:
    if not p or not isinstance(p, str):
        return None
    p = p.strip()
    return POSITION_MAP.get(p, p)

def _norm_foot(f: Optional[str]) -> Optional[str]:
    if not f or not isinstance(f, str):
        return None
    f = f.strip().lower()
    return FOOT_MAP.get(f, f.title())

def _snake(s: str) -> str:
    s = s.strip().replace("/", " ").replace(".", "").replace("%", "pct")
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_").lower()

# ---- core ------------------------------------------------------------------

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) snake-case columns
    df.columns = [_snake(c) for c in df.columns]

    # 2) locate likely columns (based on your parser output)
    #    From build step: name, shirtnumber, position, positionssecondary, mainpositiongroup,
    #    height, foot, market_value, mv_lastupdate, date_of_birth, age,
    #    placeofbirthcity, placeofbirthcountry, nationality, citizenships, playerid, sourcefile
    name_col   = next((c for c in df.columns if c in ["name","player","player_name"]), None)
    pos_col    = next((c for c in df.columns if c in ["position","main_position"]), None)
    height_col = next((c for c in df.columns if c in ["height","height_m","size"]), None)
    mv_col     = next((c for c in df.columns if c in ["market_value","mv","value"]), None)
    foot_col   = next((c for c in df.columns if c in ["foot","preferred_foot","stronger_foot"]), None)
    dob_col    = next((c for c in df.columns if c in ["date_of_birth","dob","birth_date","born"]), None)
    nat_col    = next((c for c in df.columns if c in ["nationality","nation"]), None)
    club_col   = next((c for c in df.columns if c in ["club","team"]), None)

    # extra parsed fields (snake-cased)
    shirtnumber_col   = "shirtnumber" if "shirtnumber" in df.columns else None
    pos_group_col     = "mainpositiongroup" if "mainpositiongroup" in df.columns else None
    pos2_col          = "positionssecondary" if "positionssecondary" in df.columns else None
    pob_city_col      = "placeofbirthcity" if "placeofbirthcity" in df.columns else None
    pob_country_col   = "placeofbirthcountry" if "placeofbirthcountry" in df.columns else None
    mv_update_col     = "mv_lastupdate" if "mv_lastupdate" in df.columns else None
    citizenships_col  = "citizenships" if "citizenships" in df.columns else None
    agent_col         = "agent" if "agent" in df.columns else None

    # 3) normalized columns
    if height_col:
        df["height_cm"] = df[height_col].map(_parse_height_cm)

    if mv_col:
        df["market_value_eur"] = df[mv_col].map(_parse_market_value_eur)

    if pos_col:
        df["position_std"] = df[pos_col].map(_norm_position)

    if foot_col:
        df["foot_std"] = df[foot_col].map(_norm_foot)

    # DOB + AGE
    if dob_col:
        dob_parsed = df[dob_col].map(_to_date)
        df["dob"] = dob_parsed
        df["age"] = dob_parsed.map(_calc_age)

    # Fallback: if no dob-derived age, try any existing 'age' numeric column (e.g., from page)
    if "age" not in df or df["age"].isna().all():
        legacy_age_col = next((c for c in df.columns if c in ["age","age_from_page"]), None)
        if legacy_age_col:
            df["age"] = pd.to_numeric(df[legacy_age_col], errors="coerce")

    # Nationality cleanup + light remap
    if nat_col:
        df["nationality_std"] = (
            df[nat_col]
            .astype(str)
            .str.replace(r"\s*\(.*?\)", "", regex=True)
            .str.strip()
            .map(lambda x: COUNTRY_MAP.get(x, x))
        )

    # Shirt number (as numeric)
    if shirtnumber_col:
        df["shirt_number"] = pd.to_numeric(df[shirtnumber_col], errors="coerce")

    # Main position group (keep only GK/DEF/MID/FWD)
    if pos_group_col:
        df["pos_group"] = df[pos_group_col].map(lambda x: x if x in {"GK","DEF","MID","FWD"} else None)

    # Secondary positions -> list
    if pos2_col:
        df["positions_secondary"] = df[pos2_col].fillna("").apply(
            lambda s: [p.strip() for p in re.split(r"[;/]", s) if p.strip()]
        )

    # Place of birth + flag whether born in nationality country
    if pob_city_col:
        df["placeofbirthcity"] = df[pob_city_col].replace("", pd.NA)
    if pob_country_col:
        df["placeofbirthcountry"] = df[pob_country_col].replace("", pd.NA)
    if pob_country_col and "nationality_std" in df:
        df["born_in_nat_country"] = (
            df[pob_country_col].astype(str).str.lower().str.normalize("NFKC")
            == df["nationality_std"].astype(str).str.lower().str.normalize("NFKC")
        ).astype("Int64")

    # Citizenships: list + counts + dual flag
    if citizenships_col:
        df["citizenships_list"] = df[citizenships_col].fillna("").apply(
            lambda s: [x.strip() for x in s.split(";") if x.strip()]
        )
        df["n_citizenships"] = df["citizenships_list"].apply(len)
        df["has_dual_citizenship"] = (df["n_citizenships"] > 1).astype("Int64")

    # Agent presence
    if agent_col:
        df["has_agent"] = df[agent_col].fillna("").str.len().gt(0).astype("Int64")

    # MV last update: keep raw string for now
    if mv_update_col:
        df["mv_last_update_raw"] = df[mv_update_col].replace("", pd.NA)

    # 4) reorder useful columns first (if present)
    ordered = [
        c for c in [
            name_col, club_col, "shirt_number",
            "position_std", pos_col, "pos_group",
            "height_cm", height_col, "foot_std", foot_col,
            "market_value_eur", mv_col, "dob", "age",
            "nationality_std", nat_col,
            "positions_secondary",
            "placeofbirthcity", "placeofbirthcountry",
            "born_in_nat_country",
            "citizenships_list", "n_citizenships", "has_dual_citizenship",
            "has_agent", "mv_last_update_raw",
        ] if c and (c in df.columns)
    ]
    remainder = [c for c in df.columns if c not in ordered]
    df = df[ordered + remainder]

    # 5) drop obvious dupes
    if name_col:
        df = df.drop_duplicates(subset=[name_col, club_col] if club_col else [name_col])

    return df.reset_index(drop=True)

# ---- CLI -------------------------------------------------------------------

@app.command()
def run(
    infile: str = typer.Option(..., "--in", help="Path to CSV from Step 1 (parsed/cleaned)."),
    outfile: str = typer.Option(..., "--out", help="Where to write normalized CSV."),
):
    """Normalize a parsed Transfermarkt CSV."""
    df = pd.read_csv(infile)
    norm = normalize_df(df)
    norm.to_csv(outfile, index=False)
    typer.echo(f"[normalize_tm] Wrote: {outfile}  | rows={len(norm)} cols={len(norm.columns)}")

if __name__ == "__main__":
    app()
