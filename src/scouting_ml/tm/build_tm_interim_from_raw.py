# scouting_ml/build_tm_interim_from_raw.py
"""
Build an interim CSV from already-downloaded Transfermarkt HTML files.
Reads data/raw/tm/players/*.html and extracts a minimal schema:
Name, ShirtNumber, Position, PositionsSecondary, MainPositionGroup,
Height, Foot, Market Value, MV_LastUpdate, Date of birth, Age,
PlaceOfBirthCity, PlaceOfBirthCountry, Nationality, Citizenships, PlayerID, SourceFile
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import List, Tuple
import pandas as pd
from bs4 import BeautifulSoup



# --- regex helpers (robust enough for TM variants) ---
RE_HEIGHT_M_CM = re.compile(r"(?:(\d{3})\s*cm)|([1-2](?:[.,]\d{1,2})?)\s*m", re.I)
RE_MV = re.compile(r"€\s*([0-9.,]+)\s*([kKmM])?")
RE_FOOT = re.compile(r"\b(Left|Right|Both)\b", re.I)

MONTHS = {
    m.lower(): i
    for i, m in enumerate(
        ["January","February","March","April","May","June",
         "July","August","September","October","November","December"], 1
    )
}

# Dates like: 1999-06-11 | 11.06.1999 | 11/06/1999 | Jun 11, 1999 | 11 June 1999
DATE_PATTERNS = [
    r"\b(\d{4})-(\d{2})-(\d{2})\b",                # 1999-06-11
    r"\b(\d{2})\.(\d{2})\.(\d{4})\b",              # 11.06.1999
    r"\b(\d{2})/(\d{2})/(\d{4})\b",                # 11/06/1999
    r"\b([A-Za-z]{3,9})\s+(\d{1,2}),\s*(\d{4})\b", # Jun 11, 1999
    r"\b(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{4})\b",  # 11 June 1999
]

POS_GROUP_MAP = {
    "Goalkeeper": "GK",
    "Centre-Back": "DEF",
    "Left-Back": "DEF",
    "Right-Back": "DEF",
    "Defensive Midfield": "MID",
    "Central Midfield": "MID",
    "Attacking Midfield": "MID",
    "Left Winger": "FWD",
    "Right Winger": "FWD",
    "Second Striker": "FWD",
    "Centre-Forward": "FWD",
    "Forward": "FWD",
}

# ---------------------------
# Parsing helpers
# ---------------------------

def parse_dob_iso(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    for pat in DATE_PATTERNS:
        m = re.search(pat, t)
        if not m:
            continue
        try:
            if pat == DATE_PATTERNS[0]:        # yyyy-mm-dd
                y, mo, d = m.groups()
            elif pat in (DATE_PATTERNS[1], DATE_PATTERNS[2]):  # dd.mm.yyyy | dd/mm/yyyy
                d, mo, y = m.groups()
            elif pat == DATE_PATTERNS[3]:      # Mon dd, yyyy
                mon, d, y = m.groups()
                mo = MONTHS.get(mon.lower())
            else:                               # dd Month yyyy
                d, mon, y = m.groups()
                mo = MONTHS.get(mon.lower())
            if mo:
                return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
        except Exception:
            pass
    return ""

def parse_age_in_parens(text: str) -> str:
    m = re.search(r"\((\d{1,2})\)", text or "")
    return m.group(1) if m else ""

def clean_text(s: str) -> str:
    return " ".join(s.split()) if s else s

def parse_market_value(text: str) -> str:
    m = RE_MV.search(text.replace(" ", ""))
    return m.group(0) if m else ""

def parse_height_from_text(text: str) -> str:
    """Return a canonical height string like '184 cm' or '1,84 m' if present in text; else ''."""
    m = RE_HEIGHT_M_CM.search(text)
    if not m:
        return ""
    if m.group(1):  # 'xxx cm'
        return f"{m.group(1)} cm"
    meters = m.group(2)  # with comma or dot
    return f"{meters} m"

def parse_foot(text: str) -> str:
    m = RE_FOOT.search(text)
    return m.group(1).title() if m else ""

def extract_field_by_label(soup: BeautifulSoup, labels: List[str]) -> str:
    """
    Find value by its label text. On Transfermarkt the label and value
    are usually inside the SAME <li class="data-header__label">…<span class="data-header__content">value</span></li>.
    """
    for label in labels:
        el = soup.find(string=re.compile(rf"^{label}\s*:?\s*$", re.I)) \
             or soup.find(string=re.compile(rf"^{label}", re.I))
        if not el:
            continue

        container = el.parent if el and el.parent else None
        if container and container.name not in {"li", "td", "th", "dt", "div"}:
            container = el.find_parent(["li", "td", "th", "dt", "div"])

        if container:
            same = container.find(attrs={"class": re.compile(r"data-header__content", re.I)})
            if same:
                return clean_text(same.get_text(" ", strip=True))

            text_in_container = container.get_text(" ", strip=True)
            label_text = str(el).strip()
            stripped = re.sub(rf"^{re.escape(label_text)}\s*[:\-–]?\s*", "", text_in_container, flags=re.I)
            if stripped and stripped != text_in_container:
                return clean_text(stripped)

        # fallbacks
        if el.parent and el.parent.find_next_sibling():
            val = el.parent.find_next_sibling().get_text(" ", strip=True)
            if val:
                return clean_text(val)

        nxt = el.find_next()
        if nxt:
            val = nxt.get_text(" ", strip=True)
            if val:
                return clean_text(val)

    return ""

def extract_all_citizenships(soup: BeautifulSoup) -> List[str]:
    # Gather all flags/texts from the Citizenship row
    el = soup.find(string=re.compile(r"^\s*(Citizenship|Nationality)\s*:?\s*$", re.I)) \
         or soup.find(string=re.compile(r"^\s*(Citizenship|Nationality)", re.I))
    if not el:
        return []
    container = el.find_parent(["li", "td", "th", "dt", "div"]) or el.parent
    if not container:
        return []
    vals: List[str] = []

    for a in container.find_all("a"):
        t = (a.get("title") or a.get_text(" ", strip=True) or "").strip()
        if t:
            vals.append(t)

    for img in container.find_all("img"):
        t = (img.get("title") or img.get("alt") or "").strip()
        if t:
            vals.append(t)

    text_tail = container.get_text(" ", strip=True)
    label_text = el.strip() if isinstance(el, str) else str(el).strip()
    text_tail = re.sub(rf"^{re.escape(label_text)}\s*[:\-–]?\s*", "", text_tail, flags=re.I)
    if text_tail:
        for part in [p.strip() for p in text_tail.split(",")]:
            if part and part not in vals:
                vals.append(part)

    # de-dup preserve order
    seen, out = set(), []
    for v in vals:
        if v not in seen:
            seen.add(v); out.append(v)
    return out

def extract_place_of_birth(soup: BeautifulSoup) -> Tuple[str, str]:
    # Place of birth row: city + country flag
    el = soup.find(string=re.compile(r"^\s*Place of birth\s*:?\s*$", re.I)) \
         or soup.find(string=re.compile(r"^\s*Geburtsort\s*:?\s*$", re.I))
    if not el:
        return ("", "")
    container = el.find_parent(["li", "td", "th", "dt", "div"]) or el.parent
    if not container:
        return ("", "")

    city = ""
    cont = container.find(attrs={"class": re.compile(r"data-header__content", re.I)})
    if cont:
        city = cont.get_text(" ", strip=True)

    country = ""
    img = container.find("img")
    if img:
        country = (img.get("title") or img.get("alt") or "").strip()

    return (city, country)

def extract_agent(soup: BeautifulSoup) -> str:
    el = soup.find(string=re.compile(r"^\s*Agent\s*:?\s*$", re.I))
    if not el:
        return ""
    container = el.find_parent(["li", "td", "th", "dt", "div"]) or el.parent
    if not container:
        return ""
    same = container.find(attrs={"class": re.compile(r"data-header__content", re.I)})
    if same:
        return same.get_text(" ", strip=True)
    return container.get_text(" ", strip=True)

def extract_mv_last_update(soup: BeautifulSoup) -> str:
    # The small “Last update: dd/mm/yyyy” near market value
    node = soup.find(class_=re.compile(r"data-header__last-update"))
    if not node:
        return ""
    text = node.get_text(" ", strip=True)
    m = re.search(r"(\d{2}/\d{2}/\d{4})", text)
    return m.group(1) if m else ""

# ---------------------------
# Main per-file parser
# ---------------------------

def parse_player_html(path: Path) -> dict:
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    full_text = soup.get_text(" ", strip=True)

    # Name
    name = ""
    h1 = soup.find("h1")
    if h1:
        name = clean_text(h1.get_text())
    if not name and soup.title:
        name = clean_text(soup.title.get_text())

    # Shirt number (from leading '#xx ' in Name if present)
    shirt_number = ""
    mnum = re.match(r"#\s*([0-9]{1,3})\b", name)
    if mnum:
        shirt_number = mnum.group(1)

    # Position (main)
    position = extract_field_by_label(soup, ["Main position", "Position", "Main Position"])
    if not position:
        infobox = soup.find(attrs={"class": re.compile(r"data-header|info-table|dataContent", re.I)})
        if infobox:
            txt = infobox.get_text(" ", strip=True)
            m = re.search(
                r"(Goalkeeper|Centre-Back|Left-Back|Right-Back|Defensive Midfield|Central Midfield|"
                r"Attacking Midfield|Left Winger|Right Winger|Second Striker|Centre-Forward|Forward)",
                txt,
            )
            position = m.group(1) if m else ""

    # Secondary positions (if present in the profile/information box)
    positions_secondary: List[str] = []
    info = soup.find(attrs={"class": re.compile(r"info-table|dataContent", re.I)})
    if info:
        txt = info.get_text(" ", strip=True)
        m = re.search(r"Other position(?:s)?:\s*(.+?)(?:\s{2,}|$)", txt, re.I)
        if m:
            for p in re.split(r"[,/]| - ", m.group(1)):
                pp = p.strip()
                if pp and pp != position:
                    positions_secondary.append(pp)

    # Position group
    position_group = POS_GROUP_MAP.get(position, "")

    # Height — ONLY from full page regex to avoid wrong cells
    height = parse_height_from_text(full_text)

    # Foot
    foot = extract_field_by_label(soup, ["Foot", "Preferred foot"])
    if not foot:
        foot = parse_foot(full_text)

    # Market value (current) + last update
    market_value = extract_field_by_label(soup, ["Market value", "Current market value"])
    if not market_value:
        market_value = parse_market_value(full_text)
    mv_last_update = extract_mv_last_update(soup)

    # DoB + Age (same-node content like "11/06/1999 (26)")
    dob_age_raw = extract_field_by_label(
        soup,
        ["Date of birth/Age", "Date of birth", "Born",
         "Geburtsdatum/Alter", "Geburtsdatum", "Geb./Alter"]
    )
    dob_iso = parse_dob_iso(dob_age_raw)
    age_from_page = parse_age_in_parens(dob_age_raw)

    # Citizenships (all) and primary Nationality
    cit_all = extract_all_citizenships(soup)
    nationality = cit_all[0] if cit_all else ""

    # Place of birth
    pob_city, pob_country = extract_place_of_birth(soup)

    # Agent
    agent = extract_agent(soup)

    # Player ID from filename (e.g., jon_doe_12345.html)
    m = re.search(r"_(\d+)\.html$", path.name)
    player_id = m.group(1) if m else ""

    return {
        "Name": name,
        "ShirtNumber": shirt_number,
        "Position": position,
        "PositionsSecondary": ";".join(positions_secondary) if positions_secondary else "",
        "MainPositionGroup": position_group,
        "Height": height,
        "Foot": foot,
        "Market Value": market_value,
        "MV_LastUpdate": mv_last_update,
        "Date of birth": dob_iso,
        "Age": age_from_page,
        "PlaceOfBirthCity": pob_city,
        "PlaceOfBirthCountry": pob_country,
        "Nationality": nationality,
        "Citizenships": ";".join(cit_all) if cit_all else "",
        "PlayerID": player_id,
        "SourceFile": path.name,
    }

# ---------------------------
# CLI runner
# ---------------------------

@app.command()
def run(
    raw_dir: str = typer.Option("data/raw/tm/players", "--raw-dir", help="Folder with player HTMLs for ONE club"),
    out_csv: str = typer.Option("data/interim/tm/sturm_graz_team_parsed.csv", "--out", help="Output CSV for this club"),
):
    RAW_PLAYERS_DIR = Path(raw_dir)
    OUT_CSV = Path(out_csv)

    RAW_PLAYERS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in sorted(RAW_PLAYERS_DIR.glob("*.html")):
        try:
            rows.append(parse_player_html(p))
        except Exception as e:
            rows.append({
                "Name": "", "ShirtNumber": "", "Position": "", "PositionsSecondary": "",
                "MainPositionGroup": "", "Height": "", "Foot": "", "Market Value": "",
                "MV_LastUpdate": "", "Date of birth": "", "Age": "", "PlaceOfBirthCity": "",
                "PlaceOfBirthCountry": "", "Nationality": "", "Citizenships": "",
                "PlayerID": "", "SourceFile": p.name, "parse_error": str(e)
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"[build_tm_interim_from_raw] Wrote {OUT_CSV} | rows={len(df)} | cols={len(df.columns)}")


def main():
    RAW_PLAYERS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in sorted(RAW_PLAYERS_DIR.glob("*.html")):
        try:
            rows.append(parse_player_html(p))
        except Exception as e:
            rows.append({
                "Name": "", "ShirtNumber": "", "Position": "", "PositionsSecondary": "",
                "MainPositionGroup": "", "Height": "", "Foot": "", "Market Value": "",
                "MV_LastUpdate": "", "Date of birth": "", "Age": "", "PlaceOfBirthCity": "",
                "PlaceOfBirthCountry": "", "Nationality": "", "Citizenships": "",
                "PlayerID": "", "SourceFile": p.name, "parse_error": str(e)
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"[build_tm_interim_from_raw] Wrote {OUT_CSV} | rows={len(df)} | cols={len(df.columns)}")

if __name__ == "__main__":
    main()
