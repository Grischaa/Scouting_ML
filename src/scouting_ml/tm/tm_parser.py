from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
from scouting_ml.paths import tm_html, ensure_dirs
from scouting_ml.utils.import_guard import *  # noqa: F403

BASE = "https://www.transfermarkt.com"

def parse_tm_search(filename: str) -> pd.DataFrame:
    """
    Parse a Transfermarkt HTML file (either a team roster or a club search result)
    and return a DataFrame with type, name, link, position, age, nationality, market value.
    """
    ensure_dirs()
    html_path = tm_html(filename)
    html = html_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    records: list[dict] = []

    roster_rows = soup.select("table.items tbody tr")
    if roster_rows:
        seen = set()
        for row in roster_rows:
            link_tag = row.select_one("td.hauptlink a[href*='/profil/spieler/']")
            if not link_tag:
                continue

            href = (link_tag.get("href") or "").strip()
            if not href:
                continue
            if not href.startswith("http"):
                href = BASE + href
            if href in seen:
                continue
            seen.add(href)

            name = link_tag.get_text(strip=True)

            # Extract position
            pos_td = row.select_one("td.posrela td:not(.hauptlink)")
            position = pos_td.get_text(strip=True) if pos_td else None

            # Extract date of birth / age
            dob_td = row.select("td.zentriert")
            dob_age = dob_td[1].get_text(strip=True) if len(dob_td) > 1 else None

            # Extract nationality (flag title)
            nat_td = row.select_one("td.zentriert img[title]")
            nationality = nat_td["title"].strip() if nat_td else None

            # Extract market value
            mv_tag = row.select_one("td.rechts.hauptlink a")
            market_value = mv_tag.get_text(strip=True) if mv_tag else None

            records.append({
                "type": "player",
                "name": name,
                "link": href,
                "position": position,
                "dob_age": dob_age,
                "nationality": nationality,
                "market_value": market_value
            })

    # --- fallback to club search ---
    if not records:
        for row in soup.select("table.items tbody tr"):
            link_tag = row.select_one("td.hauptlink a[href*='/verein/']")
            if not link_tag:
                continue
            href = (link_tag.get("href") or "").strip()
            if not href:
                continue
            if not href.startswith("http"):
                href = BASE + href
            name = link_tag.get_text(strip=True)
            records.append({
                "type": "club",
                "name": name,
                "link": href
            })

    df = pd.DataFrame.from_records(records)
    print(f"[tm_parser] extracted {len(df)} entries from {filename}")
    return df


if __name__ == "__main__":
    df = parse_tm_search("sturm_graz_team.html")
    print(df.head(10))
