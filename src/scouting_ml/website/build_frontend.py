from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import typer

BASE_DIR = Path(__file__).resolve().parent
BIG5_DIR = BASE_DIR / "static" / "data" / "big5"
OUTPUT_PATH = BASE_DIR / "static" / "data" / "players.js"

LEAGUE_NAMES = {
    "premier_league": "Premier League",
    "bundesliga": "Bundesliga",
    "serie_a": "Serie A",
    "ligue_1": "Ligue 1",
    "laliga": "LaLiga",
}

# UI-required columns
UI_COLUMNS = [
    "name",
    "club",
    "position_group",
    "position_main",
    "age",
    "market_value_eur",
    "sofa_minutesPlayed",
    "sofa_goals",
    "sofa_assists",
    "sofa_expectedGoals",
    "player_id",
]

NUMERIC_COLUMNS = [
    "age",
    "market_value_eur",
    "sofa_minutesPlayed",
    "sofa_goals",
    "sofa_assists",
    "sofa_expectedGoals",
]


def derive_position_group(pos: str) -> str:
    """Option C: infer high-level position from text"""
    if not isinstance(pos, str):
        pos = ""
    p = pos.lower()

    if "keeper" in p:
        return "Goalkeeper"
    if any(k in p for k in ["back", "defender", "centre back", "center back", "wing back", "left back", "right back"]):
        return "Defender"
    if "midfield" in p:
        return "Midfielder"
    if any(k in p for k in ["forward", "winger", "striker", "attacker"]):
        return "Forward"

    return "Unknown"


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # position main fallback
    df["position_main"] = df.get("position_main", "").fillna(df.get("position", ""))

    # Option C: derive group
    df["position_group"] = df["position"].apply(derive_position_group)

    # Ensure sofa columns exist
    for col in ["sofa_minutesPlayed", "sofa_goals", "sofa_assists", "sofa_expectedGoals"]:
        if col not in df.columns:
            df[col] = 0

    # Construct output
    out = pd.DataFrame()
    for col in UI_COLUMNS:
        if col in df.columns:
            out[col] = df[col]
        else:
            out[col] = ""

    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    out = out.fillna("")

    return out


def build_frontend_dataset():
    typer.echo("[frontend] Building players.js with season nesting...\n")

    combined = {}
    leagues_list = []

    for csv_path in sorted(BIG5_DIR.glob("*_dataset.csv")):
        slug = csv_path.stem.replace("_dataset", "")
        league_name = LEAGUE_NAMES.get(slug, slug)

        typer.echo(f"[frontend] Loading {league_name} from {csv_path.name}")

        df = pd.read_csv(csv_path)
        df = clean_dataset(df)
        records = df.to_dict(orient="records")

        # *** IMPORTANT: NESTING BY SEASONS ***
        if slug not in combined:
            combined[slug] = {
                "meta": {
                    "slug": slug,
                    "name": league_name
                },
                "seasons": {}
            }

        # We default all Big 5 CSVs to season "2023"
        combined[slug]["seasons"]["2023"] = {
            "meta": {
                "league": league_name,
                "season": "2023",
                "source": csv_path.name,
            },
            "players": records,
        }

        leagues_list.append({
            "slug": slug,
            "name": league_name,
            "seasons": ["2023"],
        })

    # Build JSON payload
    payload = (
        "window.SCOUTING_DATA = "
        + json.dumps(combined, ensure_ascii=False, indent=2)
        + ";\n\n"
        + "window.SCOUTING_LEAGUES = "
        + json.dumps(leagues_list, ensure_ascii=False, indent=2)
        + ";\n\n"
        + "window.SCOUTING_DEFAULT_LEAGUE = "
        + json.dumps(leagues_list[0]["slug"], ensure_ascii=False)
        + ";\n"
    )

    OUTPUT_PATH.write_text(payload, encoding="utf-8")
    typer.echo(f"\n[frontend] ✔ players.js generated → {OUTPUT_PATH}")


def main():
    build_frontend_dataset()


if __name__ == "__main__":
    typer.run(main)
