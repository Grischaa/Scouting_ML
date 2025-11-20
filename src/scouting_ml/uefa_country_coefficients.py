"""
uefa_country_coefficients.py

Utilities to fetch UEFA country coefficient rankings and
attach them as features to a players dataset for market value modelling.

Data source:
    Bert Kassies' UEFA country ranking pages, e.g.
    https://kassiesa.net/uefa/data/method5/crank2025.html

Requires:
    pip install requests pandas numpy
"""

from __future__ import annotations

import logging
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any
import re
import requests
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# 1. FETCH FROM KASSIESA.NET (PRIMARY SOURCE)
# ============================================================================

def _season_labels_for_ranking_year(year: int, n_seasons: int = 5) -> List[str]:
    """
    For a ranking year Y, return the last `n_seasons` season labels,
    e.g. year=2019 → ['14/15', '15/16', '16/17', '17/18', '18/19'].
    """
    labels: List[str] = []
    for offset in range(n_seasons, 0, -1):
        start = year - offset
        end = start + 1
        labels.append(f"{str(start)[-2:]}/{str(end)[-2:]}")
    return labels


def fetch_kassiesa_country_ranking(
    year: int,
    session: Optional[requests.Session] = None,
    user_agent: str = "Mozilla/5.0",
) -> pd.DataFrame:
    """
    Fetch the UEFA country ranking for a given year from kassiesa.net.

    Example URL (year=2025):
        https://kassiesa.net/uefa/data/method5/crank2025.html

    Returns a *wide* DataFrame with columns:
        - country
        - points_total   (5-year total, 'ranking' column on the site)
        - points_<season_label>  e.g. points_20/21, points_21/22, ...
        - rank           (1 = highest total)
        - ranking_year
    """
    if session is None:
        session = requests.Session()

    url = f"https://kassiesa.net/uefa/data/method5/crank{year}.html"
    headers = {"User-Agent": user_agent}

    logger.info("Fetching Kassiesa UEFA country ranking for year %s → %s", year, url)
    resp = session.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    # Read all tables on the page
    tables: List[pd.DataFrame] = pd.read_html(resp.text)
    if not tables:
        raise ValueError(f"No HTML tables found on Kassiesa page for {year} ({url}).")

    # Pick the table that looks like the country ranking:
    #   - has a 'country' column
    #   - has at least one season column like '20/21'
    ranking_table: Optional[pd.DataFrame] = None
    for t in tables:
        cols = [str(c).strip().lower() for c in t.columns]
        has_country = any("country" in c for c in cols)
        has_season = any(re.match(r"\d{2}/\d{2}", c) for c in cols)
        if has_country and has_season:
            ranking_table = t
            break

    if ranking_table is None:
        raise ValueError(f"Could not identify ranking table on Kassiesa page for {year}.")

    df = ranking_table.copy()

    # Drop unnamed index columns if present
    unnamed_cols = [c for c in df.columns if "unnamed" in str(c).lower()]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    # Normalise column names
    df.columns = [str(c).strip() for c in df.columns]

    # Identify key columns
    country_col = next(c for c in df.columns if "country" in c.lower())
    ranking_col = next(c for c in df.columns if "ranking" in c.lower())
    teams_col_candidates = [c for c in df.columns if "team" in c.lower()]
    teams_col = teams_col_candidates[0] if teams_col_candidates else None

    # Season columns: headers like '20/21', '21/22', ...
    season_cols = [c for c in df.columns if re.match(r"\d{2}/\d{2}", str(c))]
    if not season_cols:
        raise ValueError(f"No per-season columns found in Kassiesa table for {year}.")

    # Keep only the relevant columns in a stable order
    keep_cols = [country_col] + season_cols + [ranking_col]
    if teams_col:
        keep_cols.append(teams_col)
    df = df[keep_cols].copy()

    # Remove possible header / empty rows
    df = df[df[country_col].notna()]
    df[country_col] = df[country_col].astype(str).str.strip()
    df = df[df[country_col].str.lower() != "country"]

    # Convert numerics
    df[ranking_col] = pd.to_numeric(df[ranking_col], errors="coerce")
    for c in season_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if teams_col:
        df[teams_col] = pd.to_numeric(df[teams_col], errors="coerce")

    # Sort by total descending and assign rank (1 = highest total)
    df = df.sort_values(ranking_col, ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1, dtype=int)
    df["ranking_year"] = year

    # Rename core columns to internal schema
    df = df.rename(
        columns={
            country_col: "country",
            ranking_col: "points_total",
        }
    )

    # Rename season columns to 'points_<season>'
    rename_map: Dict[str, str] = {}
    for c in season_cols:
        rename_map[c] = f"points_{c}"  # e.g. '20/21' -> 'points_20/21'
    df = df.rename(columns=rename_map)

    return df






# ============================================================================
# 2. OPTIONAL GENERIC HTML TABLE SCRAPER (FOR OTHER SITES)
# ============================================================================

def fetch_country_coefficients_from_html(
    url: str,
    table_index: int = 0,
    user_agent: str = "Mozilla/5.0",
) -> pd.DataFrame:
    """
    Fetch a country coefficient table from a static HTML page using pandas.read_html.

    This is useful for sites that actually render an HTML <table>
    (not used for Kassiesa because his page is plain text).

    Parameters
    ----------
    url : str
        URL of the page containing the table.
    table_index : int
        Index of the table on the page (0 = first table).
    user_agent : str
        User-Agent header.

    Returns
    -------
    pd.DataFrame
    """
    headers = {"User-Agent": user_agent}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    tables: List[pd.DataFrame] = pd.read_html(resp.text)
    if table_index >= len(tables):
        raise IndexError(f"Requested table_index {table_index}, but only {len(tables)} tables found.")

    df = tables[table_index].copy()
    return df


# ============================================================================
# 3. WIDE → LONG FORMAT
# ============================================================================

def wide_to_long_seasons(
    df: pd.DataFrame,
    season_prefix: str = "points_",
    country_col: str = "country",
    total_col: str = "points_total",
    rank_col: str = "rank",
) -> pd.DataFrame:
    """
    Convert a wide country coefficients DataFrame into a long (tidy) format
    with one row per (country, season).

    Parameters
    ----------
    df : pd.DataFrame
        Wide DataFrame from `fetch_kassiesa_country_ranking` (or similar).
    season_prefix : str
        Prefix for per-season coefficient columns (default: 'points_').
    country_col : str
        Name of the country name column.
    total_col : str
        Name of the 5-year total column.
    rank_col : str
        Name of the rank column.

    Returns
    -------
    pd.DataFrame
        Columns:
            - country
            - season           (e.g. '2020/21')
            - uefa_points      (per-season coefficient)
            - points_total     (5-year total)
            - rank
            - (optional) ranking_year (if present in input)
    """
    season_cols = [c for c in df.columns if c.startswith(season_prefix)]
    if not season_cols:
        raise ValueError(f"No columns with prefix '{season_prefix}' found.")

    records: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        base: Dict[str, Any] = {
            "country": row.get(country_col),
            "points_total": row.get(total_col),
            "rank": row.get(rank_col),
        }
        if "ranking_year" in df.columns:
            base["ranking_year"] = row.get("ranking_year")

        for col in season_cols:
            # col e.g. "points_20/21"
            season_label_raw = col.replace(season_prefix, "")
            value = row.get(col)
            if pd.notna(value):
                records.append(
                    {
                        **base,
                        "season": str(season_label_raw),
                        "uefa_points": value,
                    }
                )

    tidy = pd.DataFrame.from_records(records)

    # Ensure numeric
    for col in ("points_total", "rank", "uefa_points"):
        if col in tidy.columns:
            tidy[col] = pd.to_numeric(tidy[col], errors="coerce")

    return tidy


# ============================================================================
# 4. MERGING WITH PLAYER DATA
# ============================================================================

def attach_country_coefficients(
    players_df: pd.DataFrame,
    coeff_df: pd.DataFrame,
    player_country_col: str = "league_country",
    player_season_col: str = "season",
    coeff_country_col: str = "country",
    coeff_season_col: str = "season",
) -> pd.DataFrame:
    """
    Merge UEFA country coefficients onto a players DataFrame.

    Parameters
    ----------
    players_df : pd.DataFrame
        Player-level modelling data (must include league country + season).
    coeff_df : pd.DataFrame
        Output of `wide_to_long_seasons`.
    player_country_col : str
        Column name in `players_df` for the league country (e.g. 'England').
    player_season_col : str
        Season column in `players_df` (e.g. '2024/25', '2023/24', ...).
    coeff_country_col : str
        Country column in the coefficients DataFrame.
    coeff_season_col : str
        Season column in the coefficients DataFrame.

    Returns
    -------
    pd.DataFrame
        `players_df` with new columns:
            - uefa_coeff_points
            - uefa_coeff_rank
            - uefa_coeff_5yr_total
    """
    coeff_df = coeff_df.rename(
        columns={
            coeff_country_col: "_coeff_country",
            coeff_season_col: "_coeff_season",
        }
    )

    merged = players_df.merge(
        coeff_df,
        left_on=[player_country_col, player_season_col],
        right_on=["_coeff_country", "_coeff_season"],
        how="left",
    )

    merged = merged.rename(
        columns={
            "uefa_points": "uefa_coeff_points",
            "points_total": "uefa_coeff_5yr_total",
            "rank": "uefa_coeff_rank",
        }
    )

    merged = merged.drop(columns=["_coeff_country", "_coeff_season"], errors="ignore")
    return merged


# ============================================================================
# 5. SEASON LABEL NORMALISATION
# ============================================================================

def _season_label_from_numeric(value: str) -> str:
    """
    Normalise season labels.

    Rules:
        - If contains '/' or '-', and the first part has 2 digits (e.g. '20/21'),
          expand to '2020/21' (assume 2000s for 00-49, 1900s for 50-99).
        - If contains '/' or '-' and the first part has 4 digits (e.g. '2020/21'),
          leave as-is.
        - If it is a plain 4-digit year 'YYYY', convert to 'YYYY-1/YY'
          (e.g. '2025' -> '2024/25').
        - Otherwise, return unchanged.
    """
    s = str(value).strip()
    if "/" in s or "-" in s:
        # Handle '20/21', '21/22', etc.
        delim = "/" if "/" in s else "-"
        parts = s.split(delim)
        if len(parts) == 2 and parts[0].isdigit() and len(parts[0]) == 2:
            start_short = int(parts[0])
            if start_short <= 49:
                start_full = 2000 + start_short
            else:
                start_full = 1900 + start_short
            return f"{start_full}/{parts[1]}"
        # Already full year like 2020/21
        return s

    if s.isdigit() and len(s) == 4:
        year = int(s)
        start = year - 1
        return f"{start}/{str(year)[-2:]}"

    return s


# ============================================================================
# 6. BUILD FULL DATASET OVER MULTIPLE YEARS
# ============================================================================

def build_coefficients_dataset(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Build a long-format coefficients dataset across multiple ranking years
    using Kassiesa as the source.

    For each ranking year:
        1. Fetch wide-format coefficients from Kassiesa.
        2. Convert to long format with one row per (country, season).
        3. Normalise season labels to 'YYYY/YY'.
    """
    frames: List[pd.DataFrame] = []
    session = requests.Session()

    for year in range(start_year, end_year + 1):
        wide = fetch_kassiesa_country_ranking(year=year, session=session)
        long = wide_to_long_seasons(
            wide,
            season_prefix="points_",
            country_col="country",
            total_col="points_total",
            rank_col="rank",
        )
        frames.append(long)

    result = pd.concat(frames, ignore_index=True)

    # Normalise season labels (e.g. '20/21' -> '2020/21')
    result["season"] = result["season"].apply(_season_label_from_numeric)
    result["country"] = result["country"].astype(str).str.strip()

    return result



# ============================================================================
# 7. IO HELPERS + CLI
# ============================================================================

def save_coefficients_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["season", "country"]).to_csv(output_path, index=False)
    logger.info("Wrote UEFA coefficients → %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UEFA country coefficient utilities.")
    sub = parser.add_subparsers(dest="command")

    fetch_parser = sub.add_parser("fetch", help="Fetch coefficients from Kassiesa")
    fetch_parser.add_argument("--start-year", type=int, required=True, help="First ranking year (e.g. 2019).")
    fetch_parser.add_argument("--end-year", type=int, required=True, help="Last ranking year (e.g. 2025).")
    fetch_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/external/uefa_country_coefficients.csv"),
        help="CSV path to write the long-format coefficients.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.command == "fetch":
        df = build_coefficients_dataset(args.start_year, args.end_year)
        save_coefficients_csv(df, args.output)
    else:
        raise SystemExit("Specify a command, e.g. `fetch`.")


if __name__ == "__main__":
    main()
