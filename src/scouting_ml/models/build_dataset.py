from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

from scouting_ml.features.history_strength import add_history_strength_features
from scouting_ml.paths import PROCESSED_DIR

logger = logging.getLogger(__name__)

UEFA_COEFF_PATH = Path("data/external/uefa_country_coefficients.csv")
LEAGUE_COUNTRY_MAP = {
    "English Premier League": "England",
    "LaLiga": "Spain",
    "Bundesliga": "Germany",
    "Serie A": "Italy",
    "Ligue 1": "France",
    "Primeira Liga": "Portugal",
    "Eredivisie": "Netherlands",
    "Belgian Pro League": "Belgium",
    "Turkish Super Lig": "Turkey",
    "Scottish Premiership": "Scotland",
    "Greek Super League": "Greece",
    "Austrian Bundesliga": "Austria",
    "Swiss Super League": "Switzerland",
    "Danish Superliga": "Denmark",
    "Allsvenskan": "Sweden",
    "Norwegian Eliteserien": "Norway",
    "Finnish Veikkausliiga": "Finland",
    "Czech Fortuna Liga": "Czechia",
    "Polish Ekstraklasa": "Poland",
    "Croatian HNL": "Croatia",
    "Serbian SuperLiga": "Serbia",
    "Estonian Meistriliiga": "Estonia",
}
EXTERNAL_LEAKAGE_TOKENS = (
    "market_value",
    "log_market_value",
    "expected_value",
    "value_diff",
)


def load_merged_files(files: List[Path]) -> pd.DataFrame:
    frames = []
    for path in files:
        df = pd.read_csv(path)
        df["source_file"] = str(path)
        frames.append(df)
    if not frames:
        raise ValueError("No merged files found; run the league refresh first.")
    return pd.concat(frames, ignore_index=True, sort=False)


def _normalize_season_label(value: str | float | int | None) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    season = str(value).strip()
    if not season:
        return None
    season = season.replace("\\", "/")
    if "-" in season and "/" not in season:
        season = season.replace("-", "/")
    if "/" in season:
        parts = season.split("/")
        if len(parts) >= 2:
            start = parts[0]
            end = parts[1]
            if len(end) == 4 and len(start) == 4:
                end = end[2:]
            return f"{start}/{end}"
    if len(season) == 4 and season.isdigit():
        year = int(season)
        return f"{year-1}/{str(year)[-2:]}"
    return season


def _load_uefa_coefficients(path: Path = UEFA_COEFF_PATH) -> pd.DataFrame | None:
    if not path.exists():
        logger.info("UEFA coefficient file not found at %s; skipping merge.", path)
        return None
    coeff = pd.read_csv(path)
    required = {"country", "season", "uefa_points", "rank", "points_total"}
    missing = required.difference(coeff.columns)
    if missing:
        logger.warning(
            "UEFA coefficient file missing columns %s; skipping merge.",
            ", ".join(sorted(missing)),
        )
        return None
    coeff = coeff.copy()
    coeff["season"] = coeff["season"].apply(_normalize_season_label)
    coeff["country"] = coeff["country"].astype(str).str.strip()
    for col in ["uefa_points", "rank", "points_total"]:
        coeff[col] = pd.to_numeric(coeff[col], errors="coerce")
    return coeff


def _add_uefa_coefficients(df: pd.DataFrame) -> pd.DataFrame:
    coeff = _load_uefa_coefficients()
    if coeff is None or "season" not in df.columns or "league" not in df.columns:
        return df
    out = df.copy()
    out["_season_norm"] = out["season"].apply(_normalize_season_label)
    out["league_country"] = out["league"].map(LEAGUE_COUNTRY_MAP)
    merged = out.merge(
        coeff,
        left_on=["league_country", "_season_norm"],
        right_on=["country", "season"],
        how="left",
    )
    merged = merged.rename(
        columns={
            "uefa_points": "uefa_coeff_points",
            "rank": "uefa_coeff_rank",
            "points_total": "uefa_coeff_5yr_total",
        }
    )
    merged = merged.drop(columns=["country", "season_y", "_season_norm"], errors="ignore")
    merged = merged.rename(columns={"season_x": "season"})
    return merged


def _normalize_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "season" in out.columns:
        out["season"] = out["season"].apply(_normalize_season_label)
    for key in ["player_id", "transfermarkt_id", "name", "dob", "league", "club", "country", "league_country"]:
        if key in out.columns:
            out[key] = out[key].astype(str).str.strip()
    return out


def _feature_columns_for_external(ext: pd.DataFrame, keys: Sequence[str]) -> list[str]:
    out = []
    for col in ext.columns:
        if col in keys:
            continue
        low = col.lower()
        if any(token in low for token in EXTERNAL_LEAKAGE_TOKENS):
            continue
        out.append(col)
    return out


def _merge_optional_external(
    base: pd.DataFrame,
    path: Path,
    key_options: Sequence[Sequence[str]],
    prefix: str,
) -> pd.DataFrame:
    if not path.exists():
        logger.info("External table missing (%s), skipping.", path.name)
        return base

    ext = pd.read_csv(path)
    if ext.empty:
        logger.info("External table empty (%s), skipping.", path.name)
        return base
    ext = _normalize_key_columns(ext)

    if "contract_until" in ext.columns and "contract_until_year" not in ext.columns:
        ext["contract_until_year"] = pd.to_datetime(ext["contract_until"], errors="coerce").dt.year

    for keys in key_options:
        if not all(k in base.columns for k in keys):
            continue
        if not all(k in ext.columns for k in keys):
            continue

        feature_cols = _feature_columns_for_external(ext, keys)
        if not feature_cols:
            logger.warning("External table %s has no usable feature columns after filtering.", path.name)
            return base

        use = ext[list(keys) + feature_cols].copy()
        rename_map = {c: (c if c.startswith(prefix) else f"{prefix}{c}") for c in feature_cols}
        use = use.rename(columns=rename_map)
        use = use.drop_duplicates(subset=list(keys), keep="last")

        out = base.merge(use, on=list(keys), how="left")
        logger.info(
            "Merged %s on keys %s -> +%d cols",
            path.name,
            ",".join(keys),
            len(rename_map),
        )
        return out

    logger.warning("Could not merge %s: no compatible key set found.", path.name)
    return base


def _merge_external_contexts(df: pd.DataFrame, external_dir: Path | None) -> pd.DataFrame:
    if external_dir is None:
        return df

    out = _normalize_key_columns(df)
    if "league" in out.columns and "league_country" not in out.columns:
        out["league_country"] = out["league"].map(LEAGUE_COUNTRY_MAP)

    specs = [
        (
            "player_contracts.csv",
            "contract_",
            [
                ("player_id", "season"),
                ("transfermarkt_id", "season"),
                ("name", "dob", "season"),
            ],
        ),
        (
            "player_injuries.csv",
            "injury_",
            [
                ("player_id", "season"),
                ("transfermarkt_id", "season"),
                ("name", "dob", "season"),
            ],
        ),
        (
            "player_transfers.csv",
            "transfer_",
            [
                ("player_id", "season"),
                ("transfermarkt_id", "season"),
                ("name", "dob", "season"),
            ],
        ),
        (
            "national_team_caps.csv",
            "nt_",
            [
                ("player_id", "season"),
                ("transfermarkt_id", "season"),
                ("name", "dob", "season"),
            ],
        ),
        (
            "club_context.csv",
            "clubctx_",
            [
                ("league", "season", "club"),
                ("league_country", "season", "club"),
            ],
        ),
        (
            "league_context.csv",
            "leaguectx_",
            [
                ("league", "season"),
                ("league_country", "season"),
            ],
        ),
    ]

    for filename, prefix, key_options in specs:
        out = _merge_optional_external(
            out,
            external_dir / filename,
            key_options=key_options,
            prefix=prefix,
        )

    contract_year_col = None
    if "contract_until_year" in out.columns:
        contract_year_col = "contract_until_year"
    elif "contract_contract_until_year" in out.columns:
        contract_year_col = "contract_contract_until_year"
    if contract_year_col is not None and "season_end_year" in out.columns:
        out["contract_years_left"] = (
            pd.to_numeric(out[contract_year_col], errors="coerce")
            - pd.to_numeric(out["season_end_year"], errors="coerce")
        )
    if "injury_days_missed" in out.columns and "minutes" in out.columns:
        mins = pd.to_numeric(out["minutes"], errors="coerce").replace({0: np.nan})
        out["injury_days_per_1000_min"] = pd.to_numeric(out["injury_days_missed"], errors="coerce") / (mins / 1000.0)

    return out


def add_model_features(df: pd.DataFrame, external_dir: Path | None = Path("data/external")) -> pd.DataFrame:
    result = df.copy()

    def to_numeric(col: str) -> pd.Series:
        return pd.to_numeric(result[col], errors="coerce") if col in result.columns else pd.Series(np.nan, index=result.index)

    if "market_value_eur" in result.columns:
        mv = to_numeric("market_value_eur")
        result["market_value_eur"] = mv
        result["log_market_value"] = np.log1p(mv).where(mv > 0)
    else:
        result["log_market_value"] = np.nan

    minutes = to_numeric("sofa_minutesPlayed")
    result["minutes"] = minutes
    per90 = (90.0 / minutes.replace({0: np.nan})).replace([np.inf, -np.inf], np.nan)

    volume_cols = [
        "sofa_goals",
        "sofa_assists",
        "sofa_expectedGoals",
        "sofa_totalShots",
        "sofa_shotsOnTarget",
        "sofa_keyPasses",
        "sofa_accurateFinalThirdPasses",
        "sofa_accuratePasses",
        "sofa_totalDuelsWon",
        "sofa_groundDuelsWon",
        "sofa_aerialDuelsWon",
        "sofa_successfulDribbles",
        "sofa_tackles",
        "sofa_interceptions",
        "sofa_clearances",
    ]

    for col in volume_cols:
        if col in result.columns:
            result[f"{col}_per90"] = to_numeric(col) * per90

    def safe_ratio(num_col: str, den_col: str, name: str):
        if num_col in result.columns and den_col in result.columns:
            num = to_numeric(num_col)
            den = to_numeric(den_col).replace({0: np.nan})
            result[name] = (num / den).replace([np.inf, -np.inf], np.nan)

    # Stable ratios that avoid per-90 duplicates or mismatched denominators
    safe_ratio("sofa_goals", "sofa_totalShots", "shot_conversion")
    safe_ratio("sofa_assists", "sofa_keyPasses", "assist_to_keypass")
    safe_ratio("sofa_shotsOnTarget", "sofa_totalShots", "shots_on_target_rate")

    for pct_col in [
        "sofa_totalDuelsWonPercentage",
        "sofa_groundDuelsWonPercentage",
        "sofa_aerialDuelsWonPercentage",
        "sofa_successfulDribblesPercentage",
        "sofa_accuratePassesPercentage",
    ]:
        if pct_col in result.columns:
            pct = to_numeric(pct_col)
            result[pct_col] = pct

    if "age" in result.columns:
        age = to_numeric("age")
        result["age"] = age
        result["age_sq"] = age ** 2

    if "season" in result.columns:
        result["season_end_year"] = (
            result["season"]
            .astype(str)
            .str.extract(r"(\d{4})")
            .astype(float)
        )

    result = _merge_external_contexts(result, external_dir=external_dir)

    if "position_group" in result.columns:
        result["is_forward"] = (result["position_group"].str.upper() == "FW").astype(int)
        result["is_midfielder"] = (result["position_group"].str.upper() == "MF").astype(int)
        result["is_defender"] = (result["position_group"].str.upper() == "DF").astype(int)
        result["is_goalkeeper"] = (result["position_group"].str.upper() == "GK").astype(int)

    # Club-season context from non-target stats (safe for leakage)
    group_cols = ["league", "season", "club"]
    if all(c in result.columns for c in group_cols):
        club_agg_cols = []
        for col in ["minutes", "sofa_goals", "sofa_assists", "sofa_expectedGoals", "sofa_totalShots"]:
            if col in result.columns:
                club_agg_cols.append(col)
        if club_agg_cols:
            club_ctx = (
                result.groupby(group_cols, dropna=False)[club_agg_cols]
                .sum(min_count=1)
                .add_prefix("club_")
                .reset_index()
            )
            result = result.merge(club_ctx, on=group_cols, how="left")
            if "sofa_goals" in result.columns and "club_sofa_goals" in result.columns:
                denom = pd.to_numeric(result["club_sofa_goals"], errors="coerce").replace({0: np.nan})
                result["player_goal_share"] = pd.to_numeric(result["sofa_goals"], errors="coerce") / denom
            if "sofa_assists" in result.columns and "club_sofa_assists" in result.columns:
                denom = pd.to_numeric(result["club_sofa_assists"], errors="coerce").replace({0: np.nan})
                result["player_assist_share"] = pd.to_numeric(result["sofa_assists"], errors="coerce") / denom

    # Player lag features across seasons
    if "season_end_year" in result.columns:
        if "player_id" in result.columns:
            result["_player_key"] = result["player_id"].astype(str)
        elif "name" in result.columns:
            result["_player_key"] = result["name"].astype(str)
        else:
            result["_player_key"] = np.arange(len(result)).astype(str)

        sort_cols = ["_player_key", "season_end_year"]
        if "minutes" in result.columns:
            sort_cols.append("minutes")
        result = result.sort_values(sort_cols).reset_index(drop=True)

        lag_cols = [
            c
            for c in [
                "log_market_value",
                "minutes",
                "sofa_goals_per90",
                "sofa_assists_per90",
                "sofa_expectedGoals_per90",
            ]
            if c in result.columns
        ]
        if lag_cols:
            grouped = result.groupby("_player_key", dropna=False)
            for col in lag_cols:
                prev_col = f"prev_{col}"
                result[prev_col] = grouped[col].shift(1)
                result[f"delta_{col}"] = result[col] - result[prev_col]

        result = result.drop(columns=["_player_key"], errors="ignore")

    result = _add_uefa_coefficients(result)
    result = add_history_strength_features(result)

    return result


def main(
    data_dir: str = "data/processed/Clubs combined",
    output: str = "data/model/big5_players.parquet",
    external_dir: str | None = "data/external",
) -> None:
    data_path = Path(data_dir)
    # Search recursively so nested non-Big5 folders are included.
    files = sorted({p for p in data_path.rglob("*_with_sofa.csv")})
    df = load_merged_files(files)
    ext_path = Path(external_dir) if external_dir else None
    df = add_model_features(df, external_dir=ext_path)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"[dataset] wrote {len(df):,} rows → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine *_with_sofa.csv files into a modeling dataset.")
    parser.add_argument(
        "--data-dir",
        default="data/processed/Clubs combined",
        help="Directory containing *_with_sofa.csv files (searched recursively).",
    )
    parser.add_argument("--output", default="data/model/big5_players.parquet", help="Output Parquet path.")
    parser.add_argument(
        "--external-dir",
        default="data/external",
        help="Optional directory for enrichment CSVs (player_contracts/player_injuries/player_transfers/national_team_caps/club_context/league_context).",
    )
    args = parser.parse_args()
    main(args.data_dir, args.output, args.external_dir)
