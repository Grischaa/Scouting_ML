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

EXTERNAL_PREFIXES = (
    "contract_",
    "injury_",
    "transfer_",
    "nt_",
    "clubctx_",
    "leaguectx_",
    "sb_",
    "avail_",
    "fixture_",
    "odds_",
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
    season_norm = out["season"].apply(_normalize_season_label)
    if "league_country" in out.columns:
        league_country = out["league_country"].where(out["league_country"].notna(), out["league"].map(LEAGUE_COUNTRY_MAP))
    else:
        league_country = out["league"].map(LEAGUE_COUNTRY_MAP)
        out["league_country"] = league_country

    coeff_lookup = (
        coeff.drop_duplicates(subset=["country", "season"], keep="last")
        .set_index(["country", "season"])[["uefa_points", "rank", "points_total"]]
    )
    join_index = pd.MultiIndex.from_arrays([league_country, season_norm], names=["country", "season"])
    matched = coeff_lookup.reindex(join_index)

    out["uefa_coeff_points"] = matched["uefa_points"].to_numpy(copy=False)
    out["uefa_coeff_rank"] = matched["rank"].to_numpy(copy=False)
    out["uefa_coeff_5yr_total"] = matched["points_total"].to_numpy(copy=False)
    return out


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
        (
            "statsbomb_player_season_features.csv",
            "sb_",
            [
                ("player_id", "season"),
                ("transfermarkt_id", "season"),
            ],
        ),
        (
            "player_availability.csv",
            "avail_",
            [
                ("player_id", "season"),
                ("transfermarkt_id", "season"),
                ("name", "dob", "season"),
            ],
        ),
        (
            "fixture_context.csv",
            "fixture_",
            [
                ("league", "season", "club"),
                ("league_country", "season", "club"),
            ],
        ),
        (
            "market_context.csv",
            "odds_",
            [
                ("league", "season", "club"),
                ("league_country", "season", "club"),
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


def _meaningful_presence(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").notna()
    return (
        series.astype(str)
        .str.strip()
        .replace({"nan": "", "NaN": "", "None": "", "<NA>": ""})
        .ne("")
    )


def _safe_numeric_series(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce")


def _rowwise_non_null_share(frame: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    if not cols:
        return pd.Series(0.0, index=frame.index, dtype="float64")
    presence = pd.DataFrame({_col: _meaningful_presence(frame[_col]) for _col in cols}, index=frame.index)
    return presence.mean(axis=1).astype(float)


def _add_external_presence_and_context_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for prefix in EXTERNAL_PREFIXES:
        cols = [c for c in out.columns if c.startswith(prefix)]
        if not cols:
            continue
        share_col = f"{prefix}non_null_share".replace("__", "_")
        flag_col = f"{prefix}has_data".replace("__", "_")
        share = _rowwise_non_null_share(out, cols)
        out[share_col] = share
        out[flag_col] = (share > 0).astype(int)

    profile_prefixes = ("contract_", "injury_", "transfer_", "nt_")
    provider_prefixes = ("sb_", "avail_", "fixture_", "odds_")
    context_prefixes = ("clubctx_", "leaguectx_")

    out["profile_external_coverage_share"] = _rowwise_non_null_share(
        out,
        [c for c in out.columns if c.startswith(profile_prefixes)],
    )
    out["provider_external_coverage_share"] = _rowwise_non_null_share(
        out,
        [c for c in out.columns if c.startswith(provider_prefixes)],
    )
    out["environment_context_coverage_share"] = _rowwise_non_null_share(
        out,
        [c for c in out.columns if c.startswith(context_prefixes)],
    )

    contract_years_left = _safe_numeric_series(out, "contract_years_left")
    contract_joined_year = _safe_numeric_series(out, "contract_joined_year")
    injury_days_per_1000 = _safe_numeric_series(out, "injury_days_per_1000_min")
    injury_count = _safe_numeric_series(out, "injury_count").fillna(0.0)
    injury_major = _safe_numeric_series(out, "injury_major_injury_flag").fillna(0.0)
    injury_soft_tissue = _safe_numeric_series(out, "injury_soft_tissue_count").fillna(0.0)
    injury_bone_joint = _safe_numeric_series(out, "injury_bone_joint_count").fillna(0.0)
    injury_illness = _safe_numeric_series(out, "injury_illness_count").fillna(0.0)
    injury_surgery = _safe_numeric_series(out, "injury_surgery_count").fillna(0.0)
    transfer_count_3y = _safe_numeric_series(out, "transfer_count_3y").fillna(0.0)
    transfer_paid_moves_3y = _safe_numeric_series(out, "transfer_paid_moves_3y").fillna(0.0)
    transfer_loans_3y = _safe_numeric_series(out, "transfer_loans_3y").fillna(0.0)
    transfer_fees_3y = _safe_numeric_series(out, "transfer_total_fees_3y_eur").fillna(0.0)
    transfer_max_fee = _safe_numeric_series(out, "transfer_max_fee_career_eur").fillna(0.0)
    transfer_last_fee = _safe_numeric_series(out, "transfer_last_transfer_fee_eur")
    transfer_last_loan = _safe_numeric_series(out, "transfer_last_transfer_is_loan").fillna(0.0)
    nt_caps = _safe_numeric_series(out, "nt_total_caps").fillna(0.0)
    nt_senior_caps = _safe_numeric_series(out, "nt_senior_caps").fillna(0.0)
    nt_full = _safe_numeric_series(out, "nt_is_full_international").fillna(0.0)
    club_strength = _safe_numeric_series(out, "clubctx_club_strength_proxy").fillna(0.0)
    league_strength = _safe_numeric_series(out, "leaguectx_league_strength_index").fillna(0.0)
    uefa_total = _safe_numeric_series(out, "uefa_coeff_5yr_total").fillna(0.0)
    club_goals_per90 = _safe_numeric_series(out, "clubctx_club_goals_per90").fillna(0.0)
    club_xg_per90 = _safe_numeric_series(out, "clubctx_club_xg_per90").fillna(0.0)
    odds_rank = _safe_numeric_series(out, "odds_implied_strength_rank")
    fixture_minutes_share = _safe_numeric_series(out, "avail_minutes_share")
    avail_start_share = _safe_numeric_series(out, "avail_start_share")
    avail_full_match_share = _safe_numeric_series(out, "avail_full_match_share")
    avail_unused_bench_share = _safe_numeric_series(out, "avail_unused_bench_share")
    avail_captain_share = _safe_numeric_series(out, "avail_captain_share")
    avail_appearance_share = _safe_numeric_series(out, "avail_appearance_share")
    avail_rating_mean = _safe_numeric_series(out, "avail_rating_mean")
    fixture_points_per_match = _safe_numeric_series(out, "fixture_points_per_match")
    fixture_goal_diff_per_match = _safe_numeric_series(out, "fixture_goal_diff_per_match")
    fixture_clean_sheet_share = _safe_numeric_series(out, "fixture_clean_sheet_share")
    fixture_failed_to_score_share = _safe_numeric_series(out, "fixture_failed_to_score_share")
    fixture_scoring_environment = _safe_numeric_series(out, "fixture_scoring_environment")
    injury_avg_days_per_case = _safe_numeric_series(out, "injury_avg_days_per_case")
    injury_avg_games_per_case = _safe_numeric_series(out, "injury_avg_games_per_case")
    injury_long_absence_count = _safe_numeric_series(out, "injury_long_absence_count").fillna(0.0)
    injury_repeat_soft_tissue_flag = _safe_numeric_series(out, "injury_repeat_soft_tissue_flag").fillna(0.0)
    injury_repeat_bone_joint_flag = _safe_numeric_series(out, "injury_repeat_bone_joint_flag").fillna(0.0)

    out["contract_expiring_within_1y"] = (
        contract_years_left.notna() & (contract_years_left <= 1.0)
    ).astype(int)
    out["contract_long_term_flag"] = (
        contract_years_left.notna() & (contract_years_left >= 3.0)
    ).astype(int)
    out["contract_security_score"] = contract_years_left.clip(lower=0.0, upper=5.0) / 5.0
    if contract_joined_year.notna().any() and "season_end_year" in out.columns:
        club_tenure = _safe_numeric_series(out, "season_end_year") - contract_joined_year
        out["club_tenure_years"] = club_tenure.where(club_tenure >= 0.0)
        out["recent_arrival_flag"] = (
            out["club_tenure_years"].notna() & (pd.to_numeric(out["club_tenure_years"], errors="coerce") <= 1.0)
        ).astype(int)
    if "contract_agent_name" in out.columns:
        out["contract_agent_known_flag"] = (
            out["contract_agent_name"]
            .astype(str)
            .str.strip()
            .replace({"nan": "", "NaN": "", "None": "", "<NA>": ""})
            .ne("")
        ).astype(int)
    if "contract_loan_flag" in out.columns:
        out["contract_loan_context_flag"] = (
            pd.to_numeric(out["contract_loan_flag"], errors="coerce").fillna(0.0) > 0.0
        ).astype(int)

    injury_count_nonzero = injury_count.replace({0.0: np.nan})
    out["injury_soft_tissue_share"] = (injury_soft_tissue / injury_count_nonzero).clip(lower=0.0, upper=1.0)
    out["injury_structural_share"] = (injury_bone_joint / injury_count_nonzero).clip(lower=0.0, upper=1.0)
    out["injury_illness_share"] = (injury_illness / injury_count_nonzero).clip(lower=0.0, upper=1.0)
    out["injury_surgery_flag"] = (injury_surgery > 0.0).astype(int)
    out["injury_profile_risk_score"] = (
        out["injury_structural_share"].fillna(0.0) * 0.50
        + out["injury_soft_tissue_share"].fillna(0.0) * 0.25
        + out["injury_surgery_flag"].fillna(0.0) * 0.25
    )

    out["availability_risk_score"] = (
        np.log1p(injury_days_per_1000.clip(lower=0.0).fillna(0.0)) * 0.55
        + np.log1p(injury_count.clip(lower=0.0)) * 0.20
        + injury_major.clip(lower=0.0, upper=1.0) * 0.25
        + out["injury_profile_risk_score"].fillna(0.0) * 0.35
        + np.log1p(injury_avg_days_per_case.clip(lower=0.0).fillna(0.0)) * 0.12
        + np.log1p(injury_avg_games_per_case.clip(lower=0.0).fillna(0.0)) * 0.08
        + np.log1p(injury_long_absence_count.clip(lower=0.0)) * 0.10
        + injury_repeat_soft_tissue_flag.clip(lower=0.0, upper=1.0) * 0.10
        + injury_repeat_bone_joint_flag.clip(lower=0.0, upper=1.0) * 0.12
    )
    out["durability_score"] = 1.0 / (1.0 + out["availability_risk_score"].fillna(0.0))

    out["transfer_activity_score"] = (
        np.log1p(transfer_count_3y.clip(lower=0.0)) * 0.5
        + np.log1p((transfer_fees_3y / 1_000_000.0).clip(lower=0.0)) * 0.3
        + np.log1p((transfer_max_fee / 1_000_000.0).clip(lower=0.0)) * 0.2
    )
    out["transfer_instability_flag"] = (
        (transfer_count_3y >= 2.0) | (_safe_numeric_series(out, "transfer_loans_3y").fillna(0.0) >= 1.0)
    ).astype(int)
    transfer_count_nonzero = transfer_count_3y.replace({0.0: np.nan})
    out["transfer_recent_paid_share_3y"] = (transfer_paid_moves_3y / transfer_count_nonzero).clip(lower=0.0, upper=1.0)
    out["transfer_recent_loan_share_3y"] = (transfer_loans_3y / transfer_count_nonzero).clip(lower=0.0, upper=1.0)
    out["transfer_last_move_paid_flag"] = (
        transfer_last_fee.fillna(0.0).gt(0.0) & transfer_last_loan.fillna(0.0).le(0.0)
    ).astype(int)
    if "transfer_last_transfer_fee_text" in out.columns:
        out["transfer_last_move_free_flag"] = (
            out["transfer_last_transfer_fee_text"]
            .astype(str)
            .str.casefold()
            .str.contains(r"free|abl.sefrei", regex=True, na=False)
            | (
                transfer_last_fee.fillna(0.0).le(0.0)
                & transfer_last_loan.fillna(0.0).le(0.0)
                & out["transfer_last_transfer_fee_text"].astype(str).str.strip().ne("")
            )
        ).astype(int)

    out["international_exposure_score"] = (
        np.log1p(nt_caps.clip(lower=0.0)) * 0.45
        + np.log1p(nt_senior_caps.clip(lower=0.0)) * 0.40
        + nt_full.clip(lower=0.0, upper=1.0) * 0.15
    )

    context_elite_parts = [
        np.log1p(nt_caps.clip(lower=0.0)),
        club_strength.clip(lower=0.0),
        league_strength.clip(lower=0.0),
        np.log1p(uefa_total.clip(lower=0.0)),
        np.log1p((transfer_max_fee / 1_000_000.0).clip(lower=0.0)),
    ]
    out["elite_context_score"] = sum(context_elite_parts) / float(len(context_elite_parts))
    out["elite_transfer_signal"] = np.log1p((transfer_max_fee / 1_000_000.0).clip(lower=0.0))
    out["club_attacking_environment_score"] = club_goals_per90 + club_xg_per90

    if odds_rank.notna().any():
        out["odds_strength_score"] = 1.0 / odds_rank.clip(lower=1.0)
    if fixture_minutes_share.notna().any():
        out["availability_trust_score"] = fixture_minutes_share.clip(lower=0.0, upper=1.0)
    if (
        avail_start_share.notna().any()
        or avail_full_match_share.notna().any()
        or avail_appearance_share.notna().any()
    ):
        out["availability_selection_score"] = (
            avail_start_share.fillna(0.0) * 0.40
            + avail_full_match_share.fillna(0.0) * 0.25
            + avail_appearance_share.fillna(0.0) * 0.20
            + fixture_minutes_share.fillna(0.0).clip(lower=0.0, upper=1.0) * 0.10
            + avail_captain_share.fillna(0.0) * 0.05
            - avail_unused_bench_share.fillna(0.0) * 0.20
        )
    if avail_rating_mean.notna().any():
        out["availability_performance_hint"] = avail_rating_mean.clip(lower=0.0) / 10.0
    if (
        fixture_points_per_match.notna().any()
        or fixture_goal_diff_per_match.notna().any()
        or fixture_clean_sheet_share.notna().any()
    ):
        out["fixture_team_form_score"] = (
            fixture_points_per_match.fillna(0.0).clip(lower=0.0) / 3.0 * 0.45
            + fixture_goal_diff_per_match.fillna(0.0).clip(lower=-2.0, upper=2.0).add(2.0).div(4.0) * 0.30
            + fixture_clean_sheet_share.fillna(0.0) * 0.15
            - fixture_failed_to_score_share.fillna(0.0) * 0.10
        )
        out["fixture_environment_score"] = (
            fixture_scoring_environment.fillna(0.0).clip(lower=0.0, upper=6.0) / 6.0 * 0.45
            + fixture_clean_sheet_share.fillna(0.0) * 0.20
            + (1.0 - fixture_failed_to_score_share.fillna(0.0).clip(lower=0.0, upper=1.0)) * 0.35
        )

    out["high_value_readiness_score"] = (
        out["elite_context_score"].fillna(0.0) * 0.45
        + out["international_exposure_score"].fillna(0.0) * 0.20
        + out["contract_security_score"].fillna(0.0) * 0.15
        + out["durability_score"].fillna(0.0) * 0.10
        + out["club_attacking_environment_score"].fillna(0.0) * 0.10
    )

    league_strength_norm = (league_strength.clip(lower=0.0) / 0.45).clip(lower=0.0, upper=1.0)
    uefa_norm = (
        np.log1p(uefa_total.clip(lower=0.0)) / np.log1p(25.0)
    ).clip(lower=0.0, upper=1.0)
    out["league_strength_blend"] = (
        league_strength_norm.fillna(0.0) * 0.65
        + uefa_norm.fillna(0.0) * 0.35
    ).clip(lower=0.0, upper=1.0)
    out["club_league_strength_interaction"] = (
        club_strength.clip(lower=0.0).fillna(0.0) * out["league_strength_blend"].fillna(0.0)
    )
    out["international_league_strength_interaction"] = (
        out["international_exposure_score"].fillna(0.0) * out["league_strength_blend"].fillna(0.0)
    )
    out["elite_context_league_interaction"] = (
        out["elite_context_score"].fillna(0.0) * out["league_strength_blend"].fillna(0.0)
    )

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
    result = _add_external_presence_and_context_features(result)

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
    print(f"[dataset] wrote {len(df):,} rows -> {output_path}")


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
        help=(
            "Optional directory for enrichment CSVs "
            "(player_contracts/player_injuries/player_transfers/national_team_caps/"
            "club_context/league_context/statsbomb_player_season_features/"
            "player_availability/fixture_context/market_context)."
        ),
    )
    args = parser.parse_args()
    main(args.data_dir, args.output, args.external_dir)
