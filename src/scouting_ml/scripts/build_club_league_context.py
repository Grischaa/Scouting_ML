from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def normalize_season_label(value: str | float | int | None) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
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


def _read_players_table(path: Path) -> pd.DataFrame:
    if path.is_dir():
        files = sorted(path.glob("*_with_sofa.csv"))
        if not files:
            raise ValueError(f"No *_with_sofa.csv files found in {path}")
        frames = [pd.read_csv(p) for p in files]
        return pd.concat(frames, ignore_index=True, sort=False)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _to_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return (num / den.replace({0: np.nan})).replace([np.inf, -np.inf], np.nan)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    mask = v.notna() & w.notna() & (w > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(v[mask], weights=w[mask]))


def _build_club_context(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["season"] = work["season"].apply(normalize_season_label)
    work["age_num"] = _to_numeric(work, "age")
    work["minutes_num"] = _to_numeric(work, "sofa_minutesPlayed").fillna(0.0)
    work["goals_num"] = _to_numeric(work, "sofa_goals").fillna(0.0)
    work["assists_num"] = _to_numeric(work, "sofa_assists").fillna(0.0)
    work["xg_num"] = _to_numeric(work, "sofa_expectedGoals").fillna(0.0)
    work["shots_num"] = _to_numeric(work, "sofa_totalShots").fillna(0.0)
    work["key_passes_num"] = _to_numeric(work, "sofa_keyPasses").fillna(0.0)
    work["tackles_num"] = _to_numeric(work, "sofa_tackles").fillna(0.0)
    work["interceptions_num"] = _to_numeric(work, "sofa_interceptions").fillna(0.0)
    work["clearances_num"] = _to_numeric(work, "sofa_clearances").fillna(0.0)
    work["duels_num"] = _to_numeric(work, "sofa_totalDuelsWon").fillna(0.0)
    work["pass_pct_num"] = _to_numeric(work, "sofa_accuratePassesPercentage")

    group_keys = ["league", "season", "club"]
    missing = [c for c in group_keys if c not in work.columns]
    if missing:
        raise ValueError(f"Missing required columns for context build: {missing}")

    base = (
        work.groupby(group_keys, dropna=False)
        .agg(
            club_player_rows=("name", "size"),
            club_unique_players=("player_id", "nunique") if "player_id" in work.columns else ("name", "nunique"),
            club_avg_age=("age_num", "mean"),
            club_minutes_sum=("minutes_num", "sum"),
            club_goals_sum=("goals_num", "sum"),
            club_assists_sum=("assists_num", "sum"),
            club_xg_sum=("xg_num", "sum"),
            club_shots_sum=("shots_num", "sum"),
            club_key_passes_sum=("key_passes_num", "sum"),
            club_tackles_sum=("tackles_num", "sum"),
            club_interceptions_sum=("interceptions_num", "sum"),
            club_clearances_sum=("clearances_num", "sum"),
            club_duels_won_sum=("duels_num", "sum"),
        )
        .reset_index()
    )

    pass_pct = (
        work.groupby(group_keys, dropna=False)
        .apply(lambda g: _weighted_mean(g["pass_pct_num"], g["minutes_num"]))
        .reset_index(name="club_weighted_pass_pct")
    )
    base = base.merge(pass_pct, on=group_keys, how="left")

    mins_90 = base["club_minutes_sum"] / 90.0
    base["club_goals_per90"] = _safe_div(base["club_goals_sum"], mins_90)
    base["club_assists_per90"] = _safe_div(base["club_assists_sum"], mins_90)
    base["club_xg_per90"] = _safe_div(base["club_xg_sum"], mins_90)
    base["club_shots_per90"] = _safe_div(base["club_shots_sum"], mins_90)
    base["club_key_passes_per90"] = _safe_div(base["club_key_passes_sum"], mins_90)
    base["club_def_actions_per90"] = _safe_div(
        base["club_tackles_sum"] + base["club_interceptions_sum"] + base["club_clearances_sum"],
        mins_90,
    )
    base["club_duels_won_per90"] = _safe_div(base["club_duels_won_sum"], mins_90)

    # Simple blended proxy that is stable and based only on non-target stats.
    base["club_strength_proxy"] = (
        base["club_goals_per90"].fillna(0.0) * 0.50
        + base["club_xg_per90"].fillna(0.0) * 0.35
        + base["club_key_passes_per90"].fillna(0.0) * 0.10
        + base["club_def_actions_per90"].fillna(0.0) * 0.05
    )

    grp = base.groupby(["league", "season"], dropna=False)["club_strength_proxy"]
    base["club_strength_rank_proxy"] = grp.rank(method="dense", ascending=False)
    base["club_strength_z"] = (
        base["club_strength_proxy"] - grp.transform("mean")
    ) / grp.transform("std").replace({0: np.nan})
    base["club_strength_top3_flag"] = (base["club_strength_rank_proxy"] <= 3).astype(int)

    league_minutes = (
        base.groupby(["league", "season"], dropna=False)["club_minutes_sum"]
        .transform("sum")
        .replace({0: np.nan})
    )
    base["club_minutes_share_in_league"] = base["club_minutes_sum"] / league_minutes

    return base


def _build_league_context(df: pd.DataFrame, club_ctx: pd.DataFrame) -> pd.DataFrame:
    player_level = (
        df.groupby(["league", "season"], dropna=False)
        .agg(
            league_player_rows=("name", "size"),
            league_unique_players=("player_id", "nunique") if "player_id" in df.columns else ("name", "nunique"),
            league_avg_age=("age", lambda s: float(pd.to_numeric(s, errors="coerce").mean())),
        )
        .reset_index()
    )

    league_from_club = (
        club_ctx.groupby(["league", "season"], dropna=False)
        .agg(
            league_club_count=("club", "nunique"),
            league_minutes_sum=("club_minutes_sum", "sum"),
            league_avg_goals_per90=("club_goals_per90", "mean"),
            league_avg_xg_per90=("club_xg_per90", "mean"),
            league_avg_def_actions_per90=("club_def_actions_per90", "mean"),
            league_strength_index=("club_strength_proxy", "mean"),
            league_strength_spread=("club_strength_proxy", "std"),
            league_avg_club_pass_pct=("club_weighted_pass_pct", "mean"),
        )
        .reset_index()
    )

    out = league_from_club.merge(player_level, on=["league", "season"], how="left")
    out["league_competitiveness_index"] = 1.0 / (1.0 + out["league_strength_spread"].fillna(0.0))
    return out


def build_club_and_league_context(
    players_source: str,
    club_output: str,
    league_output: str,
    start_season: str | None = None,
    end_season: str | None = None,
    min_player_minutes: int = 0,
) -> None:
    raw = _read_players_table(Path(players_source))
    if raw.empty:
        raise ValueError("No rows found in players source.")

    if "season" not in raw.columns:
        raise ValueError("Input data must include a 'season' column.")

    work = raw.copy()
    work["season"] = work["season"].apply(normalize_season_label)

    start_norm = normalize_season_label(start_season) if start_season else None
    end_norm = normalize_season_label(end_season) if end_season else None
    if start_norm or end_norm:
        mask = work["season"].astype(str).apply(lambda s: in_season_range(s, start_norm, end_norm))
        work = work[mask].copy()

    if work.empty:
        raise ValueError("No rows remain after season filtering.")

    if min_player_minutes > 0 and "sofa_minutesPlayed" in work.columns:
        mins = pd.to_numeric(work["sofa_minutesPlayed"], errors="coerce").fillna(0.0)
        work = work[mins >= float(min_player_minutes)].copy()
        if work.empty:
            raise ValueError("No rows remain after min-player-minutes filtering.")

    club_ctx = _build_club_context(work)
    league_ctx = _build_league_context(work, club_ctx=club_ctx)

    club_out = Path(club_output)
    club_out.parent.mkdir(parents=True, exist_ok=True)
    club_ctx.to_csv(club_out, index=False)

    league_out = Path(league_output)
    league_out.parent.mkdir(parents=True, exist_ok=True)
    league_ctx.to_csv(league_out, index=False)

    print(f"[context] wrote club context: {len(club_ctx):,} rows -> {club_out}")
    print(f"[context] wrote league context: {len(league_ctx):,} rows -> {league_out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build club-level and league-level context features from merged player-season files."
    )
    parser.add_argument(
        "--players-source",
        default="data/processed/Clubs combined",
        help="Directory with *_with_sofa.csv files, or a single CSV/Parquet file.",
    )
    parser.add_argument("--club-output", default="data/external/club_context.csv")
    parser.add_argument("--league-output", default="data/external/league_context.csv")
    parser.add_argument("--start-season", default=None, help="Optional lower bound, e.g. 2019/20")
    parser.add_argument("--end-season", default=None, help="Optional upper bound, e.g. 2024/25")
    parser.add_argument(
        "--min-player-minutes",
        type=int,
        default=0,
        help="Optional filter before context aggregation.",
    )
    args = parser.parse_args()

    build_club_and_league_context(
        players_source=args.players_source,
        club_output=args.club_output,
        league_output=args.league_output,
        start_season=args.start_season,
        end_season=args.end_season,
        min_player_minutes=args.min_player_minutes,
    )


if __name__ == "__main__":
    main()
