from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from scouting_ml.providers.identity import load_link_table, merge_club_links, normalize_season_label


def _implied_prob(price: object) -> float:
    try:
        val = float(price)
    except (TypeError, ValueError):
        return float("nan")
    if val <= 1.0:
        return float("nan")
    return 1.0 / val


def normalize_odds_events(payload: Any, *, season: str, league: str) -> pd.DataFrame:
    if not isinstance(payload, list):
        return pd.DataFrame()
    rows: list[dict] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        home = item.get("home_team")
        away = item.get("away_team")
        commence = item.get("commence_time")
        bookmakers = item.get("bookmakers") or []
        home_probs: list[float] = []
        away_probs: list[float] = []
        draw_probs: list[float] = []
        totals: list[float] = []
        for book in bookmakers:
            if not isinstance(book, dict):
                continue
            for market in book.get("markets") or []:
                if not isinstance(market, dict):
                    continue
                key = str(market.get("key") or "")
                outcomes = market.get("outcomes") or []
                if key == "h2h":
                    for outcome in outcomes:
                        if not isinstance(outcome, dict):
                            continue
                        name = str(outcome.get("name") or "")
                        prob = _implied_prob(outcome.get("price"))
                        if name == home:
                            home_probs.append(prob)
                        elif name == away:
                            away_probs.append(prob)
                        elif name.lower() == "draw":
                            draw_probs.append(prob)
                elif key == "totals":
                    for outcome in outcomes:
                        if not isinstance(outcome, dict):
                            continue
                        point = outcome.get("point")
                        if point is None:
                            continue
                        try:
                            totals.append(float(point))
                        except (TypeError, ValueError):
                            pass
        rows.append(
            {
                "season": normalize_season_label(season),
                "league": league,
                "match_date": commence,
                "home_team_name": home,
                "away_team_name": away,
                "home_implied_win_prob": float(np.nanmean(home_probs)) if home_probs else np.nan,
                "away_implied_win_prob": float(np.nanmean(away_probs)) if away_probs else np.nan,
                "draw_implied_prob": float(np.nanmean(draw_probs)) if draw_probs else np.nan,
                "bookmaker_count": len(bookmakers),
                "expected_total_goals": float(np.nanmean(totals)) if totals else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["match_date"] = pd.to_datetime(out["match_date"], errors="coerce", utc=True)
    return out


def build_market_context(
    odds_events: pd.DataFrame,
    *,
    club_links_path: str | None = None,
    snapshot_date: str | None = None,
    retrieved_at: str | None = None,
) -> pd.DataFrame:
    if odds_events.empty:
        return pd.DataFrame()
    home = odds_events.rename(
        columns={
            "home_team_name": "team_name",
            "away_team_name": "opponent_name",
            "home_implied_win_prob": "team_win_prob",
            "away_implied_win_prob": "opponent_win_prob",
        }
    ).copy()
    home["is_home"] = 1
    away = odds_events.rename(
        columns={
            "away_team_name": "team_name",
            "home_team_name": "opponent_name",
            "away_implied_win_prob": "team_win_prob",
            "home_implied_win_prob": "opponent_win_prob",
        }
    ).copy()
    away["is_home"] = 0
    team_rows = pd.concat([home, away], ignore_index=True, sort=False)
    team_rows["provider_team_id"] = ""
    links = load_link_table(club_links_path)
    team_rows = merge_club_links(team_rows, links, provider="odds", provider_team_id_col="provider_team_id", team_name_col="team_name")
    out = (
        team_rows.groupby(["league", "season", "team_name", "club"], dropna=False)
        .agg(
            odds_matches=("match_date", "count"),
            odds_home_share=("is_home", "mean"),
            odds_implied_team_strength=("team_win_prob", "mean"),
            odds_implied_opponent_strength=("opponent_win_prob", "mean"),
            odds_draw_probability=("draw_implied_prob", "mean"),
            odds_expected_total_goals=("expected_total_goals", "mean"),
            odds_bookmaker_count=("bookmaker_count", "mean"),
        )
        .reset_index()
    )
    out["odds_upset_probability"] = 1.0 - out["odds_implied_team_strength"]
    out["source_provider"] = "odds"
    out["source_version"] = "snapshot"
    out["retrieved_at"] = retrieved_at
    out["snapshot_date"] = snapshot_date
    out["coverage_note"] = "Market context is season-aggregated from odds snapshots."
    return out
