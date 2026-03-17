from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scouting_ml.providers.identity import normalize_club_name, normalize_person_name, normalize_season_label


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _load_players(path: Path) -> pd.DataFrame:
    if path.is_dir():
        files = sorted(path.rglob("*_with_sofa.csv"))
        if not files:
            raise ValueError(f"No *_with_sofa.csv files found in {path}")
        return pd.concat([pd.read_csv(file) for file in files], ignore_index=True, sort=False)
    return _read_table(path)


def _safe_series(frame: pd.DataFrame, col: str) -> pd.Series:
    if col in frame.columns:
        return frame[col]
    return pd.Series("", index=frame.index, dtype="object")


def _non_empty_mask(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .replace({"nan": "", "NaN": "", "None": "", "<NA>": ""})
        .ne("")
    )


def _prepare_players(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["season"] = _safe_series(out, "season").apply(normalize_season_label)
    out["player_id"] = _safe_series(out, "player_id").astype(str).str.strip()
    out["transfermarkt_id"] = _safe_series(out, "transfermarkt_id").astype(str).str.strip()
    out["name"] = _safe_series(out, "name").astype(str)
    out["club"] = _safe_series(out, "club").astype(str)
    out["league"] = _safe_series(out, "league").astype(str)
    out["dob"] = _safe_series(out, "dob").astype(str)
    out["_norm_name"] = out["name"].apply(normalize_person_name)
    out["_norm_club"] = out["club"].apply(normalize_club_name)
    out["_norm_league"] = out["league"].astype(str).str.strip().str.lower()
    return out


def _candidate_player_rows(provider: str, path: Path) -> pd.DataFrame:
    frame = _read_table(path)
    if frame.empty:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "provider": provider,
            "provider_player_id": _safe_series(frame, "provider_player_id").astype(str).str.strip(),
            "provider_player_name": _safe_series(frame, "player_name").astype(str),
            "provider_team_id": _safe_series(frame, "provider_team_id").astype(str).str.strip(),
            "provider_team_name": _safe_series(frame, "team_name").astype(str),
            "league": _safe_series(frame, "league").astype(str),
            "season": _safe_series(frame, "season").apply(normalize_season_label),
        }
    )
    out = out[(out["provider_player_name"].str.strip() != "") & (out["season"].notna())].drop_duplicates()
    out["_norm_name"] = out["provider_player_name"].apply(normalize_person_name)
    out["_norm_club"] = out["provider_team_name"].apply(normalize_club_name)
    out["_norm_league"] = out["league"].astype(str).str.strip().str.lower()
    return out


def _candidate_club_rows(provider: str, path: Path) -> pd.DataFrame:
    frame = _read_table(path)
    if frame.empty:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "provider": provider,
            "provider_team_id": _safe_series(frame, "provider_team_id").astype(str).str.strip(),
            "provider_team_name": _safe_series(frame, "team_name").astype(str),
            "league": _safe_series(frame, "league").astype(str),
            "season": _safe_series(frame, "season").apply(normalize_season_label),
        }
    )
    out = out[(out["provider_team_name"].str.strip() != "") & (out["season"].notna())].drop_duplicates()
    out["_norm_club"] = out["provider_team_name"].apply(normalize_club_name)
    out["_norm_league"] = out["league"].astype(str).str.strip().str.lower()
    return out


def _match_players(base: pd.DataFrame, candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame()
    strict = candidates.merge(
        base[
            [
                "player_id",
                "transfermarkt_id",
                "name",
                "club",
                "league",
                "dob",
                "season",
                "_norm_name",
                "_norm_club",
                "_norm_league",
            ]
        ],
        on=["_norm_name", "_norm_club", "season"],
        how="left",
        suffixes=("", "_base"),
    )
    strict["league_match"] = (strict["_norm_league"] == strict["_norm_league_base"]).astype(int)
    strict["match_confidence"] = 0.85 + (strict["league_match"] * 0.1)
    strict["match_method"] = strict["league_match"].map({1: "name_club_season_league", 0: "name_club_season"})
    strict_has_player = _non_empty_mask(strict["player_id"])
    matched = strict[strict_has_player].copy()
    matched = matched.sort_values(["provider", "provider_player_id", "match_confidence"], ascending=[True, True, False])
    matched = matched.drop_duplicates(["provider", "provider_player_id", "season"], keep="first")

    unmatched = strict[~strict_has_player].copy()
    unresolved_keys = unmatched[["provider", "provider_player_id", "season", "_norm_name", "_norm_league"]].drop_duplicates()
    if not unresolved_keys.empty:
        base_unique_name_league = (
            base.groupby(["_norm_name", "_norm_league", "season"], dropna=False)
            .filter(lambda grp: len(grp) == 1)
            [[
                "player_id",
                "transfermarkt_id",
                "name",
                "club",
                "league",
                "dob",
                "season",
                "_norm_name",
                "_norm_league",
            ]]
        )
        fallback = unresolved_keys.merge(
            base_unique_name_league,
            on=["_norm_name", "_norm_league", "season"],
            how="left",
        )
        fallback["match_confidence"] = 0.74
        fallback["match_method"] = "name_league_season_unique"
        fallback = fallback[_non_empty_mask(fallback["player_id"])]
        if not fallback.empty:
            matched = pd.concat([matched, fallback], ignore_index=True, sort=False)
            matched = matched.sort_values(["provider", "provider_player_id", "match_confidence"], ascending=[True, True, False])
            matched = matched.drop_duplicates(["provider", "provider_player_id", "season"], keep="first")

    unresolved_keys = candidates.merge(
        matched[["provider", "provider_player_id", "season"]].drop_duplicates(),
        on=["provider", "provider_player_id", "season"],
        how="left",
        indicator=True,
    )
    unresolved_keys = unresolved_keys[unresolved_keys["_merge"] == "left_only"].drop(columns=["_merge"])
    if not unresolved_keys.empty:
        base_unique_name = (
            base.groupby(["_norm_name", "season"], dropna=False)
            .filter(lambda grp: len(grp) == 1)
            [[
                "player_id",
                "transfermarkt_id",
                "name",
                "club",
                "league",
                "dob",
                "season",
                "_norm_name",
            ]]
        )
        fallback = unresolved_keys.merge(base_unique_name, on=["_norm_name", "season"], how="left")
        fallback["match_confidence"] = 0.66
        fallback["match_method"] = "name_season_unique"
        fallback = fallback[_non_empty_mask(fallback["player_id"])]
        if not fallback.empty:
            matched = pd.concat([matched, fallback], ignore_index=True, sort=False)
            matched = matched.sort_values(["provider", "provider_player_id", "match_confidence"], ascending=[True, True, False])
            matched = matched.drop_duplicates(["provider", "provider_player_id", "season"], keep="first")

    unmatched = candidates.merge(
        matched[["provider", "provider_player_id", "season"]].drop_duplicates(),
        on=["provider", "provider_player_id", "season"],
        how="left",
        indicator=True,
    )
    unmatched = unmatched[unmatched["_merge"] == "left_only"].drop(columns=["_merge"])
    return matched, unmatched


def _match_clubs(base: pd.DataFrame, candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame()
    club_base = base[["club", "league", "season", "_norm_club", "_norm_league"]].drop_duplicates()
    merged = candidates.merge(
        club_base,
        on=["_norm_club", "season"],
        how="left",
        suffixes=("", "_base"),
    )
    merged["league_match"] = (merged["_norm_league"] == merged["_norm_league_base"]).astype(int)
    merged["match_confidence"] = 0.9 + (merged["league_match"] * 0.05)
    merged["match_method"] = merged["league_match"].map({1: "club_season_league", 0: "club_season"})
    merged_has_club = _non_empty_mask(merged["club"])
    matched = merged[merged_has_club].copy()
    matched = matched.sort_values(["provider", "provider_team_name", "match_confidence"], ascending=[True, True, False])
    matched = matched.drop_duplicates(["provider", "provider_team_name", "season"], keep="first")
    unmatched = merged[~merged_has_club].copy()
    return matched, unmatched


def bootstrap_provider_links(
    *,
    players_source: str,
    external_dir: str = "data/external",
    player_links_out: str = "data/external/player_provider_links.csv",
    club_links_out: str = "data/external/club_provider_links.csv",
    statsbomb_input: str | None = None,
    availability_input: str | None = None,
    fixture_input: str | None = None,
    market_input: str | None = None,
    review_confidence_threshold: float = 0.80,
) -> dict[str, Any]:
    base = _prepare_players(_load_players(Path(players_source)))
    ext_dir = Path(external_dir)
    provider_sources = {
        "statsbomb": Path(statsbomb_input) if statsbomb_input else ext_dir / "statsbomb_player_season_features.csv",
        "sportmonks_availability": Path(availability_input) if availability_input else ext_dir / "player_availability.csv",
        "sportmonks_fixture": Path(fixture_input) if fixture_input else ext_dir / "fixture_context.csv",
        "odds": Path(market_input) if market_input else ext_dir / "market_context.csv",
    }

    player_candidates = pd.concat(
        [
            _candidate_player_rows("statsbomb", provider_sources["statsbomb"]),
            _candidate_player_rows("sportmonks", provider_sources["sportmonks_availability"]),
        ],
        ignore_index=True,
        sort=False,
    )
    club_candidates = pd.concat(
        [
            _candidate_club_rows("sportmonks", provider_sources["sportmonks_fixture"]),
            _candidate_club_rows("odds", provider_sources["odds"]),
        ],
        ignore_index=True,
        sort=False,
    )

    matched_players, unmatched_players = _match_players(base, player_candidates)
    matched_clubs, unmatched_clubs = _match_clubs(base, club_candidates)
    review_confidence_threshold = float(review_confidence_threshold)

    strict_player_methods = {"name_club_season_league", "name_club_season"}
    strict_club_methods = {"club_season_league", "club_season"}

    player_out = Path(player_links_out)
    player_out.parent.mkdir(parents=True, exist_ok=True)
    keep_player_cols = [
        "provider",
        "provider_player_id",
        "provider_player_name",
        "provider_team_id",
        "provider_team_name",
        "season",
        "player_id",
        "transfermarkt_id",
        "name",
        "club",
        "league",
        "dob",
        "match_confidence",
        "match_method",
    ]
    if matched_players.empty:
        pd.DataFrame(columns=keep_player_cols).to_csv(player_out, index=False)
    else:
        matched_players.reindex(columns=keep_player_cols).to_csv(player_out, index=False)

    club_out = Path(club_links_out)
    club_out.parent.mkdir(parents=True, exist_ok=True)
    keep_club_cols = [
        "provider",
        "provider_team_id",
        "provider_team_name",
        "season",
        "club",
        "league",
        "match_confidence",
        "match_method",
    ]
    if matched_clubs.empty:
        pd.DataFrame(columns=keep_club_cols).to_csv(club_out, index=False)
    else:
        matched_clubs.reindex(columns=keep_club_cols).to_csv(club_out, index=False)

    unmatched_players_path = player_out.with_name("player_provider_link_candidates_unmatched.csv")
    unmatched_clubs_path = club_out.with_name("club_provider_link_candidates_unmatched.csv")
    unmatched_players.to_csv(unmatched_players_path, index=False)
    unmatched_clubs.to_csv(unmatched_clubs_path, index=False)

    player_confidence = (
        pd.to_numeric(matched_players["match_confidence"], errors="coerce")
        if "match_confidence" in matched_players
        else pd.Series(0.0, index=matched_players.index, dtype="float64")
    ).fillna(0.0)
    player_method = (
        matched_players["match_method"].astype(str)
        if "match_method" in matched_players
        else pd.Series("", index=matched_players.index, dtype="object")
    )
    club_confidence = (
        pd.to_numeric(matched_clubs["match_confidence"], errors="coerce")
        if "match_confidence" in matched_clubs
        else pd.Series(0.0, index=matched_clubs.index, dtype="float64")
    ).fillna(0.0)
    club_method = (
        matched_clubs["match_method"].astype(str)
        if "match_method" in matched_clubs
        else pd.Series("", index=matched_clubs.index, dtype="object")
    )

    player_review = matched_players[
        (player_confidence < review_confidence_threshold)
        | (~player_method.isin(strict_player_methods))
    ].copy()
    club_review = matched_clubs[
        (club_confidence < review_confidence_threshold)
        | (~club_method.isin(strict_club_methods))
    ].copy()
    player_review_path = player_out.with_name("player_provider_link_review_queue.csv")
    club_review_path = club_out.with_name("club_provider_link_review_queue.csv")
    player_review.reindex(columns=keep_player_cols).to_csv(player_review_path, index=False)
    club_review.reindex(columns=keep_club_cols).to_csv(club_review_path, index=False)

    return {
        "players_source": players_source,
        "player_links_out": str(player_out),
        "club_links_out": str(club_out),
        "player_links_rows": int(len(matched_players)),
        "club_links_rows": int(len(matched_clubs)),
        "unmatched_player_candidates": int(len(unmatched_players)),
        "unmatched_club_candidates": int(len(unmatched_clubs)),
        "player_review_queue_rows": int(len(player_review)),
        "club_review_queue_rows": int(len(club_review)),
        "player_review_queue_path": str(player_review_path),
        "club_review_queue_path": str(club_review_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap provider link tables from local processed data and provider outputs.")
    parser.add_argument("--players-source", required=True)
    parser.add_argument("--external-dir", default="data/external")
    parser.add_argument("--player-links-out", default="data/external/player_provider_links.csv")
    parser.add_argument("--club-links-out", default="data/external/club_provider_links.csv")
    parser.add_argument("--statsbomb-input", default="")
    parser.add_argument("--availability-input", default="")
    parser.add_argument("--fixture-input", default="")
    parser.add_argument("--market-input", default="")
    parser.add_argument("--review-confidence-threshold", type=float, default=0.80)
    args = parser.parse_args()

    payload = bootstrap_provider_links(
        players_source=args.players_source,
        external_dir=args.external_dir,
        player_links_out=args.player_links_out,
        club_links_out=args.club_links_out,
        statsbomb_input=args.statsbomb_input or None,
        availability_input=args.availability_input or None,
        fixture_input=args.fixture_input or None,
        market_input=args.market_input or None,
        review_confidence_threshold=args.review_confidence_threshold,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
