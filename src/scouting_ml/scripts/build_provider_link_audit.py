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
            return pd.DataFrame()
        return pd.concat([pd.read_csv(file) for file in files], ignore_index=True, sort=False)
    return _read_table(path)


def _non_empty(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().replace({"nan": "", "NaN": "", "None": "", "<NA>": ""})


def _link_coverage_against_players(players: pd.DataFrame, links: pd.DataFrame) -> dict[str, Any]:
    if players.empty or links.empty:
        return {"total_players": int(len(players)), "matched_players": 0, "coverage_0_to_1": 0.0}

    base = players.copy()
    if "season" in base.columns:
        base["season"] = base["season"].apply(normalize_season_label)
    if "player_id" in base.columns:
        base["player_id"] = _non_empty(base["player_id"])
    if "transfermarkt_id" in base.columns:
        base["transfermarkt_id"] = _non_empty(base["transfermarkt_id"])
    base["_norm_name"] = base.get("name", pd.Series("", index=base.index)).apply(normalize_person_name)
    base["_norm_club"] = base.get("club", pd.Series("", index=base.index)).apply(normalize_club_name)

    matched_ids: set[str] = set()
    if "player_id" in links.columns:
        matched_ids.update(_non_empty(links["player_id"]).replace({"": pd.NA}).dropna().tolist())
    if "transfermarkt_id" in links.columns and "transfermarkt_id" in base.columns:
        matched_tm = set(_non_empty(links["transfermarkt_id"]).replace({"": pd.NA}).dropna().tolist())
        matched_ids.update(base.loc[base["transfermarkt_id"].isin(matched_tm), "player_id"].dropna().astype(str).tolist())

    if "provider_player_name" in links.columns:
        link_copy = links.copy()
        link_copy["season"] = link_copy["season"].apply(normalize_season_label) if "season" in link_copy.columns else None
        link_copy["_norm_name"] = link_copy["provider_player_name"].apply(normalize_person_name)
        if "provider_team_name" in link_copy.columns:
            link_copy["_norm_club"] = link_copy["provider_team_name"].apply(normalize_club_name)
            merged = base.merge(
                link_copy[["_norm_name", "_norm_club", "season"]].drop_duplicates(),
                on=["_norm_name", "_norm_club", "season"],
                how="inner",
            )
        else:
            merged = base.merge(link_copy[["_norm_name", "season"]].drop_duplicates(), on=["_norm_name", "season"], how="inner")
        matched_ids.update(merged["player_id"].dropna().astype(str).tolist())

    total_players = int(base.get("player_id", pd.Series(dtype=str)).replace({"": pd.NA}).dropna().nunique())
    matched_players = len({value for value in matched_ids if value})
    return {
        "total_players": total_players,
        "matched_players": int(matched_players),
        "coverage_0_to_1": float(matched_players / total_players) if total_players else 0.0,
    }


def _link_coverage_against_clubs(players: pd.DataFrame, links: pd.DataFrame) -> dict[str, Any]:
    if players.empty or links.empty or "club" not in players.columns:
        return {"total_club_seasons": 0, "matched_club_seasons": 0, "coverage_0_to_1": 0.0}

    base = players.copy()
    base["season"] = base["season"].apply(normalize_season_label)
    base["_norm_club"] = base["club"].apply(normalize_club_name)
    base["_norm_league"] = base.get("league", pd.Series("", index=base.index)).astype(str).str.strip().str.lower()
    base_keys = base[["_norm_club", "_norm_league", "season"]].drop_duplicates()

    link_copy = links.copy()
    link_copy["season"] = link_copy["season"].apply(normalize_season_label) if "season" in link_copy.columns else None
    if "club" in link_copy.columns:
        link_copy["_norm_club"] = link_copy["club"].apply(normalize_club_name)
    elif "provider_team_name" in link_copy.columns:
        link_copy["_norm_club"] = link_copy["provider_team_name"].apply(normalize_club_name)
    else:
        return {"total_club_seasons": int(len(base_keys)), "matched_club_seasons": 0, "coverage_0_to_1": 0.0}
    link_copy["_norm_league"] = link_copy.get("league", pd.Series("", index=link_copy.index)).astype(str).str.strip().str.lower()
    matched = base_keys.merge(link_copy[["_norm_club", "_norm_league", "season"]].drop_duplicates(), on=["_norm_club", "_norm_league", "season"], how="inner")
    total = int(len(base_keys))
    count = int(len(matched))
    return {
        "total_club_seasons": total,
        "matched_club_seasons": count,
        "coverage_0_to_1": float(count / total) if total else 0.0,
    }


def _external_match_stats(path: Path, key_col: str) -> dict[str, Any]:
    frame = _read_table(path)
    if frame.empty:
        return {"path": str(path), "rows": 0, "matched_rows": 0, "coverage_0_to_1": 0.0}
    matched = 0
    if key_col in frame.columns:
        matched = int((_non_empty(frame[key_col]) != "").sum())
    return {
        "path": str(path),
        "rows": int(len(frame)),
        "matched_rows": matched,
        "coverage_0_to_1": float(matched / len(frame)) if len(frame) else 0.0,
        "providers": sorted({str(value) for value in frame.get("source_provider", pd.Series(dtype=str)).dropna().astype(str).tolist()}),
    }


def _season_coverage_against_players(players: pd.DataFrame, frame: pd.DataFrame, *, key_col: str) -> list[dict[str, Any]]:
    if players.empty or "season" not in players.columns:
        return []

    base = players.copy()
    base["season"] = base["season"].apply(normalize_season_label)
    total_by_season = (
        base.groupby("season", dropna=False)
        .size()
        .rename("total_rows")
        .reset_index()
    )

    if frame.empty or "season" not in frame.columns:
        total_by_season["matched_rows"] = 0
        total_by_season["coverage_0_to_1"] = 0.0
        return total_by_season.to_dict(orient="records")

    work = frame.copy()
    work["season"] = work["season"].apply(normalize_season_label)
    if key_col in work.columns:
        work = work[_non_empty(work[key_col]) != ""].copy()
    matched_by_season = (
        work.groupby("season", dropna=False)
        .size()
        .rename("matched_rows")
        .reset_index()
    )
    merged = total_by_season.merge(matched_by_season, on="season", how="left")
    merged["matched_rows"] = merged["matched_rows"].fillna(0).astype(int)
    merged["coverage_0_to_1"] = merged["matched_rows"] / merged["total_rows"].replace({0: pd.NA})
    merged["coverage_0_to_1"] = merged["coverage_0_to_1"].fillna(0.0).astype(float)
    return merged.sort_values("season").to_dict(orient="records")


def build_provider_link_audit(
    *,
    players_source: str,
    external_dir: str = "data/external",
    player_links: str = "data/external/player_provider_links.csv",
    club_links: str = "data/external/club_provider_links.csv",
    out_json: str | None = None,
    out_csv: str | None = None,
) -> dict[str, Any]:
    players = _load_players(Path(players_source))
    ext_dir = Path(external_dir)
    player_link_df = _read_table(Path(player_links))
    club_link_df = _read_table(Path(club_links))

    link_summary: dict[str, Any] = {}
    for provider in sorted({str(value) for value in player_link_df.get("provider", pd.Series(dtype=str)).dropna().astype(str).tolist()}):
        subset = player_link_df[player_link_df["provider"].astype(str) == provider].copy()
        link_summary[f"player_links:{provider}"] = {
            "rows": int(len(subset)),
            **_link_coverage_against_players(players, subset),
        }
    for provider in sorted({str(value) for value in club_link_df.get("provider", pd.Series(dtype=str)).dropna().astype(str).tolist()}):
        subset = club_link_df[club_link_df["provider"].astype(str) == provider].copy()
        link_summary[f"club_links:{provider}"] = {
            "rows": int(len(subset)),
            **_link_coverage_against_clubs(players, subset),
        }

    external_summary = {
        "statsbomb_player_season_features": _external_match_stats(ext_dir / "statsbomb_player_season_features.csv", "player_id"),
        "player_availability": _external_match_stats(ext_dir / "player_availability.csv", "player_id"),
        "fixture_context": _external_match_stats(ext_dir / "fixture_context.csv", "club"),
        "market_context": _external_match_stats(ext_dir / "market_context.csv", "club"),
    }
    season_coverage = {
        "links": {},
        "external": {
            "statsbomb_player_season_features": _season_coverage_against_players(
                players,
                _read_table(ext_dir / "statsbomb_player_season_features.csv"),
                key_col="player_id",
            ),
            "player_availability": _season_coverage_against_players(
                players,
                _read_table(ext_dir / "player_availability.csv"),
                key_col="player_id",
            ),
        },
    }
    for provider in sorted({str(value) for value in player_link_df.get("provider", pd.Series(dtype=str)).dropna().astype(str).tolist()}):
        subset = player_link_df[player_link_df["provider"].astype(str) == provider].copy()
        season_coverage["links"][f"player_links:{provider}"] = _season_coverage_against_players(
            players,
            subset,
            key_col="player_id",
        )

    audit = {
        "players_source": str(players_source),
        "external_dir": str(ext_dir),
        "player_links": str(player_links),
        "club_links": str(club_links),
        "link_summary": link_summary,
        "external_summary": external_summary,
        "season_coverage": season_coverage,
    }

    if out_json:
        out_path = Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    if out_csv:
        rows: list[dict[str, Any]] = []
        for name, payload in link_summary.items():
            rows.append({"section": "links", "name": name, **payload})
        for name, payload in external_summary.items():
            rows.append({"section": "external", "name": name, **payload})
        for name, payload in season_coverage["links"].items():
            for season_row in payload:
                rows.append({"section": "season_links", "name": name, **season_row})
        for name, payload in season_coverage["external"].items():
            for season_row in payload:
                rows.append({"section": "season_external", "name": name, **season_row})
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)

    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit provider link tables and external provider coverage.")
    parser.add_argument("--players-source", required=True)
    parser.add_argument("--external-dir", default="data/external")
    parser.add_argument("--player-links", default="data/external/player_provider_links.csv")
    parser.add_argument("--club-links", default="data/external/club_provider_links.csv")
    parser.add_argument("--out-json", default="data/external/provider_link_audit.json")
    parser.add_argument("--out-csv", default="data/external/provider_link_audit.csv")
    args = parser.parse_args()

    payload = build_provider_link_audit(
        players_source=args.players_source,
        external_dir=args.external_dir,
        player_links=args.player_links,
        club_links=args.club_links,
        out_json=args.out_json,
        out_csv=args.out_csv,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
