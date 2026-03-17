from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

TEXT_RE = re.compile(r"[^a-z0-9]+")
CLUB_STOPWORDS = {
    "fc",
    "cf",
    "club",
    "ac",
    "sc",
    "sv",
    "fk",
    "afc",
    "the",
}


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii").lower().strip()
    return TEXT_RE.sub(" ", text).strip()


def normalize_person_name(value: object) -> str:
    return normalize_text(value)


def normalize_club_name(value: object) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    tokens = [tok for tok in text.split() if tok not in CLUB_STOPWORDS]
    return " ".join(tokens) if tokens else text


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
        return f"{start}/{end[-2:]}"
    m_short = re.match(r"^(\d{2})/(\d{2})$", season)
    if m_short:
        start2 = int(m_short.group(1))
        start = 2000 + start2 if start2 <= 69 else 1900 + start2
        return f"{start}/{m_short.group(2)}"
    m_year = re.match(r"^(\d{4})$", season)
    if m_year:
        year = int(m_year.group(1))
        return f"{year-1}/{str(year)[-2:]}"
    return season


def load_link_table(path: Path | str | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    frame = pd.read_csv(p)
    if "season" in frame.columns:
        frame["season"] = frame["season"].apply(normalize_season_label)
    if "player_name" in frame.columns:
        frame["_norm_player_name"] = frame["player_name"].apply(normalize_person_name)
    if "provider_player_name" in frame.columns:
        frame["_norm_provider_player_name"] = frame["provider_player_name"].apply(normalize_person_name)
    if "club" in frame.columns:
        frame["_norm_club"] = frame["club"].apply(normalize_club_name)
    if "provider_team_name" in frame.columns:
        frame["_norm_provider_team_name"] = frame["provider_team_name"].apply(normalize_club_name)
    if "league" in frame.columns:
        frame["_norm_league"] = frame["league"].apply(normalize_text)
    return frame


def _coerce_id_series(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series("", index=index, dtype="object")
    return (
        series.astype(str)
        .str.strip()
        .replace({"nan": "", "NaN": "", "None": "", "<NA>": ""})
        .fillna("")
    )


def _coalesce_duplicate_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        left = f"{col}_x"
        right = f"{col}_y"
        link = f"{col}_link"
        if left in out.columns or right in out.columns:
            base = out[left] if left in out.columns else pd.Series(pd.NA, index=out.index)
            if right in out.columns:
                base = base.fillna(out[right])
            if link in out.columns:
                base = base.fillna(out[link])
            out[col] = base
    return out


def merge_player_links(
    frame: pd.DataFrame,
    links: pd.DataFrame,
    *,
    provider: str,
    provider_id_col: str = "provider_player_id",
    player_name_col: str = "player_name",
    club_col: str | None = "club",
    season_col: str = "season",
) -> pd.DataFrame:
    out = frame.copy()
    out[season_col] = out[season_col].apply(normalize_season_label)
    out["_provider_key"] = _coerce_id_series(out.get(provider_id_col), out.index)
    out["_norm_player_name"] = out[player_name_col].apply(normalize_person_name) if player_name_col in out.columns else ""
    out["_norm_club"] = out[club_col].apply(normalize_club_name) if club_col and club_col in out.columns else ""

    if links.empty:
        return out.drop(columns=["_provider_key", "_norm_player_name", "_norm_club"], errors="ignore")

    link = links.copy()
    if "provider" in link.columns:
        link = link[link["provider"].astype(str).str.strip().str.lower() == provider.lower()].copy()
    if link.empty:
        return out.drop(columns=["_provider_key", "_norm_player_name", "_norm_club"], errors="ignore")

    if "provider_player_id" in link.columns:
        link["_provider_key"] = _coerce_id_series(link["provider_player_id"], link.index)
        keys = ["_provider_key", "season"]
        if all(k in link.columns for k in keys) and link["_provider_key"].ne("").any() and out["_provider_key"].ne("").any():
            use_cols = [c for c in ["player_id", "transfermarkt_id", "dob"] if c in link.columns]
            out = out.merge(link[keys + use_cols].drop_duplicates(keys, keep="last"), on=keys, how="left")
            out = _coalesce_duplicate_columns(out, ["player_id", "transfermarkt_id", "dob"])

    if "player_id" not in out.columns or out["player_id"].isna().any():
        if "_norm_provider_player_name" not in link.columns and "provider_player_name" in link.columns:
            link["_norm_provider_player_name"] = link["provider_player_name"].apply(normalize_person_name)
        right_name = "_norm_provider_player_name" if "_norm_provider_player_name" in link.columns else "_norm_player_name"
        if club_col and "_norm_provider_team_name" in link.columns:
            merged = out.merge(
                link[[right_name, "_norm_provider_team_name", "season"] + [c for c in ["player_id", "transfermarkt_id", "dob"] if c in link.columns]]
                .drop_duplicates([right_name, "_norm_provider_team_name", "season"], keep="last"),
                left_on=["_norm_player_name", "_norm_club", "season"],
                right_on=[right_name, "_norm_provider_team_name", "season"],
                how="left",
                suffixes=("", "_link"),
            )
        else:
            merged = out.merge(
                link[[right_name, "season"] + [c for c in ["player_id", "transfermarkt_id", "dob"] if c in link.columns]]
                .drop_duplicates([right_name, "season"], keep="last"),
                left_on=["_norm_player_name", "season"],
                right_on=[right_name, "season"],
                how="left",
                suffixes=("", "_link"),
            )
        for col in ["player_id", "transfermarkt_id", "dob"]:
            link_col = f"{col}_link"
            if link_col in merged.columns:
                if col in merged.columns:
                    merged[col] = merged[col].fillna(merged[link_col])
                else:
                    merged[col] = merged[link_col]
        out = merged

    return out.drop(
        columns=[
            "_provider_key",
            "_norm_player_name",
            "_norm_club",
            "_norm_provider_player_name",
            "_norm_provider_team_name",
            "player_id_x",
            "player_id_y",
            "player_id_link",
            "transfermarkt_id_x",
            "transfermarkt_id_y",
            "transfermarkt_id_link",
            "dob_x",
            "dob_y",
            "dob_link",
        ],
        errors="ignore",
    )


def merge_club_links(
    frame: pd.DataFrame,
    links: pd.DataFrame,
    *,
    provider: str,
    provider_team_id_col: str = "provider_team_id",
    team_name_col: str = "team_name",
    season_col: str = "season",
    league_col: str | None = "league",
) -> pd.DataFrame:
    out = frame.copy()
    out[season_col] = out[season_col].apply(normalize_season_label)
    out["_provider_team_key"] = _coerce_id_series(out.get(provider_team_id_col), out.index)
    out["_norm_team_name"] = out[team_name_col].apply(normalize_club_name) if team_name_col in out.columns else ""
    if league_col and league_col in out.columns:
        out["_norm_league"] = out[league_col].apply(normalize_text)

    if links.empty:
        return out.drop(columns=["_provider_team_key", "_norm_team_name", "_norm_league"], errors="ignore")

    link = links.copy()
    if "provider" in link.columns:
        link = link[link["provider"].astype(str).str.strip().str.lower() == provider.lower()].copy()
    if link.empty:
        return out.drop(columns=["_provider_team_key", "_norm_team_name", "_norm_league"], errors="ignore")

    if "provider_team_id" in link.columns:
        link["_provider_team_key"] = _coerce_id_series(link["provider_team_id"], link.index)
        keys = ["_provider_team_key", "season"]
        if all(k in link.columns for k in keys) and link["_provider_team_key"].ne("").any() and out["_provider_team_key"].ne("").any():
            use_cols = [c for c in ["club", "league"] if c in link.columns]
            out = out.merge(link[keys + use_cols].drop_duplicates(keys, keep="last"), on=keys, how="left")
            out = _coalesce_duplicate_columns(out, ["club", "league"])

    if "club" not in out.columns or out["club"].isna().any():
        if "_norm_provider_team_name" not in link.columns and "provider_team_name" in link.columns:
            link["_norm_provider_team_name"] = link["provider_team_name"].apply(normalize_club_name)
        join_right = ["_norm_provider_team_name", "season"]
        join_left = ["_norm_team_name", "season"]
        if league_col and "_norm_league" in link.columns and "_norm_league" in out.columns:
            join_right.append("_norm_league")
            join_left.append("_norm_league")
        use_cols = join_right + [c for c in ["club", "league"] if c in link.columns]
        merged = out.merge(link[use_cols].drop_duplicates(join_right, keep="last"), left_on=join_left, right_on=join_right, how="left", suffixes=("", "_link"))
        for col in ["club", "league"]:
            link_col = f"{col}_link"
            if link_col in merged.columns:
                if col in merged.columns:
                    merged[col] = merged[col].fillna(merged[link_col])
                else:
                    merged[col] = merged[link_col]
        out = merged

    return out.drop(
        columns=[
            "_provider_team_key",
            "_norm_team_name",
            "_norm_league",
            "_norm_provider_team_name",
            "club_x",
            "club_y",
            "club_link",
            "league_x",
            "league_y",
            "league_link",
        ],
        errors="ignore",
    )
