# src/scouting_ml/merge_tm_sofa.py
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

from scouting_ml.reporting.operator_health import latest_timestamp, write_json_sidecar
from scouting_ml.utils.import_guard import *  # noqa: F403

EMPTY_SOFA_INPUT_COLUMNS = ["player", "team", "player id", "team id"]
REQUIRED_SOFA_COLUMNS = ["player_name", "team_name", "sofa_player_id", "sofa_team_id"]
PROVIDER_SNAPSHOT_COLUMNS = {
    "sb_snapshot_date": "sb",
    "avail_snapshot_date": "avail",
    "fixture_snapshot_date": "fixture",
    "odds_snapshot_date": "odds",
}


def _norm_name(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    s = re.sub(r"[.]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _read_sofa_frame(path: str | Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=EMPTY_SOFA_INPUT_COLUMNS)
    if len(frame.columns) == 0:
        return pd.DataFrame(columns=EMPTY_SOFA_INPUT_COLUMNS)
    return frame


def _ensure_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.Series(dtype="object")
    return frame


def merge_tm_sofa(
    tm_path: str = "data/processed/austrian_bundesliga_2025-26_clean.csv",
    sofa_path: str = "data/processed/sofa_austrian_bundesliga_25-26.csv",
    out_path: str = "data/processed/austrian_bundesliga_2025-26_with_sofa.csv",
) -> None:
    tm_df = pd.read_csv(tm_path)
    sofa_df = _read_sofa_frame(sofa_path)

    tm_df["name_norm"] = tm_df["name"].apply(_norm_name)
    if "club" in tm_df.columns:
        tm_df["club_norm"] = tm_df["club"].apply(_norm_name)
    else:
        tm_df["club_norm"] = ""

    rename_map = {
        "player": "player_name",
        "team": "team_name",
        "player id": "sofa_player_id",
        "team id": "sofa_team_id",
    }
    sofa_df = sofa_df.rename(columns={k: v for k, v in rename_map.items() if k in sofa_df.columns})
    sofa_df = _ensure_columns(sofa_df, REQUIRED_SOFA_COLUMNS)

    sofa_df["player_name_norm"] = sofa_df["player_name"].apply(_norm_name)
    sofa_df["team_name_norm"] = sofa_df["team_name"].apply(_norm_name)
    sofa_df["sofa_team_name"] = sofa_df["team_name"]
    sofa_df = sofa_df.drop(columns=["team_name"])

    base_cols = {"player_name", "player_name_norm", "team_name_norm", "sofa_team_name"}
    value_cols = [c for c in sofa_df.columns if c not in base_cols]
    rename_prefixed: dict[str, str] = {}
    for col in value_cols:
        if not col.startswith("sofa_"):
            safe = col.replace(" ", "_")
            rename_prefixed[col] = f"sofa_{safe}"
    if rename_prefixed:
        sofa_df = sofa_df.rename(columns=rename_prefixed)

    sofa_value_cols = [c for c in sofa_df.columns if c.startswith("sofa_") and c not in {"sofa_team_name"}]

    merged = tm_df.merge(
        sofa_df,
        left_on=["name_norm", "club_norm"],
        right_on=["player_name_norm", "team_name_norm"],
        how="left",
    )

    mask_unmatched = merged["sofa_player_id"].isna()
    if mask_unmatched.any():
        fallback = sofa_df.drop_duplicates("player_name_norm")
        fallback = fallback[["player_name_norm", "team_name_norm", "sofa_team_name"] + sofa_value_cols]
        tmp = (
            merged.loc[mask_unmatched, ["name_norm"]]
            .reset_index()
            .merge(
                fallback,
                left_on="name_norm",
                right_on="player_name_norm",
                how="left",
            )
            .groupby("index", as_index=False)
            .first()
            .set_index("index")
        )
        for col in ["team_name_norm", "sofa_team_name"] + sofa_value_cols:
            updates = pd.Series(tmp[col], index=tmp.index)
            merged.loc[tmp.index, col] = updates.values

    merged["sofa_matched"] = merged["sofa_player_id"].notna().astype(int)
    merged = merged.drop(columns=["name_norm", "club_norm", "player_name_norm", "team_name_norm"], errors="ignore")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    provider_snapshot_dates = {
        alias: latest_timestamp(merged[col].tolist()) if col in merged.columns else None
        for col, alias in PROVIDER_SNAPSHOT_COLUMNS.items()
    }
    match_total = int(merged["sofa_matched"].sum()) if "sofa_matched" in merged.columns else 0
    write_json_sidecar(
        out_path,
        {
            "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
            "tm_rows": int(len(tm_df)),
            "sofa_rows": int(len(sofa_df)),
            "matched_rows": match_total,
            "match_rate": (float(match_total) / float(len(tm_df))) if len(tm_df) else None,
            "sofa_zero_rows": bool(sofa_df.empty),
            "provider_snapshot_dates": provider_snapshot_dates,
            "columns": [str(col) for col in merged.columns],
        },
    )
    print(f"[merge_tm_sofa] Wrote {len(merged)} rows -> {out_path.resolve()}")
    print(f"[merge_tm_sofa] Matched {merged['sofa_matched'].sum()} / {len(merged)} players")
    if sofa_df.empty:
        print("[merge_tm_sofa] Warning: Sofascore input had no player rows; merged file contains TM rows with sofa_matched=0.")


def main() -> None:
    merge_tm_sofa()


if __name__ == "__main__":
    main()
