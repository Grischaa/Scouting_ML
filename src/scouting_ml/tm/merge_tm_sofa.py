# src/scouting_ml/merge_tm_sofa.py
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from scouting_ml.utils.import_guard import *  # noqa: F403
import pandas as pd


def _norm_name(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    s = re.sub(r"[.]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def merge_tm_sofa(
    tm_path: str = "data/processed/austrian_bundesliga_2025-26_clean.csv",
    sofa_path: str = "data/processed/sofa_austrian_bundesliga_25-26.csv",
    out_path: str = "data/processed/austrian_bundesliga_2025-26_with_sofa.csv",
) -> None:
    tm_df = pd.read_csv(tm_path)
    sofa_df = pd.read_csv(sofa_path)

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
    print(f"[merge_tm_sofa] Wrote {len(merged)} rows -> {out_path.resolve()}")
    print(f"[merge_tm_sofa] Matched {merged['sofa_matched'].sum()} / {len(merged)} players")


def main() -> None:
    merge_tm_sofa()


if __name__ == "__main__":
    main()
