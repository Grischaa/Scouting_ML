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
    # remove accents
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    # collapse spaces, drop dots
    s = re.sub(r"[.]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def merge_tm_sofa(
    tm_path: str = "data/processed/austrian_bundesliga_2025-26_clean.csv",
    sofa_path: str = "data/processed/sofa_players_seasons.csv",
    out_path: str = "data/processed/austrian_bundesliga_2025-26_with_sofa.csv",
) -> None:
    tm_df = pd.read_csv(tm_path)
    sofa_df = pd.read_csv(sofa_path)

    # add normalized names
    tm_df["name_norm"] = tm_df["name"].apply(_norm_name)
    if "club" in tm_df.columns:
        tm_df["club_norm"] = tm_df["club"].apply(_norm_name)
    else:
        tm_df["club_norm"] = ""

    sofa_df["player_name_norm"] = sofa_df["player_name"].apply(_norm_name)

    # ----- first pass: name + club -----
    merged = tm_df.merge(
        sofa_df,
        left_on=["name_norm", "club_norm"],
        right_on=["player_name_norm", "player_name_norm"],  # club often missing in Sofascore season files
        how="left",
        suffixes=("", "_sofa"),
    )

    # that was too naive — Sofascore season files don’t have team by default.
    # So a better way is: join just on name_norm, then later you can filter manually.
    # We'll do a second pass for rows that stayed empty.

    mask_unmatched = merged["sofa_player_id"].isna()
    if mask_unmatched.any():
        # join only on name
        tmp = tm_df.loc[mask_unmatched, ["name_norm"]].merge(
            sofa_df[["player_name_norm", "sofa_player_id", "season_id", "appearances", "minutes", "goals", "assists", "rating"]],
            left_on="name_norm",
            right_on="player_name_norm",
            how="left",
        )
        # now fill back
        merged.loc[mask_unmatched, "sofa_player_id"] = tmp["sofa_player_id"].values
        merged.loc[mask_unmatched, "sofa_season_id"] = tmp["season_id"].values
        merged.loc[mask_unmatched, "sofa_appearances"] = tmp["appearances"].values
        merged.loc[mask_unmatched, "sofa_minutes"] = tmp["minutes"].values
        merged.loc[mask_unmatched, "sofa_goals"] = tmp["goals"].values
        merged.loc[mask_unmatched, "sofa_assists"] = tmp["assists"].values
        merged.loc[mask_unmatched, "sofa_rating"] = tmp["rating"].values

    # final tidy columns
    merged["sofa_matched"] = merged["sofa_player_id"].notna().astype(int)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"[merge_tm_sofa] Wrote {len(merged)} rows -> {out_path.resolve()}")
    print(f"[merge_tm_sofa] Matched {merged['sofa_matched'].sum()} / {len(merged)} players")


def main():
    merge_tm_sofa()


if __name__ == "__main__":
    main()
