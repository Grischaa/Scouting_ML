from __future__ import annotations

import pandas as pd

from scouting_ml.tm.merge_tm_sofa import merge_tm_sofa


def test_merge_tm_sofa_handles_blank_sofa_csv(tmp_path) -> None:
    tm_path = tmp_path / "league_clean.csv"
    sofa_path = tmp_path / "sofa_blank.csv"
    out_path = tmp_path / "league_with_sofa.csv"

    pd.DataFrame(
        [
            {"player_id": "p1", "name": "Jan Kowalski", "club": "Legia"},
            {"player_id": "p2", "name": "Piotr Nowak", "club": "Lech"},
        ]
    ).to_csv(tm_path, index=False)
    sofa_path.write_text("", encoding="utf-8")

    merge_tm_sofa(
        tm_path=str(tm_path),
        sofa_path=str(sofa_path),
        out_path=str(out_path),
    )

    merged = pd.read_csv(out_path)
    assert len(merged) == 2
    assert merged["sofa_matched"].sum() == 0
    assert "sofa_player_id" in merged.columns
    assert merged["sofa_player_id"].isna().all()
