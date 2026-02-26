from __future__ import annotations

import json

import pandas as pd

from scouting_ml.features.history_strength import add_history_strength_features
from scouting_ml.scripts.build_player_transfers import parse_transfer_rows


def test_history_strength_score_ranks_stable_profile_higher() -> None:
    frame = pd.DataFrame(
        [
            {
                "minutes": 2500,
                "transfer_count_3y": 1,
                "transfer_loans_3y": 0,
                "transfer_paid_moves_3y": 1,
                "transfer_total_fees_3y_eur": 25_000_000,
                "transfer_max_fee_career_eur": 35_000_000,
                "contract_years_left": 3.5,
                "injury_days_per_1000_min": 9.0,
                "nt_total_caps": 18,
                "delta_minutes": 220,
                "delta_sofa_expectedGoals_per90": 0.03,
                "delta_sofa_assists_per90": 0.02,
                "delta_sofa_goals_per90": 0.01,
            },
            {
                "minutes": 780,
                "transfer_count_3y": 5,
                "transfer_loans_3y": 3,
                "transfer_paid_moves_3y": 0,
                "transfer_total_fees_3y_eur": 0,
                "transfer_max_fee_career_eur": 2_000_000,
                "contract_years_left": 0.4,
                "injury_days_per_1000_min": 145.0,
                "nt_total_caps": 0,
                "delta_minutes": -540,
                "delta_sofa_expectedGoals_per90": -0.04,
                "delta_sofa_assists_per90": -0.03,
                "delta_sofa_goals_per90": -0.02,
            },
        ]
    )

    out = add_history_strength_features(frame)
    assert "history_strength_score" in out.columns
    assert "history_strength_coverage" in out.columns
    assert "history_strength_tier" in out.columns
    assert out.loc[0, "history_strength_score"] > out.loc[1, "history_strength_score"]
    assert 0.0 <= float(out.loc[0, "history_strength_coverage"]) <= 1.0
    assert 0.0 <= float(out.loc[1, "history_strength_coverage"]) <= 1.0


def test_parse_transfer_rows_supports_json_payload() -> None:
    payload = {
        "transfers": [
            {
                "season": "2023/24",
                "date": "01/07/2023",
                "transferFee": "€12.00m",
                "isLoan": False,
            },
            {
                "season": "2021/22",
                "date": "31/08/2021",
                "transferFee": "Loan fee: €1.5m",
                "isLoan": True,
            },
        ]
    }
    rows = parse_transfer_rows(json.dumps(payload).encode("utf-8"))
    assert len(rows) == 2

    first = rows[0]
    assert first["season"] == "2023/24"
    assert first["fee_eur"] == 12_000_000.0
    assert first["is_loan"] == 0

    second = rows[1]
    assert second["season"] == "2021/22"
    assert second["is_loan"] == 1
