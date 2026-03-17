from __future__ import annotations

from pathlib import Path

import pandas as pd

from scouting_ml.models.build_dataset import add_model_features
from scouting_ml.scripts.build_player_contracts import parse_contract_fields
from scouting_ml.scripts.build_player_injuries import aggregate_player_injuries


def test_parse_contract_fields_extracts_joined_year() -> None:
    html = b"""
    <html>
      <body>
        <li class="data-header__label">Contract expires:
          <span class="data-header__content">30/06/2027</span>
        </li>
        <li class="data-header__label">Joined:
          <span class="data-header__content">01/07/2023</span>
        </li>
        <li class="data-header__label">Player agent:
          <span class="data-header__content">Example Agency</span>
        </li>
      </body>
    </html>
    """
    parsed = parse_contract_fields(html)
    assert parsed["contract_until_year"] == 2027
    assert parsed["joined_year"] == 2023
    assert parsed["agent_name"] == "Example Agency"


def test_aggregate_player_injuries_tracks_type_buckets() -> None:
    rows = [
        {"season": "2024/25", "injury_name": "Hamstring injury", "days_missed": 21, "games_missed": 4},
        {"season": "2024/25", "injury_name": "Knee surgery", "days_missed": 95, "games_missed": 14},
        {"season": "2024/25", "injury_name": "Flu", "days_missed": 7, "games_missed": 1},
        {"season": "2024/25", "injury_name": "Calf strain", "days_missed": 52, "games_missed": 7},
    ]
    out = aggregate_player_injuries(rows, target_seasons=["2024/25"])
    season = out["2024/25"]
    assert season["injury_count"] == 4
    assert season["injury_soft_tissue_count"] == 2
    assert season["injury_bone_joint_count"] == 1
    assert season["injury_illness_count"] == 1
    assert season["injury_surgery_count"] == 1
    assert season["injury_long_absence_count"] == 2
    assert season["injury_repeat_soft_tissue_flag"] == 1
    assert season["injury_repeat_bone_joint_flag"] == 0
    assert season["injury_avg_days_per_case"] == 43.75
    assert season["injury_avg_games_per_case"] == 6.5


def test_add_model_features_derives_tm_context_fields(tmp_path: Path) -> None:
    external_dir = tmp_path / "external"
    external_dir.mkdir()

    pd.DataFrame(
        [
            {
                "player_id": "p1",
                "season": "2024/25",
                "contract_until_year": 2027,
                "agent_name": "Agency X",
                "loan_flag": 1,
                "joined_year": 2023,
            }
        ]
    ).to_csv(external_dir / "player_contracts.csv", index=False)

    pd.DataFrame(
        [
            {
                "player_id": "p1",
                "season": "2024/25",
                "days_missed": 120,
                "games_missed": 18,
                "injury_count": 4,
                "major_injury_flag": 1,
                "injury_long_absence_count": 2,
                "injury_avg_days_per_case": 30.0,
                "injury_avg_games_per_case": 4.5,
                "injury_repeat_soft_tissue_flag": 1,
                "injury_repeat_bone_joint_flag": 0,
                "injury_soft_tissue_count": 2,
                "injury_bone_joint_count": 1,
                "injury_illness_count": 1,
                "injury_surgery_count": 1,
            }
        ]
    ).to_csv(external_dir / "player_injuries.csv", index=False)

    pd.DataFrame(
        [
            {
                "player_id": "p1",
                "season": "2024/25",
                "transfer_count_3y": 2,
                "transfer_loans_3y": 1,
                "transfer_paid_moves_3y": 1,
                "transfer_total_fees_3y_eur": 9000000,
                "transfer_max_fee_career_eur": 12000000,
                "last_transfer_fee_eur": 9000000,
                "last_transfer_is_loan": 0,
                "last_transfer_fee_text": "EUR9.0m",
            }
        ]
    ).to_csv(external_dir / "player_transfers.csv", index=False)

    pd.DataFrame(
        [
            {
                "player_id": "p1",
                "season": "2024/25",
                "avail_reports": 20,
                "avail_minutes": 1600,
                "avail_start_share": 0.8,
                "avail_appearance_share": 0.9,
                "avail_full_match_share": 0.55,
                "avail_unused_bench_share": 0.05,
                "avail_captain_share": 0.10,
                "avail_minutes_share": 1600 / (20 * 90),
                "avail_rating_mean": 7.2,
            }
        ]
    ).to_csv(external_dir / "player_availability.csv", index=False)

    pd.DataFrame(
        [
            {
                "league": "Eredivisie",
                "club": "Example FC",
                "season": "2024/25",
                "fixture_points_per_match": 1.8,
                "fixture_goal_diff_per_match": 0.5,
                "fixture_clean_sheet_share": 0.25,
                "fixture_failed_to_score_share": 0.10,
                "fixture_scoring_environment": 2.9,
            }
        ]
    ).to_csv(external_dir / "fixture_context.csv", index=False)

    base = pd.DataFrame(
        [
            {
                "player_id": "p1",
                "name": "Player One",
                "league": "Eredivisie",
                "club": "Example FC",
                "season": "2024/25",
                "position_group": "FW",
                "age": 21,
                "market_value_eur": 7000000,
                "sofa_minutesPlayed": 1800,
            }
        ]
    )
    out = add_model_features(base, external_dir=external_dir)

    assert out.loc[0, "club_tenure_years"] == 1.0
    assert out.loc[0, "recent_arrival_flag"] == 1
    assert out.loc[0, "contract_agent_known_flag"] == 1
    assert out.loc[0, "contract_loan_context_flag"] == 1
    assert out.loc[0, "injury_soft_tissue_share"] == 0.5
    assert out.loc[0, "injury_surgery_flag"] == 1
    assert out.loc[0, "transfer_recent_paid_share_3y"] == 0.5
    assert out.loc[0, "transfer_recent_loan_share_3y"] == 0.5
    assert out.loc[0, "transfer_last_move_paid_flag"] == 1
    assert out.loc[0, "availability_selection_score"] > 0.5
    assert out.loc[0, "availability_performance_hint"] == 0.72
    assert out.loc[0, "fixture_team_form_score"] > 0.4
    assert out.loc[0, "fixture_environment_score"] > 0.5
    assert out.loc[0, "availability_risk_score"] > 0.0
