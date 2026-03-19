"""Static profile taxonomy and template data for market-value detail views."""

from __future__ import annotations

from typing import Any


PROFILE_METRIC_SPECS: tuple[tuple[str, str, int], ...] = (
    ("sofa_goals_per90", "Goals/90", 1),
    ("sofa_assists_per90", "Assists/90", 1),
    ("sofa_expectedGoals_per90", "xG/90", 1),
    ("sofa_totalShots_per90", "Shots/90", 1),
    ("sofa_keyPasses_per90", "Key passes/90", 1),
    ("sofa_successfulDribbles_per90", "Successful dribbles/90", 1),
    ("sofa_accuratePassesPercentage", "Pass accuracy %", 1),
    ("sofa_totalDuelsWonPercentage", "Duels won %", 1),
    ("sofa_tackles_per90", "Tackles/90", 1),
    ("sofa_interceptions_per90", "Interceptions/90", 1),
    ("sofa_clearances_per90", "Clearances/90", 1),
    ("injury_days_per_1000_min", "Injury days/1000 min", -1),
    ("history_strength_score", "History strength", 1),
)

ADVANCED_METRIC_SPECS: tuple[tuple[str, str, int], ...] = (
    ("sofa_goals_per90", "Goals/90", 1),
    ("sofa_assists_per90", "Assists/90", 1),
    ("sofa_expectedGoals_per90", "xG/90", 1),
    ("sofa_totalShots_per90", "Shots/90", 1),
    ("sofa_shotsOnTarget_per90", "Shots on target/90", 1),
    ("sofa_goalConversionPercentage", "Goal conversion %", 1),
    ("sofa_keyPasses_per90", "Key passes/90", 1),
    ("sofa_successfulDribbles_per90", "Dribbles/90", 1),
    ("sofa_accurateFinalThirdPasses_per90", "Final-third passes/90", 1),
    ("sofa_accurateCrossesPercentage", "Cross accuracy %", 1),
    ("sofa_accuratePassesPercentage", "Pass accuracy %", 1),
    ("sofa_totalDuelsWonPercentage", "Duels won %", 1),
    ("sofa_aerialDuelsWon_per90", "Aerial duels won/90", 1),
    ("sofa_aerialDuelsWonPercentage", "Aerial duels won %", 1),
    ("sofa_tackles_per90", "Tackles/90", 1),
    ("sofa_interceptions_per90", "Interceptions/90", 1),
    ("sofa_clearances_per90", "Clearances/90", 1),
    ("sofa_saves", "Saves", 1),
    ("sofa_highClaims", "High claims", 1),
    ("sofa_successfulRunsOut", "Successful runs out", 1),
    ("sofa_accurateLongBallsPercentage", "Long-ball accuracy %", 1),
    ("sb_progressive_passes_per90", "Progressive passes/90", 1),
    ("sb_progressive_carries_per90", "Progressive carries/90", 1),
    ("sb_passes_into_box_per90", "Passes into box/90", 1),
    ("sb_shot_assists_per90", "Shot assists/90", 1),
    ("sb_duel_win_rate", "Duel win rate", 1),
    ("sb_aerial_win_rate", "Aerial win rate", 1),
    ("injury_days_per_1000_min", "Injury days/1000 min", -1),
    ("history_strength_score", "History strength", 1),
)

ADVANCED_METRIC_DIRECTION: dict[str, int] = {key: direction for key, _, direction in ADVANCED_METRIC_SPECS}
ADVANCED_METRIC_LABEL: dict[str, str] = {key: label for key, label, _ in ADVANCED_METRIC_SPECS}

PROFILE_CONTEXT_FIELDS: tuple[str, ...] = (
    "player_id",
    "name",
    "club",
    "league",
    "season",
    "country",
    "nationality",
    "position_main",
    "position_group",
    "model_position",
    "age",
    "height",
    "foot",
    "contract_years_left",
)

PROFILE_STAT_GROUP_ORDER: tuple[str, ...] = (
    "Profile & Context",
    "Value & Model",
    "Attacking",
    "Passing & Progression",
    "Defending & Duels",
    "Goalkeeping",
    "External Tactical",
    "Availability & Physical",
    "History & Context",
    "Schedule & Market",
    "Other Metrics",
)

PROFILE_STAT_SKIP_FIELDS: tuple[str, ...] = (
    "split",
    "source",
    "source_file",
    "league_norm",
    "position_norm",
    "value_segment",
    "expected_value_raw_eur",
    "expected_value_low_raw_eur",
    "expected_value_high_raw_eur",
)

ROLE_KEY_LABELS: dict[str, str] = {
    "GK": "Goalkeeper",
    "CB": "Centre-back",
    "FB": "Fullback / Wing-back",
    "DF": "Defender",
    "DM": "Defensive midfielder",
    "CM": "Central midfielder",
    "AM": "Attacking midfielder",
    "MF": "Midfielder",
    "W": "Winger",
    "SS": "Support forward",
    "ST": "Striker",
    "FW": "Forward",
    "UNK": "Unknown role",
}

RADAR_AXES_BY_POSITION: dict[str, tuple[str, ...]] = {
    "GK": (
        "sofa_saves",
        "sofa_highClaims",
        "sofa_successfulRunsOut",
        "sofa_accuratePassesPercentage",
        "sofa_accurateLongBallsPercentage",
        "injury_days_per_1000_min",
    ),
    "CB": (
        "sb_aerial_win_rate",
        "sb_duel_win_rate",
        "sofa_interceptions_per90",
        "sofa_clearances_per90",
        "sb_progressive_passes_per90",
        "sofa_accuratePassesPercentage",
    ),
    "FB": (
        "sofa_tackles_per90",
        "sofa_interceptions_per90",
        "sb_progressive_carries_per90",
        "sb_progressive_passes_per90",
        "sofa_accurateCrossesPercentage",
        "sofa_keyPasses_per90",
    ),
    "DM": (
        "sofa_tackles_per90",
        "sofa_interceptions_per90",
        "sb_duel_win_rate",
        "sb_progressive_passes_per90",
        "sofa_accuratePassesPercentage",
        "history_strength_score",
    ),
    "CM": (
        "sb_progressive_passes_per90",
        "sofa_accurateFinalThirdPasses_per90",
        "sofa_keyPasses_per90",
        "sofa_accuratePassesPercentage",
        "sb_progressive_carries_per90",
        "sofa_tackles_per90",
    ),
    "AM": (
        "sofa_keyPasses_per90",
        "sb_shot_assists_per90",
        "sb_passes_into_box_per90",
        "sofa_successfulDribbles_per90",
        "sofa_expectedGoals_per90",
        "sofa_assists_per90",
    ),
    "W": (
        "sofa_successfulDribbles_per90",
        "sb_progressive_carries_per90",
        "sb_shot_assists_per90",
        "sb_passes_into_box_per90",
        "sofa_expectedGoals_per90",
        "sofa_goalConversionPercentage",
    ),
    "SS": (
        "sofa_keyPasses_per90",
        "sb_shot_assists_per90",
        "sofa_successfulDribbles_per90",
        "sofa_expectedGoals_per90",
        "sofa_shotsOnTarget_per90",
        "sb_passes_into_box_per90",
    ),
    "ST": (
        "sofa_expectedGoals_per90",
        "sofa_totalShots_per90",
        "sofa_shotsOnTarget_per90",
        "sofa_goalConversionPercentage",
        "sb_aerial_win_rate",
        "sofa_goals_per90",
    ),
    "DF": (
        "sofa_accuratePassesPercentage",
        "sofa_clearances_per90",
        "sofa_interceptions_per90",
        "sofa_tackles_per90",
        "sofa_aerialDuelsWon_per90",
        "sofa_totalDuelsWonPercentage",
    ),
    "MF": (
        "sofa_keyPasses_per90",
        "sofa_assists_per90",
        "sofa_accuratePassesPercentage",
        "sofa_tackles_per90",
        "sofa_interceptions_per90",
        "sofa_successfulDribbles_per90",
    ),
    "FW": (
        "sofa_goals_per90",
        "sofa_expectedGoals_per90",
        "sofa_shotsOnTarget_per90",
        "sofa_keyPasses_per90",
        "sofa_successfulDribbles_per90",
        "sofa_totalDuelsWonPercentage",
    ),
}

ARCHETYPE_TEMPLATES: dict[str, tuple[dict[str, Any], ...]] = {
    "GK": (
        {
            "name": "Shot-Stopper",
            "description": "Goalkeeper profile centered on save volume and box command.",
            "targets": {
                "sofa_saves": 0.86,
                "sofa_highClaims": 0.70,
                "sofa_accuratePassesPercentage": 0.44,
                "sofa_successfulRunsOut": 0.42,
                "injury_days_per_1000_min": 0.70,
            },
        },
        {
            "name": "Sweeper Keeper",
            "description": "Proactive keeper for high-line structures and circulation.",
            "targets": {
                "sofa_successfulRunsOut": 0.86,
                "sofa_accuratePassesPercentage": 0.80,
                "sofa_accurateLongBallsPercentage": 0.74,
                "sofa_highClaims": 0.60,
                "sofa_saves": 0.52,
            },
        },
    ),
    "CB": (
        {
            "name": "Ball-Playing Centre-Back",
            "description": "Centre-back who defends the box and progresses play cleanly.",
            "targets": {
                "sb_progressive_passes_per90": 0.82,
                "sofa_accuratePassesPercentage": 0.86,
                "sofa_interceptions_per90": 0.66,
                "sb_aerial_win_rate": 0.70,
                "sofa_clearances_per90": 0.50,
            },
        },
        {
            "name": "Stopper Centre-Back",
            "description": "Duel-heavy centre-back with aerial and box-defending volume.",
            "targets": {
                "sofa_clearances_per90": 0.90,
                "sofa_interceptions_per90": 0.74,
                "sb_duel_win_rate": 0.76,
                "sb_aerial_win_rate": 0.80,
                "sofa_accuratePassesPercentage": 0.50,
            },
        },
    ),
    "FB": (
        {
            "name": "Overlapping Fullback",
            "description": "Wide defender with crossing, carrying, and repeat transition volume.",
            "targets": {
                "sb_progressive_carries_per90": 0.80,
                "sofa_accurateCrossesPercentage": 0.72,
                "sofa_keyPasses_per90": 0.66,
                "sofa_tackles_per90": 0.62,
                "sofa_interceptions_per90": 0.58,
            },
        },
        {
            "name": "Inverted Fullback",
            "description": "Fullback who supports buildup and protects central zones.",
            "targets": {
                "sb_progressive_passes_per90": 0.78,
                "sofa_accuratePassesPercentage": 0.82,
                "sofa_interceptions_per90": 0.64,
                "sofa_tackles_per90": 0.62,
                "sb_duel_win_rate": 0.60,
            },
        },
    ),
    "DM": (
        {
            "name": "Anchor 6",
            "description": "Defensive midfielder focused on regains, duels, and screening.",
            "targets": {
                "sofa_tackles_per90": 0.86,
                "sofa_interceptions_per90": 0.88,
                "sb_duel_win_rate": 0.74,
                "sofa_accuratePassesPercentage": 0.62,
                "sb_progressive_passes_per90": 0.52,
            },
        },
        {
            "name": "Regista 6",
            "description": "Deep midfielder who progresses play without sacrificing control.",
            "targets": {
                "sb_progressive_passes_per90": 0.86,
                "sofa_accuratePassesPercentage": 0.88,
                "sofa_accurateFinalThirdPasses_per90": 0.64,
                "sofa_interceptions_per90": 0.62,
                "sofa_tackles_per90": 0.56,
            },
        },
    ),
    "CM": (
        {
            "name": "Deep Playmaker",
            "description": "Tempo-setter with progression and final-third delivery.",
            "targets": {
                "sb_progressive_passes_per90": 0.86,
                "sofa_accuratePassesPercentage": 0.88,
                "sofa_accurateFinalThirdPasses_per90": 0.72,
                "sofa_keyPasses_per90": 0.64,
                "sofa_tackles_per90": 0.46,
            },
        },
        {
            "name": "Box-to-Box Midfielder",
            "description": "Balanced midfielder with carrying, pressing, and end-product support.",
            "targets": {
                "sb_progressive_carries_per90": 0.70,
                "sofa_keyPasses_per90": 0.60,
                "sofa_tackles_per90": 0.68,
                "sofa_interceptions_per90": 0.60,
                "sofa_expectedGoals_per90": 0.46,
            },
        },
    ),
    "AM": (
        {
            "name": "Chance Creator 10",
            "description": "Final-third specialist driven by chance creation and box access.",
            "targets": {
                "sofa_keyPasses_per90": 0.90,
                "sb_shot_assists_per90": 0.86,
                "sb_passes_into_box_per90": 0.80,
                "sofa_successfulDribbles_per90": 0.68,
                "sofa_assists_per90": 0.70,
            },
        },
        {
            "name": "Goal Threat 10",
            "description": "Attacking midfielder who adds real shot and scoring value.",
            "targets": {
                "sofa_expectedGoals_per90": 0.76,
                "sofa_shotsOnTarget_per90": 0.72,
                "sofa_keyPasses_per90": 0.72,
                "sofa_successfulDribbles_per90": 0.62,
                "sb_passes_into_box_per90": 0.68,
            },
        },
    ),
    "W": (
        {
            "name": "Direct Winger",
            "description": "Wide attacker who wins ground with carrying and dribbling.",
            "targets": {
                "sofa_successfulDribbles_per90": 0.90,
                "sb_progressive_carries_per90": 0.86,
                "sofa_expectedGoals_per90": 0.62,
                "sb_passes_into_box_per90": 0.64,
                "sofa_goalConversionPercentage": 0.56,
            },
        },
        {
            "name": "Creator Winger",
            "description": "Wide creator focused on shot assists, final ball, and box service.",
            "targets": {
                "sb_shot_assists_per90": 0.90,
                "sofa_keyPasses_per90": 0.88,
                "sb_passes_into_box_per90": 0.82,
                "sofa_accurateCrossesPercentage": 0.70,
                "sofa_successfulDribbles_per90": 0.62,
            },
        },
    ),
    "SS": (
        {
            "name": "Link Forward",
            "description": "Support striker who creates and combines around the box.",
            "targets": {
                "sofa_keyPasses_per90": 0.82,
                "sb_shot_assists_per90": 0.76,
                "sofa_successfulDribbles_per90": 0.70,
                "sofa_expectedGoals_per90": 0.62,
                "sb_passes_into_box_per90": 0.66,
            },
        },
        {
            "name": "Shadow Striker",
            "description": "Second forward who attacks space and still contributes creation.",
            "targets": {
                "sofa_expectedGoals_per90": 0.80,
                "sofa_shotsOnTarget_per90": 0.76,
                "sofa_keyPasses_per90": 0.64,
                "sb_passes_into_box_per90": 0.64,
                "sofa_successfulDribbles_per90": 0.56,
            },
        },
    ),
    "ST": (
        {
            "name": "Poacher",
            "description": "Final-third finisher with high shot and xG output.",
            "targets": {
                "sofa_goals_per90": 0.92,
                "sofa_expectedGoals_per90": 0.90,
                "sofa_totalShots_per90": 0.86,
                "sofa_shotsOnTarget_per90": 0.82,
                "sofa_goalConversionPercentage": 0.68,
            },
        },
        {
            "name": "Mobile 9",
            "description": "Striker who stretches the line and adds carrying or combination work.",
            "targets": {
                "sofa_expectedGoals_per90": 0.80,
                "sofa_successfulDribbles_per90": 0.70,
                "sofa_keyPasses_per90": 0.58,
                "sb_passes_into_box_per90": 0.54,
                "sofa_shotsOnTarget_per90": 0.72,
            },
        },
        {
            "name": "Target Forward",
            "description": "Central reference point with aerial value and efficient finishing.",
            "targets": {
                "sb_aerial_win_rate": 0.84,
                "sofa_aerialDuelsWonPercentage": 0.82,
                "sofa_shotsOnTarget_per90": 0.70,
                "sofa_goalConversionPercentage": 0.64,
                "sofa_goals_per90": 0.72,
            },
        },
    ),
    "DF": (
        {
            "name": "Ball-Playing Defender",
            "description": "Build-up defender with passing control and progression.",
            "targets": {
                "sofa_accuratePassesPercentage": 0.90,
                "sb_progressive_passes_per90": 0.72,
                "sofa_interceptions_per90": 0.64,
                "sofa_clearances_per90": 0.44,
                "sofa_totalDuelsWonPercentage": 0.64,
            },
        },
        {
            "name": "Stopper",
            "description": "Duel-oriented defender with heavy defensive volume.",
            "targets": {
                "sofa_clearances_per90": 0.90,
                "sofa_interceptions_per90": 0.72,
                "sofa_tackles_per90": 0.72,
                "sofa_totalDuelsWonPercentage": 0.74,
                "sofa_accuratePassesPercentage": 0.44,
            },
        },
    ),
    "MF": (
        {
            "name": "Deep Playmaker",
            "description": "Tempo-setter with high passing quality and buildup contribution.",
            "targets": {
                "sofa_accuratePassesPercentage": 0.90,
                "sofa_keyPasses_per90": 0.72,
                "sofa_assists_per90": 0.62,
                "sofa_tackles_per90": 0.45,
                "sofa_interceptions_per90": 0.55,
            },
        },
        {
            "name": "Ball-Winning Midfielder",
            "description": "Midfielder focused on regains and duel control.",
            "targets": {
                "sofa_tackles_per90": 0.88,
                "sofa_interceptions_per90": 0.88,
                "sofa_totalDuelsWonPercentage": 0.76,
                "sofa_accuratePassesPercentage": 0.60,
                "sofa_keyPasses_per90": 0.40,
            },
        },
    ),
    "FW": (
        {
            "name": "Creator Forward",
            "description": "Chance-creating attacker with link-play and dribble volume.",
            "targets": {
                "sofa_assists_per90": 0.80,
                "sofa_keyPasses_per90": 0.90,
                "sofa_successfulDribbles_per90": 0.82,
                "sofa_accuratePassesPercentage": 0.62,
                "sofa_goals_per90": 0.58,
            },
        },
        {
            "name": "Pressing Forward",
            "description": "Work-rate forward that contributes in duels and defensive actions.",
            "targets": {
                "sofa_tackles_per90": 0.80,
                "sofa_totalDuelsWonPercentage": 0.72,
                "sofa_keyPasses_per90": 0.62,
                "sofa_goals_per90": 0.55,
                "sofa_successfulDribbles_per90": 0.55,
            },
        },
    ),
}

FORMATION_FIT_TEMPLATES: dict[str, tuple[dict[str, Any], ...]] = {
    "GK": (
        {
            "formation": "4-3-3",
            "role": "High-line Keeper",
            "targets": {
                "sofa_successfulRunsOut": 0.80,
                "sofa_accuratePassesPercentage": 0.72,
                "sofa_accurateLongBallsPercentage": 0.68,
                "sofa_highClaims": 0.62,
            },
        },
        {
            "formation": "5-3-2",
            "role": "Box Keeper",
            "targets": {
                "sofa_saves": 0.82,
                "sofa_highClaims": 0.72,
                "sofa_successfulRunsOut": 0.42,
                "sofa_accuratePassesPercentage": 0.50,
            },
        },
    ),
    "CB": (
        {
            "formation": "4-3-3",
            "role": "Centre-Back",
            "targets": {
                "sofa_clearances_per90": 0.72,
                "sofa_interceptions_per90": 0.66,
                "sb_aerial_win_rate": 0.72,
                "sofa_accuratePassesPercentage": 0.64,
            },
        },
        {
            "formation": "3-4-3",
            "role": "Wide Centre-Back",
            "targets": {
                "sb_progressive_passes_per90": 0.74,
                "sofa_accuratePassesPercentage": 0.76,
                "sofa_interceptions_per90": 0.60,
                "sofa_tackles_per90": 0.56,
            },
        },
    ),
    "FB": (
        {
            "formation": "4-3-3",
            "role": "Overlapping Fullback",
            "targets": {
                "sb_progressive_carries_per90": 0.78,
                "sofa_accurateCrossesPercentage": 0.68,
                "sofa_keyPasses_per90": 0.62,
                "sofa_tackles_per90": 0.60,
            },
        },
        {
            "formation": "3-4-3",
            "role": "Wing-Back",
            "targets": {
                "sb_progressive_carries_per90": 0.82,
                "sb_passes_into_box_per90": 0.66,
                "sofa_accurateCrossesPercentage": 0.62,
                "sofa_interceptions_per90": 0.54,
            },
        },
        {
            "formation": "4-2-3-1",
            "role": "Inverted Fullback",
            "targets": {
                "sb_progressive_passes_per90": 0.78,
                "sofa_accuratePassesPercentage": 0.82,
                "sofa_interceptions_per90": 0.60,
                "sofa_tackles_per90": 0.58,
            },
        },
    ),
    "DM": (
        {
            "formation": "4-3-3",
            "role": "No. 6",
            "targets": {
                "sofa_interceptions_per90": 0.80,
                "sofa_tackles_per90": 0.80,
                "sofa_accuratePassesPercentage": 0.78,
                "sb_duel_win_rate": 0.66,
            },
        },
        {
            "formation": "4-2-3-1",
            "role": "Double Pivot Controller",
            "targets": {
                "sb_progressive_passes_per90": 0.80,
                "sofa_accuratePassesPercentage": 0.84,
                "sofa_interceptions_per90": 0.62,
                "sofa_tackles_per90": 0.58,
            },
        },
    ),
    "CM": (
        {
            "formation": "4-3-3",
            "role": "No. 8",
            "targets": {
                "sb_progressive_passes_per90": 0.74,
                "sofa_accurateFinalThirdPasses_per90": 0.64,
                "sofa_keyPasses_per90": 0.60,
                "sofa_tackles_per90": 0.58,
            },
        },
        {
            "formation": "4-2-3-1",
            "role": "Circulating 8",
            "targets": {
                "sofa_accuratePassesPercentage": 0.84,
                "sb_progressive_passes_per90": 0.72,
                "sb_progressive_carries_per90": 0.58,
                "sofa_interceptions_per90": 0.52,
            },
        },
    ),
    "AM": (
        {
            "formation": "4-2-3-1",
            "role": "No. 10",
            "targets": {
                "sofa_keyPasses_per90": 0.84,
                "sb_shot_assists_per90": 0.82,
                "sb_passes_into_box_per90": 0.74,
                "sofa_successfulDribbles_per90": 0.66,
            },
        },
        {
            "formation": "4-3-3",
            "role": "Advanced 8",
            "targets": {
                "sofa_keyPasses_per90": 0.72,
                "sofa_expectedGoals_per90": 0.60,
                "sofa_accurateFinalThirdPasses_per90": 0.66,
                "sb_progressive_carries_per90": 0.58,
            },
        },
    ),
    "W": (
        {
            "formation": "4-3-3",
            "role": "Touchline Winger",
            "targets": {
                "sofa_successfulDribbles_per90": 0.84,
                "sb_progressive_carries_per90": 0.82,
                "sofa_accurateCrossesPercentage": 0.66,
                "sb_shot_assists_per90": 0.68,
            },
        },
        {
            "formation": "4-2-3-1",
            "role": "Inverted Winger",
            "targets": {
                "sofa_expectedGoals_per90": 0.72,
                "sb_passes_into_box_per90": 0.68,
                "sofa_successfulDribbles_per90": 0.74,
                "sofa_goalConversionPercentage": 0.58,
            },
        },
    ),
    "SS": (
        {
            "formation": "3-5-2",
            "role": "Second Striker",
            "targets": {
                "sofa_keyPasses_per90": 0.72,
                "sb_shot_assists_per90": 0.68,
                "sofa_expectedGoals_per90": 0.66,
                "sofa_successfulDribbles_per90": 0.62,
            },
        },
        {
            "formation": "4-4-2",
            "role": "Support Forward",
            "targets": {
                "sofa_keyPasses_per90": 0.70,
                "sb_passes_into_box_per90": 0.62,
                "sofa_shotsOnTarget_per90": 0.66,
                "sofa_successfulDribbles_per90": 0.58,
            },
        },
    ),
    "ST": (
        {
            "formation": "4-3-3",
            "role": "Lone 9",
            "targets": {
                "sofa_goals_per90": 0.82,
                "sofa_expectedGoals_per90": 0.84,
                "sofa_totalShots_per90": 0.78,
                "sofa_shotsOnTarget_per90": 0.76,
            },
        },
        {
            "formation": "4-2-3-1",
            "role": "Linking 9",
            "targets": {
                "sofa_keyPasses_per90": 0.64,
                "sb_passes_into_box_per90": 0.58,
                "sofa_expectedGoals_per90": 0.72,
                "sofa_goalConversionPercentage": 0.56,
            },
        },
        {
            "formation": "3-5-2",
            "role": "Target 9",
            "targets": {
                "sb_aerial_win_rate": 0.82,
                "sofa_aerialDuelsWonPercentage": 0.80,
                "sofa_shotsOnTarget_per90": 0.70,
                "sofa_goals_per90": 0.72,
            },
        },
    ),
    "DF": (
        {
            "formation": "4-3-3",
            "role": "Defender",
            "targets": {
                "sofa_clearances_per90": 0.68,
                "sofa_interceptions_per90": 0.62,
                "sofa_totalDuelsWonPercentage": 0.68,
                "sofa_accuratePassesPercentage": 0.60,
            },
        },
    ),
    "MF": (
        {
            "formation": "4-3-3",
            "role": "Midfielder",
            "targets": {
                "sofa_keyPasses_per90": 0.66,
                "sofa_accuratePassesPercentage": 0.76,
                "sofa_tackles_per90": 0.60,
                "sofa_interceptions_per90": 0.56,
            },
        },
    ),
    "FW": (
        {
            "formation": "4-3-3",
            "role": "Forward",
            "targets": {
                "sofa_goals_per90": 0.76,
                "sofa_expectedGoals_per90": 0.74,
                "sofa_keyPasses_per90": 0.56,
                "sofa_successfulDribbles_per90": 0.58,
            },
        },
    ),
}


__all__ = [
    "ADVANCED_METRIC_DIRECTION",
    "ADVANCED_METRIC_LABEL",
    "ADVANCED_METRIC_SPECS",
    "ARCHETYPE_TEMPLATES",
    "FORMATION_FIT_TEMPLATES",
    "PROFILE_CONTEXT_FIELDS",
    "PROFILE_METRIC_SPECS",
    "PROFILE_STAT_GROUP_ORDER",
    "PROFILE_STAT_SKIP_FIELDS",
    "RADAR_AXES_BY_POSITION",
    "ROLE_KEY_LABELS",
]
