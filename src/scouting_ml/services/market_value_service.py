"""Service functions for market-value prediction artifacts."""

from __future__ import annotations

import json
import os
import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Sequence

import numpy as np
import pandas as pd

from scouting_ml.features.history_strength import (
    HISTORY_COMPONENT_COLUMNS,
    HISTORY_COMPONENT_LABELS,
    HISTORY_COMPONENT_WEIGHTS,
    add_history_strength_features,
)
from scouting_ml.reporting.market_value_benchmarks import build_market_value_benchmark_payload

Split = Literal["test", "val"]
ChampionRole = Literal["valuation", "future_shortlist"]

TEST_PRED_ENV = "SCOUTING_TEST_PREDICTIONS_PATH"
VAL_PRED_ENV = "SCOUTING_VAL_PREDICTIONS_PATH"
METRICS_ENV = "SCOUTING_METRICS_PATH"
MODEL_MANIFEST_ENV = "SCOUTING_MODEL_MANIFEST_PATH"
BENCHMARK_REPORT_ENV = "SCOUTING_BENCHMARK_REPORT_PATH"
ENABLE_RESIDUAL_CALIBRATION_ENV = "SCOUTING_ENABLE_RESIDUAL_CALIBRATION"
CALIBRATION_MIN_SAMPLES_ENV = "SCOUTING_CALIBRATION_MIN_SAMPLES"
WATCHLIST_PATH_ENV = "SCOUTING_WATCHLIST_PATH"
VALUATION_TEST_PRED_ENV = "SCOUTING_VALUATION_TEST_PREDICTIONS_PATH"
VALUATION_VAL_PRED_ENV = "SCOUTING_VALUATION_VAL_PREDICTIONS_PATH"
VALUATION_METRICS_ENV = "SCOUTING_VALUATION_METRICS_PATH"
FUTURE_TEST_PRED_ENV = "SCOUTING_FUTURE_SHORTLIST_TEST_PREDICTIONS_PATH"
FUTURE_VAL_PRED_ENV = "SCOUTING_FUTURE_SHORTLIST_VAL_PREDICTIONS_PATH"
FUTURE_METRICS_ENV = "SCOUTING_FUTURE_SHORTLIST_METRICS_PATH"

DEFAULT_TEST_PRED = Path("data/model/big5_predictions_full_v2.csv")
DEFAULT_VAL_PRED = Path("data/model/big5_predictions_full_v2_val.csv")
DEFAULT_METRICS = Path("data/model/big5_predictions_full_v2.metrics.json")
DEFAULT_MODEL_MANIFEST = Path("data/model/model_manifest.json")
DEFAULT_BENCHMARK_REPORT = Path("data/model/reports/market_value_benchmark_report.json")
DEFAULT_WATCHLIST_PATH = Path("data/model/scout_watchlist.jsonl")

SPLIT_TO_PATH = {
    "test": (TEST_PRED_ENV, DEFAULT_TEST_PRED),
    "val": (VAL_PRED_ENV, DEFAULT_VAL_PRED),
}
ROLE_TO_MANIFEST_KEY: dict[ChampionRole, str] = {
    "valuation": "valuation_champion",
    "future_shortlist": "future_shortlist_champion",
}
ROLE_ENV_NAMES: dict[ChampionRole, dict[str, str]] = {
    "valuation": {
        "test": VALUATION_TEST_PRED_ENV,
        "val": VALUATION_VAL_PRED_ENV,
        "metrics": VALUATION_METRICS_ENV,
    },
    "future_shortlist": {
        "test": FUTURE_TEST_PRED_ENV,
        "val": FUTURE_VAL_PRED_ENV,
        "metrics": FUTURE_METRICS_ENV,
    },
}
FUTURE_OVERLAY_COLUMNS: tuple[str, ...] = (
    "_player_key",
    "_season_key",
    "future_growth_probability",
    "future_scout_blend_score",
    "future_scout_score",
    "has_next_season_target",
    "next_market_value_eur",
    "next_minutes",
    "next_season",
    "value_growth_gt25pct_flag",
    "value_growth_next_season_eur",
    "value_growth_next_season_log_delta",
    "value_growth_next_season_pct",
    "value_growth_positive_flag",
)

BIG5_LEAGUES = {
    "english premier league",
    "premier league",
    "spanish la liga",
    "la liga",
    "laliga",
    "italian serie a",
    "serie a",
    "german bundesliga",
    "bundesliga",
    "french ligue 1",
    "ligue 1",
}

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


class ArtifactNotFoundError(FileNotFoundError):
    """Raised when required model artifacts are missing."""


@dataclass
class _FrameCache:
    key: str
    version: tuple[tuple[str, int], ...]
    frame: pd.DataFrame


@dataclass
class _ResidualCalibrationCache:
    path: Path
    mtime_ns: int
    min_samples: int
    payload: dict[str, Any]


_PRED_CACHE: Dict[str, _FrameCache] = {}
_METRICS_CACHE: Dict[str, tuple[Path, int, dict[str, Any]]] = {}
_RESIDUAL_CALIBRATION_CACHE: _ResidualCalibrationCache | None = None
_REQUIRED_ARTIFACT_ENVS = (TEST_PRED_ENV, VAL_PRED_ENV, METRICS_ENV)


def _resolve_path(env_var: str, default_path: Path) -> Path:
    value = os.getenv(env_var, "").strip()
    return Path(value) if value else default_path


def _manifest_path() -> Path:
    manifest_env_raw = os.getenv(MODEL_MANIFEST_ENV, "").strip()
    return Path(manifest_env_raw) if manifest_env_raw else DEFAULT_MODEL_MANIFEST


def _load_manifest_payload() -> dict[str, Any] | None:
    manifest_path = _manifest_path()
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _manifest_role_section(payload: dict[str, Any] | None, role: ChampionRole) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    key = ROLE_TO_MANIFEST_KEY[role]
    section = payload.get(key)
    return section if isinstance(section, dict) else None


def _legacy_default_role(payload: dict[str, Any] | None) -> ChampionRole:
    if isinstance(payload, dict) and str(payload.get("legacy_default_role") or "").strip() == "future_shortlist":
        return "future_shortlist"
    return "valuation"


def _resolve_role_artifact_paths(role: ChampionRole) -> dict[str, Path]:
    manifest_payload = _load_manifest_payload()
    manifest_env_raw = os.getenv(MODEL_MANIFEST_ENV, "").strip()
    role_envs = ROLE_ENV_NAMES[role]
    test_env = os.getenv(role_envs["test"], "").strip()
    val_env = os.getenv(role_envs["val"], "").strip()
    metrics_env = os.getenv(role_envs["metrics"], "").strip()
    legacy_paths = {
        "test_predictions": _resolve_path(*SPLIT_TO_PATH["test"]),
        "val_predictions": _resolve_path(*SPLIT_TO_PATH["val"]),
        "metrics": _resolve_path(METRICS_ENV, DEFAULT_METRICS),
    }
    if test_env or val_env or metrics_env:
        return {
            "test_predictions": Path(test_env) if test_env else legacy_paths["test_predictions"],
            "val_predictions": Path(val_env) if val_env else legacy_paths["val_predictions"],
            "metrics": Path(metrics_env) if metrics_env else legacy_paths["metrics"],
        }

    legacy_env_explicit = any(
        os.getenv(env_name, "").strip() for env_name in (TEST_PRED_ENV, VAL_PRED_ENV, METRICS_ENV)
    )
    manifest_matches_legacy = bool(
        manifest_payload
        and _manifest_targets_active_artifacts(
            manifest_payload,
            test_path=legacy_paths["test_predictions"],
            val_path=legacy_paths["val_predictions"],
            metrics_path=legacy_paths["metrics"],
        )
    )
    can_use_manifest = bool(manifest_payload) and (
        bool(manifest_env_raw) or not legacy_env_explicit or manifest_matches_legacy
    )
    if can_use_manifest:
        section = _manifest_role_section(manifest_payload, role)
        if isinstance(section, dict):
            artifacts = section.get("artifacts")
            if isinstance(artifacts, dict):
                test_meta = artifacts.get("test_predictions")
                val_meta = artifacts.get("val_predictions")
                metrics_meta = artifacts.get("metrics")
                if all(isinstance(meta, dict) and meta.get("path") for meta in (test_meta, val_meta, metrics_meta)):
                    return {
                        "test_predictions": Path(str(test_meta["path"])),
                        "val_predictions": Path(str(val_meta["path"])),
                        "metrics": Path(str(metrics_meta["path"])),
                    }

        if manifest_payload is not None and _legacy_default_role(manifest_payload) == role:
            artifacts = manifest_payload.get("artifacts")
            if isinstance(artifacts, dict):
                test_meta = artifacts.get("test_predictions")
                val_meta = artifacts.get("val_predictions")
                metrics_meta = artifacts.get("metrics")
                if all(isinstance(meta, dict) and meta.get("path") for meta in (test_meta, val_meta, metrics_meta)):
                    return {
                        "test_predictions": Path(str(test_meta["path"])),
                        "val_predictions": Path(str(val_meta["path"])),
                        "metrics": Path(str(metrics_meta["path"])),
                    }

    return legacy_paths


def _prediction_cache_version(*paths: Path) -> tuple[tuple[str, int], ...]:
    return tuple(
        (_normalized_path_str(path), int(path.stat().st_mtime_ns) if path.exists() else -1)
        for path in paths
    )


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _watchlist_path() -> Path:
    return _resolve_path(WATCHLIST_PATH_ENV, DEFAULT_WATCHLIST_PATH)


def _file_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "size_bytes": None,
            "mtime_utc": None,
            "mtime_epoch": None,
        }
    stat = path.stat()
    dt = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    return {
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime_utc": dt,
        "mtime_epoch": float(stat.st_mtime),
    }


def _safe_numeric(frame: pd.DataFrame, col: str) -> None:
    if col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _replace_inf_with_nan(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return out
    numeric = out.loc[:, numeric_cols]
    out.loc[:, numeric_cols] = numeric.where(np.isfinite(numeric), np.nan)
    return out


def get_resolved_artifact_paths() -> dict[str, str]:
    test_path = _resolve_path(*SPLIT_TO_PATH["test"])
    val_path = _resolve_path(*SPLIT_TO_PATH["val"])
    metrics_path = _resolve_path(METRICS_ENV, DEFAULT_METRICS)
    return {
        "test_predictions_path": str(test_path),
        "val_predictions_path": str(val_path),
        "metrics_path": str(metrics_path),
    }


def get_active_artifacts() -> dict[str, Any]:
    paths = get_resolved_artifact_paths()
    test_path = Path(paths["test_predictions_path"])
    val_path = Path(paths["val_predictions_path"])
    metrics_path = Path(paths["metrics_path"])
    valuation_paths = _resolve_role_artifact_paths("valuation")
    future_paths = _resolve_role_artifact_paths("future_shortlist")
    return {
        "test_predictions_path": str(test_path),
        "val_predictions_path": str(val_path),
        "metrics_path": str(metrics_path),
        "test_predictions_sha256": _sha256_file(test_path),
        "val_predictions_sha256": _sha256_file(val_path),
        "metrics_sha256": _sha256_file(metrics_path),
        "prediction_service_base_role": "valuation",
        "shortlist_overlay_role": "future_shortlist",
        "valuation": {
            "test_predictions_path": str(valuation_paths["test_predictions"]),
            "val_predictions_path": str(valuation_paths["val_predictions"]),
            "metrics_path": str(valuation_paths["metrics"]),
            "test_predictions_sha256": _sha256_file(valuation_paths["test_predictions"]),
            "val_predictions_sha256": _sha256_file(valuation_paths["val_predictions"]),
            "metrics_sha256": _sha256_file(valuation_paths["metrics"]),
        },
        "future_shortlist": {
            "test_predictions_path": str(future_paths["test_predictions"]),
            "val_predictions_path": str(future_paths["val_predictions"]),
            "metrics_path": str(future_paths["metrics"]),
            "test_predictions_sha256": _sha256_file(future_paths["test_predictions"]),
            "val_predictions_sha256": _sha256_file(future_paths["val_predictions"]),
            "metrics_sha256": _sha256_file(future_paths["metrics"]),
        },
    }


def validate_strict_artifact_env() -> None:
    missing_env = [env_name for env_name in _REQUIRED_ARTIFACT_ENVS if not os.getenv(env_name, "").strip()]
    if missing_env:
        raise RuntimeError(
            "Strict artifacts mode is enabled, but required env vars are missing: "
            + ", ".join(missing_env)
        )

    missing_files: list[str] = []
    for env_name in _REQUIRED_ARTIFACT_ENVS:
        raw = os.getenv(env_name, "").strip()
        path = Path(raw)
        if not path.exists():
            missing_files.append(f"{env_name}={path}")
    if missing_files:
        raise RuntimeError(
            "Strict artifacts mode is enabled, but artifact files do not exist: "
            + ", ".join(missing_files)
        )


def _clean_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out = out.loc[:, ~out.columns.duplicated()].copy()

    # Some exports may accidentally contain a duplicated header row.
    if "player_id" in out.columns:
        out = out[out["player_id"].astype(str).str.lower() != "player_id"].copy()

    numeric_cols = [
        "market_value_eur",
        "expected_value_eur",
        "fair_value_eur",
        "expected_value_low_eur",
        "expected_value_high_eur",
        "value_diff",
        "value_abs_error",
        "value_gap_eur",
        "value_gap_conservative_eur",
        "undervaluation_confidence",
        "undervaluation_score",
        "prior_mae_eur",
        "prior_medae_eur",
        "prior_p75ae_eur",
        "prior_qae_eur",
        "prior_interval_q",
        "age",
        "minutes",
        "sofa_minutesPlayed",
        "season_end_year",
    ]
    for col in numeric_cols:
        _safe_numeric(out, col)

    if "undervalued_flag" in out.columns:
        out["undervalued_flag"] = pd.to_numeric(out["undervalued_flag"], errors="coerce").fillna(0).astype(int)

    if "position_group" in out.columns:
        out["position_group"] = out["position_group"].astype(str).str.upper()
    if "model_position" in out.columns:
        out["model_position"] = out["model_position"].astype(str).str.upper()

    return out


def _load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ArtifactNotFoundError(f"Predictions artifact not found: {path}")
    frame = pd.read_csv(path, low_memory=False)
    return _clean_prediction_frame(frame)


def _get_role_predictions(role: ChampionRole, split: Split) -> pd.DataFrame:
    paths = _resolve_role_artifact_paths(role)
    path = paths["test_predictions"] if split == "test" else paths["val_predictions"]
    if not path.exists():
        raise ArtifactNotFoundError(f"{role} {split} predictions artifact not found: {path}")

    cache_key = f"{role}:{split}"
    version = _prediction_cache_version(path)
    cached = _PRED_CACHE.get(cache_key)
    if cached and cached.version == version:
        return cached.frame.copy()

    frame = _load_predictions(path)
    _PRED_CACHE[cache_key] = _FrameCache(key=cache_key, version=version, frame=frame)
    return frame.copy()


def _merge_future_shortlist_overlay(base_frame: pd.DataFrame, future_frame: pd.DataFrame) -> pd.DataFrame:
    overlay_cols = [col for col in FUTURE_OVERLAY_COLUMNS if col in future_frame.columns and col not in base_frame.columns]
    if not overlay_cols:
        return base_frame.copy()

    merge_keys: list[str] = []
    for candidate in (["player_id", "season"], ["player_id"], ["name", "club", "season"]):
        if all(col in base_frame.columns and col in future_frame.columns for col in candidate):
            merge_keys = candidate
            break
    if not merge_keys:
        return base_frame.copy()

    overlay = future_frame.loc[:, merge_keys + overlay_cols].copy()
    overlay = overlay.drop_duplicates(subset=merge_keys, keep="first")
    return base_frame.merge(overlay, on=merge_keys, how="left")


def get_predictions(split: Split = "test") -> pd.DataFrame:
    valuation_paths = _resolve_role_artifact_paths("valuation")
    future_paths = _resolve_role_artifact_paths("future_shortlist")
    valuation_path = valuation_paths["test_predictions"] if split == "test" else valuation_paths["val_predictions"]
    future_path = future_paths["test_predictions"] if split == "test" else future_paths["val_predictions"]

    merged_cache_key = f"merged:{split}"
    merged_version = _prediction_cache_version(valuation_path, future_path)
    cached = _PRED_CACHE.get(merged_cache_key)
    if cached and cached.version == merged_version:
        return cached.frame.copy()

    base_frame = _get_role_predictions("valuation", split)
    if _normalized_path_str(valuation_path) == _normalized_path_str(future_path):
        _PRED_CACHE[merged_cache_key] = _FrameCache(key=merged_cache_key, version=merged_version, frame=base_frame)
        return base_frame.copy()

    future_frame = _get_role_predictions("future_shortlist", split)
    merged = _merge_future_shortlist_overlay(base_frame, future_frame)
    _PRED_CACHE[merged_cache_key] = _FrameCache(key=merged_cache_key, version=merged_version, frame=merged)
    return merged.copy()


def get_metrics(role: ChampionRole = "valuation") -> dict[str, Any]:
    path = _resolve_role_artifact_paths(role)["metrics"]
    if not path.exists():
        raise ArtifactNotFoundError(f"{role} metrics artifact not found: {path}")

    cache_key = f"{role}:{_normalized_path_str(path)}"
    mtime = path.stat().st_mtime_ns
    cached = _METRICS_CACHE.get(cache_key)
    if cached and cached[0] == path and cached[1] == mtime:
        return dict(cached[2])

    payload = json.loads(path.read_text(encoding="utf-8"))
    _METRICS_CACHE[cache_key] = (path, mtime, payload)
    return dict(payload)


def get_benchmark_report() -> dict[str, Any]:
    report_path = _resolve_path(BENCHMARK_REPORT_ENV, DEFAULT_BENCHMARK_REPORT)
    if report_path.exists():
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        payload["_meta"] = {
            "source": "file",
            "path": str(report_path),
            "mtime_utc": _file_meta(report_path).get("mtime_utc"),
            "sha256": _sha256_file(report_path),
        }
        return payload

    valuation_paths = _resolve_role_artifact_paths("valuation")
    metrics_path = valuation_paths["metrics"]
    test_path = valuation_paths["test_predictions"]
    payload = build_market_value_benchmark_payload(
        metrics_path=str(metrics_path),
        predictions_path=str(test_path),
    )
    payload["_meta"] = {
        "source": "derived",
        "path": str(report_path),
        "mtime_utc": None,
        "sha256": None,
    }
    return payload


def _to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    out = _replace_inf_with_nan(frame)
    out = out.where(pd.notna(out), None)
    return out.to_dict(orient="records")


def _minutes_series(frame: pd.DataFrame) -> pd.Series:
    if "minutes" in frame.columns:
        return pd.to_numeric(frame["minutes"], errors="coerce")
    if "sofa_minutesPlayed" in frame.columns:
        return pd.to_numeric(frame["sofa_minutesPlayed"], errors="coerce")
    return pd.Series(np.nan, index=frame.index)


def _position_series(frame: pd.DataFrame) -> pd.Series:
    if "model_position" in frame.columns:
        return frame["model_position"].astype(str).str.upper()
    if "position_group" in frame.columns:
        return frame["position_group"].astype(str).str.upper()
    return pd.Series("UNK", index=frame.index)


def _prediction_value_column(frame: pd.DataFrame) -> str:
    if "fair_value_eur" in frame.columns:
        return "fair_value_eur"
    if "expected_value_eur" in frame.columns:
        return "expected_value_eur"
    return "fair_value_eur"


def _ensure_value_segment(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "value_segment" in out.columns:
        out["value_segment"] = out["value_segment"].astype(str)
        return out
    market = pd.to_numeric(out.get("market_value_eur"), errors="coerce")
    segment = pd.Series("unknown", index=out.index, dtype=object)
    segment.loc[(market >= 0.0) & (market < 5_000_000.0)] = "under_5m"
    segment.loc[(market >= 5_000_000.0) & (market < 20_000_000.0)] = "5m_to_20m"
    segment.loc[market >= 20_000_000.0] = "over_20m"
    out["value_segment"] = segment.astype(str)
    return out


def _to_numeric_series(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce")


def _build_capped_gap_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    cons = _to_numeric_series(out, "value_gap_conservative_eur")
    raw = _to_numeric_series(out, "value_gap_eur")
    cons = cons.fillna(raw)

    prior_q = _to_numeric_series(out, "prior_qae_eur")
    prior_p75 = _to_numeric_series(out, "prior_p75ae_eur")
    prior_mae = _to_numeric_series(out, "prior_mae_eur")

    cap_df = pd.DataFrame(
        {
            "q": np.where(prior_q > 0, 2.5 * prior_q, np.nan),
            "p75": np.where(prior_p75 > 0, 3.0 * prior_p75, np.nan),
            "mae": np.where(prior_mae > 0, 4.0 * prior_mae, np.nan),
        },
        index=out.index,
    )
    cap_threshold = cap_df.min(axis=1, skipna=True)
    capped = cons.copy()
    mask = cons.notna() & (cons > 0) & cap_threshold.notna()
    capped.loc[mask] = np.minimum(cons.loc[mask], cap_threshold.loc[mask])
    cap_applied = mask & (capped + 1.0 < cons)
    cap_ratio = cons / np.maximum(cap_threshold, 1.0)

    out["value_gap_cap_threshold_eur"] = cap_threshold
    out["value_gap_capped_eur"] = capped
    out["value_gap_cap_applied"] = cap_applied.astype(int)
    out["value_gap_cap_ratio"] = cap_ratio.where(np.isfinite(cap_ratio), np.nan)
    return out


def _normalized_path_str(path: Path | str) -> str:
    return os.path.normcase(str(Path(path)))


def _manifest_targets_active_artifacts(
    payload: dict[str, Any],
    *,
    test_path: Path,
    val_path: Path,
    metrics_path: Path,
) -> bool:
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        return False

    expectations = {
        "test_predictions": test_path,
        "val_predictions": val_path,
        "metrics": metrics_path,
    }
    for key, expected in expectations.items():
        section = artifacts.get(key)
        if not isinstance(section, dict):
            return False
        raw_path = section.get("path")
        if not raw_path:
            return False
        if _normalized_path_str(raw_path) != _normalized_path_str(expected):
            return False
    return True


def _fit_residual_calibrator(val_frame: pd.DataFrame, min_samples: int) -> dict[str, Any]:
    if val_frame.empty:
        return {"min_samples": int(min_samples), "global_adjustment_eur": 0.0}

    pred_col = _prediction_value_column(val_frame)
    if pred_col not in val_frame.columns or "market_value_eur" not in val_frame.columns:
        return {"min_samples": int(min_samples), "global_adjustment_eur": 0.0}

    work = _ensure_value_segment(val_frame)
    work["league_norm"] = work.get("league", pd.Series("", index=work.index)).astype(str).str.strip().str.casefold()
    work["position_norm"] = _position_series(work).astype(str).str.upper()
    work["pred_eur"] = pd.to_numeric(work[pred_col], errors="coerce")
    work["market_eur"] = pd.to_numeric(work["market_value_eur"], errors="coerce")
    work["residual_eur"] = work["market_eur"] - work["pred_eur"]
    work = work[np.isfinite(work["residual_eur"])].copy()

    if work.empty:
        return {"min_samples": int(min_samples), "global_adjustment_eur": 0.0}

    def _group_map(cols: list[str], threshold: int) -> dict[tuple[str, ...], float]:
        grouped = (
            work.groupby(cols, dropna=False)["residual_eur"]
            .agg(median_residual="median", n="count")
            .reset_index()
        )
        grouped = grouped[grouped["n"] >= int(max(threshold, 1))].copy()
        out: dict[tuple[str, ...], float] = {}
        for _, row in grouped.iterrows():
            key = tuple(str(row[c]) for c in cols)
            # One-sided conservative correction: only reduce optimistic predictions.
            out[key] = float(min(float(row["median_residual"]), 0.0))
        return out

    lvl1 = _group_map(["league_norm", "position_norm", "value_segment"], threshold=min_samples)
    lvl2 = _group_map(["league_norm", "position_norm"], threshold=max(min_samples // 2, 12))
    lvl3 = _group_map(["position_norm", "value_segment"], threshold=max(min_samples // 2, 12))
    lvl4 = _group_map(["value_segment"], threshold=max(min_samples // 3, 8))
    global_adjustment = 0.0
    if len(work) >= int(max(min_samples, 1)):
        global_adjustment = float(min(float(work["residual_eur"].median()), 0.0))

    return {
        "min_samples": int(min_samples),
        "level1": lvl1,
        "level2": lvl2,
        "level3": lvl3,
        "level4": lvl4,
        "global_adjustment_eur": global_adjustment,
    }


def _get_residual_calibrator() -> dict[str, Any]:
    global _RESIDUAL_CALIBRATION_CACHE
    if not _env_flag(ENABLE_RESIDUAL_CALIBRATION_ENV, default=True):
        return {"enabled": False, "global_adjustment_eur": 0.0}

    min_samples_raw = os.getenv(CALIBRATION_MIN_SAMPLES_ENV, "30").strip()
    try:
        min_samples = max(int(min_samples_raw), 1)
    except ValueError:
        min_samples = 30

    val_path = _resolve_path(*SPLIT_TO_PATH["val"])
    if not val_path.exists():
        return {"enabled": False, "global_adjustment_eur": 0.0}

    mtime = val_path.stat().st_mtime_ns
    if (
        _RESIDUAL_CALIBRATION_CACHE is not None
        and _RESIDUAL_CALIBRATION_CACHE.path == val_path
        and _RESIDUAL_CALIBRATION_CACHE.mtime_ns == mtime
        and _RESIDUAL_CALIBRATION_CACHE.min_samples == min_samples
    ):
        return dict(_RESIDUAL_CALIBRATION_CACHE.payload)

    val_frame = get_predictions(split="val")
    payload = _fit_residual_calibrator(val_frame=val_frame, min_samples=min_samples)
    payload["enabled"] = True
    _RESIDUAL_CALIBRATION_CACHE = _ResidualCalibrationCache(
        path=val_path,
        mtime_ns=mtime,
        min_samples=min_samples,
        payload=dict(payload),
    )
    return payload


def _lookup_residual_adjustment(row: pd.Series, calibrator: dict[str, Any]) -> float:
    if not calibrator.get("enabled", False):
        return 0.0
    league = str(row.get("league_norm", "")).strip().casefold()
    pos = str(row.get("position_norm", "")).strip().upper()
    seg = str(row.get("value_segment", "unknown"))

    lvl1 = calibrator.get("level1", {})
    lvl2 = calibrator.get("level2", {})
    lvl3 = calibrator.get("level3", {})
    lvl4 = calibrator.get("level4", {})

    if (league, pos, seg) in lvl1:
        return float(lvl1[(league, pos, seg)])
    if (league, pos) in lvl2:
        return float(lvl2[(league, pos)])
    if (pos, seg) in lvl3:
        return float(lvl3[(pos, seg)])
    if (seg,) in lvl4:
        return float(lvl4[(seg,)])
    return float(calibrator.get("global_adjustment_eur", 0.0) or 0.0)


def _apply_residual_calibration(frame: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_value_segment(frame)
    if out.empty:
        return out

    # If artifacts already include a training-time residual calibration, avoid double-calibration.
    if "residual_calibration_applied" in out.columns:
        applied = pd.to_numeric(out["residual_calibration_applied"], errors="coerce").fillna(0)
        if (applied > 0).any() and "expected_value_calibration_eur" in out.columns:
            return _build_capped_gap_columns(out)

    pred_col = _prediction_value_column(out)
    if pred_col not in out.columns:
        out = _build_capped_gap_columns(out)
        return out

    out["league_norm"] = out.get("league", pd.Series("", index=out.index)).astype(str).str.strip().str.casefold()
    out["position_norm"] = _position_series(out).astype(str).str.upper()

    calibrator = _get_residual_calibrator()
    if calibrator.get("enabled", False):
        out["expected_value_calibration_eur"] = out.apply(
            lambda row: _lookup_residual_adjustment(row, calibrator), axis=1
        )
    else:
        out["expected_value_calibration_eur"] = 0.0

    pred_raw = _to_numeric_series(out, pred_col)
    out["expected_value_raw_eur"] = pred_raw
    out["expected_value_eur"] = pred_raw + pd.to_numeric(out["expected_value_calibration_eur"], errors="coerce").fillna(0.0)
    out["fair_value_eur"] = out["expected_value_eur"]

    if "expected_value_low_eur" in out.columns:
        low_raw = _to_numeric_series(out, "expected_value_low_eur")
        out["expected_value_low_raw_eur"] = low_raw
        out["expected_value_low_eur"] = (low_raw + out["expected_value_calibration_eur"]).clip(lower=0.0)
    if "expected_value_high_eur" in out.columns:
        high_raw = _to_numeric_series(out, "expected_value_high_eur")
        out["expected_value_high_raw_eur"] = high_raw
        out["expected_value_high_eur"] = high_raw + out["expected_value_calibration_eur"]

    market = _to_numeric_series(out, "market_value_eur")
    computed_raw_gap = out["expected_value_eur"] - market
    existing_raw_gap = _to_numeric_series(out, "value_gap_raw_eur")
    existing_gap = _to_numeric_series(out, "value_gap_eur")
    out["value_gap_raw_eur"] = existing_raw_gap.where(existing_raw_gap.notna(), existing_gap)
    out["value_gap_raw_eur"] = out["value_gap_raw_eur"].where(out["value_gap_raw_eur"].notna(), computed_raw_gap)
    out["value_gap_eur"] = out["value_gap_raw_eur"]

    existing_cons_gap = _to_numeric_series(out, "value_gap_conservative_eur")
    if "expected_value_low_eur" in out.columns:
        conservative_gap = _to_numeric_series(out, "expected_value_low_eur") - market
    else:
        conservative_gap = out["value_gap_raw_eur"]
    out["value_gap_conservative_eur"] = existing_cons_gap.where(existing_cons_gap.notna(), conservative_gap)

    out = _build_capped_gap_columns(out)
    return out


def _prepare_predictions_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = _apply_residual_calibration(frame)
    out = add_history_strength_features(out)
    out = _replace_inf_with_nan(out)
    return out


def _ranking_gap_series(frame: pd.DataFrame) -> pd.Series:
    if "value_gap_capped_eur" in frame.columns:
        return pd.to_numeric(frame["value_gap_capped_eur"], errors="coerce")
    if "value_gap_conservative_eur" in frame.columns:
        return pd.to_numeric(frame["value_gap_conservative_eur"], errors="coerce")
    return pd.to_numeric(frame.get("value_gap_eur"), errors="coerce")


def _history_factor_series(frame: pd.DataFrame) -> pd.Series:
    if "history_strength_score" not in frame.columns:
        return pd.Series(1.0, index=frame.index, dtype=float)

    strength = pd.to_numeric(frame["history_strength_score"], errors="coerce") / 100.0
    strength = strength.clip(lower=0.0, upper=1.0)

    if "history_strength_coverage" in frame.columns:
        coverage = pd.to_numeric(frame["history_strength_coverage"], errors="coerce").clip(lower=0.0, upper=1.0)
    else:
        coverage = pd.Series(1.0, index=frame.index, dtype=float)

    factor = (0.85 + 0.35 * strength) * (0.90 + 0.10 * coverage)
    sparse = coverage < 0.35
    factor = factor.where(~sparse, 1.0)
    return factor.fillna(1.0)


def _infer_future_outcome_label(frame: pd.DataFrame) -> tuple[pd.Series | None, str | None]:
    candidates = [
        "future_outcome_label",
        "is_future_undervalued_success",
        "future_success",
        "value_growth_next_season_eur",
        "future_value_growth_eur",
    ]
    for col in candidates:
        if col not in frame.columns:
            continue
        series = pd.to_numeric(frame[col], errors="coerce")
        if col.endswith("_eur"):
            label = (series > 0).astype(float)
        else:
            label = (series > 0).astype(float)
        if label.notna().sum() >= 20:
            return label, col
    return None, None


def _precision_at_k(
    frame: pd.DataFrame,
    *,
    score_col: str,
    k_values: Sequence[int] = (10, 25, 50, 100),
) -> dict[str, Any]:
    if score_col not in frame.columns:
        return {"available": False, "reason": f"missing_score_col:{score_col}"}

    labels, label_col = _infer_future_outcome_label(frame)
    if labels is None or label_col is None:
        return {"available": False, "reason": "missing_future_outcome_label"}

    work = frame.copy()
    work["_score"] = pd.to_numeric(work[score_col], errors="coerce")
    work["_label"] = pd.to_numeric(labels, errors="coerce")
    work = work[np.isfinite(work["_score"]) & np.isfinite(work["_label"])].copy()
    if work.empty:
        return {"available": False, "reason": "no_rows_with_labels"}

    work = work.sort_values("_score", ascending=False).reset_index(drop=True)
    out_rows: list[dict[str, Any]] = []
    for k in sorted({int(max(k, 1)) for k in k_values}):
        top = work.head(k)
        n = int(len(top))
        if n == 0:
            continue
        precision = float((top["_label"] > 0).mean())
        out_rows.append({"k": int(k), "n": n, "precision": precision})

    return {
        "available": bool(out_rows),
        "label_column": label_col,
        "n_labeled_rows": int(len(work)),
        "rows": out_rows,
    }


def _preferred_scout_score_col(frame: pd.DataFrame, *, default: str) -> str:
    for col in ("future_scout_blend_score", "future_growth_probability", default):
        if col in frame.columns:
            return col
    return default


def _ranking_basis_for_score_col(score_col: str, *, default_basis: str) -> str:
    if score_col == "future_scout_blend_score":
        return "future_target_tuned_blend"
    if score_col == "future_growth_probability":
        return "future_target_probability"
    return default_basis


def query_predictions(
    split: Split = "test",
    season: str | None = None,
    league: str | None = None,
    club: str | None = None,
    position: str | None = None,
    role_keys: Sequence[str] | None = None,
    min_minutes: float | None = None,
    min_age: float | None = None,
    max_age: float | None = None,
    max_market_value_eur: float | None = None,
    max_contract_years_left: float | None = None,
    non_big5_only: bool = False,
    undervalued_only: bool = False,
    min_confidence: float | None = None,
    min_value_gap_eur: float | None = None,
    sort_by: str = "value_gap_capped_eur",
    sort_order: Literal["asc", "desc"] = "desc",
    limit: int = 100,
    offset: int = 0,
    columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    frame = _prepare_predictions_frame(get_predictions(split=split))

    if season and "season" in frame.columns:
        frame = frame[frame["season"].astype(str) == str(season)].copy()
    if league and "league" in frame.columns:
        frame = frame[frame["league"].astype(str).str.casefold() == str(league).casefold()].copy()
    if club and "club" in frame.columns:
        frame = frame[frame["club"].astype(str).str.casefold() == str(club).casefold()].copy()
    if position:
        pos_series = _position_series(frame)
        frame = frame[pos_series == str(position).upper()].copy()

    if role_keys:
        wanted_roles = {str(role).strip().upper() for role in role_keys if str(role).strip()}
        if wanted_roles:
            inferred_roles = frame.apply(_infer_position_role_key, axis=1)
            frame = frame[inferred_roles.isin(wanted_roles)].copy()

    if min_minutes is not None:
        frame = frame[_minutes_series(frame).fillna(0) >= float(min_minutes)].copy()
    if (min_age is not None or max_age is not None) and "age" in frame.columns:
        age = pd.to_numeric(frame["age"], errors="coerce")
        if min_age is not None:
            frame = frame[age >= float(min_age)].copy()
            age = pd.to_numeric(frame["age"], errors="coerce")
        if max_age is not None:
            frame = frame[age <= float(max_age)].copy()

    if max_market_value_eur is not None and "market_value_eur" in frame.columns:
        market = pd.to_numeric(frame["market_value_eur"], errors="coerce")
        frame = frame[market <= float(max_market_value_eur)].copy()

    if max_contract_years_left is not None and "contract_years_left" in frame.columns:
        contract_years = pd.to_numeric(frame["contract_years_left"], errors="coerce")
        frame = frame[contract_years.notna() & (contract_years <= float(max_contract_years_left))].copy()

    if non_big5_only and "league" in frame.columns:
        league_norm = frame["league"].astype(str).str.strip().str.casefold()
        frame = frame[~league_norm.isin(BIG5_LEAGUES)].copy()

    if undervalued_only:
        if "undervalued_flag" in frame.columns:
            frame = frame[pd.to_numeric(frame["undervalued_flag"], errors="coerce").fillna(0) == 1].copy()
        elif "value_gap_conservative_eur" in frame.columns:
            frame = frame[pd.to_numeric(frame["value_gap_conservative_eur"], errors="coerce") > 0].copy()

    if min_confidence is not None and "undervaluation_confidence" in frame.columns:
        conf = pd.to_numeric(frame["undervaluation_confidence"], errors="coerce")
        frame = frame[conf >= float(min_confidence)].copy()

    if min_value_gap_eur is not None:
        gap = _ranking_gap_series(frame)
        frame = frame[gap >= float(min_value_gap_eur)].copy()

    total = int(len(frame))
    if sort_by not in frame.columns:
        fallback = (
            "value_gap_capped_eur"
            if "value_gap_capped_eur" in frame.columns
            else (
                "value_gap_conservative_eur"
                if "value_gap_conservative_eur" in frame.columns
                else ("value_gap_eur" if "value_gap_eur" in frame.columns else frame.columns[0])
            )
        )
        sort_by = fallback
    ascending = sort_order == "asc"
    frame = frame.sort_values(sort_by, ascending=ascending, na_position="last")

    if columns:
        keep = [c for c in columns if c in frame.columns]
        if keep:
            frame = frame[keep].copy()

    start = max(int(offset), 0)
    end = start + max(int(limit), 0)
    page = frame.iloc[start:end].copy()

    return {
        "split": split,
        "total": total,
        "count": int(len(page)),
        "limit": int(limit),
        "offset": int(offset),
        "sort_by": sort_by,
        "sort_order": sort_order,
        "items": _to_records(page),
    }


def query_shortlist(
    split: Split = "test",
    top_n: int = 100,
    min_minutes: float = 900,
    min_age: float | None = None,
    max_age: float | None = 25,
    positions: Sequence[str] | None = None,
    role_keys: Sequence[str] | None = None,
    non_big5_only: bool = False,
    max_market_value_eur: float | None = None,
    max_contract_years_left: float | None = None,
) -> dict[str, Any]:
    frame = _prepare_predictions_frame(get_predictions(split=split))

    pred_col = "fair_value_eur" if "fair_value_eur" in frame.columns else "expected_value_eur"
    if pred_col not in frame.columns or "market_value_eur" not in frame.columns:
        raise ValueError("Prediction artifact does not include required columns.")

    work = frame.copy()
    work["market_value_eur"] = pd.to_numeric(work["market_value_eur"], errors="coerce")
    work[pred_col] = pd.to_numeric(work[pred_col], errors="coerce")

    if "value_gap_conservative_eur" not in work.columns:
        work["value_gap_conservative_eur"] = work[pred_col] - work["market_value_eur"]
    else:
        work["value_gap_conservative_eur"] = pd.to_numeric(work["value_gap_conservative_eur"], errors="coerce")

    work["ranking_gap_eur"] = _ranking_gap_series(work).fillna(work["value_gap_conservative_eur"])

    if "undervaluation_confidence" not in work.columns:
        denom = max(float(np.nanmedian(np.abs(work[pred_col] - work["market_value_eur"]))), 1.0)
        work["undervaluation_confidence"] = work["value_gap_conservative_eur"] / denom
    else:
        work["undervaluation_confidence"] = pd.to_numeric(work["undervaluation_confidence"], errors="coerce")

    work["minutes_used"] = _minutes_series(work).fillna(0.0)
    work["position_used"] = _position_series(work)
    work["age_num"] = pd.to_numeric(work["age"], errors="coerce") if "age" in work.columns else np.nan

    work = work[work["ranking_gap_eur"] > 0].copy()
    work = work[work["minutes_used"] >= float(min_minutes)].copy()

    if min_age is not None:
        work = work[work["age_num"].fillna(-1) >= float(min_age)].copy()
    if max_age is not None:
        work = work[work["age_num"].fillna(999) <= float(max_age)].copy()

    if positions:
        pos_set = {p.upper() for p in positions}
        work = work[work["position_used"].isin(pos_set)].copy()

    if role_keys:
        wanted_roles = {str(role).strip().upper() for role in role_keys if str(role).strip()}
        if wanted_roles:
            inferred_roles = work.apply(_infer_position_role_key, axis=1)
            work = work[inferred_roles.isin(wanted_roles)].copy()

    if non_big5_only:
        league_norm = (
            work["league"].astype(str).str.strip().str.casefold()
            if "league" in work.columns
            else pd.Series("", index=work.index, dtype=str)
        )
        work = work[~league_norm.isin(BIG5_LEAGUES)].copy()

    if max_market_value_eur is not None:
        work = work[work["market_value_eur"].fillna(np.inf) <= float(max_market_value_eur)].copy()

    if max_contract_years_left is not None and "contract_years_left" in work.columns:
        contract_years = pd.to_numeric(work["contract_years_left"], errors="coerce")
        work = work[contract_years.notna() & (contract_years <= float(max_contract_years_left))].copy()

    reliability = np.clip(work["minutes_used"] / 1800.0, 0.3, 1.2)
    confidence = work["undervaluation_confidence"].clip(lower=0.0).fillna(0.0)
    age = work["age_num"].fillna(26.0)
    age_factor = np.where(age <= 23, 1.15, np.where(age <= 26, 1.0, 0.85))
    history_factor = _history_factor_series(work)

    work["shortlist_score"] = (
        (work["ranking_gap_eur"] / 1_000_000.0)
        * np.log1p(confidence)
        * reliability
        * age_factor
        * history_factor
    )
    score_col = _preferred_scout_score_col(work, default="shortlist_score")
    work = work.sort_values(
        [score_col, "ranking_gap_eur", "value_gap_conservative_eur", "undervaluation_confidence"],
        ascending=False,
    )
    shortlist = work.head(max(int(top_n), 0)).copy()
    precision = _precision_at_k(
        work,
        score_col=score_col,
        k_values=(10, 25, 50, int(top_n)),
    )
    return {
        "split": split,
        "total_candidates": int(len(work)),
        "count": int(len(shortlist)),
        "diagnostics": {
            "ranking_basis": _ranking_basis_for_score_col(
                score_col,
                default_basis="guardrailed_gap_confidence_history",
            ),
            "score_column": score_col,
            "precision_at_k": precision,
        },
        "items": _to_records(shortlist),
    }


def query_scout_targets(
    split: Split = "test",
    top_n: int = 100,
    min_minutes: float = 900,
    min_age: float | None = None,
    max_age: float | None = 23,
    min_confidence: float = 0.50,
    min_value_gap_eur: float = 1_000_000.0,
    positions: Sequence[str] | None = None,
    role_keys: Sequence[str] | None = None,
    non_big5_only: bool = True,
    include_leagues: Sequence[str] | None = None,
    exclude_leagues: Sequence[str] | None = None,
    min_expected_value_eur: float | None = None,
    max_expected_value_eur: float | None = None,
    max_market_value_eur: float | None = None,
    max_contract_years_left: float | None = None,
) -> dict[str, Any]:
    frame = _prepare_predictions_frame(get_predictions(split=split))
    pred_col = "fair_value_eur" if "fair_value_eur" in frame.columns else "expected_value_eur"
    if pred_col not in frame.columns or "market_value_eur" not in frame.columns:
        raise ValueError("Prediction artifact does not include required valuation columns.")

    work = frame.copy()
    work["market_value_eur"] = pd.to_numeric(work["market_value_eur"], errors="coerce")
    work[pred_col] = pd.to_numeric(work[pred_col], errors="coerce")

    if "value_gap_conservative_eur" in work.columns:
        work["value_gap_conservative_eur"] = pd.to_numeric(
            work["value_gap_conservative_eur"], errors="coerce"
        )
    else:
        work["value_gap_conservative_eur"] = work[pred_col] - work["market_value_eur"]

    work["ranking_gap_eur"] = _ranking_gap_series(work).fillna(work["value_gap_conservative_eur"])

    if "undervaluation_confidence" in work.columns:
        work["undervaluation_confidence"] = pd.to_numeric(
            work["undervaluation_confidence"], errors="coerce"
        )
    else:
        denom = max(float(np.nanmedian(np.abs(work[pred_col] - work["market_value_eur"]))), 1.0)
        work["undervaluation_confidence"] = work["value_gap_conservative_eur"] / denom

    work["minutes_used"] = _minutes_series(work).fillna(0.0)
    work["position_used"] = _position_series(work)
    work["age_num"] = pd.to_numeric(work["age"], errors="coerce") if "age" in work.columns else np.nan
    work["league_norm"] = (
        work["league"].astype(str).str.strip().str.casefold() if "league" in work.columns else "unknown"
    )

    work = work[work["ranking_gap_eur"] > 0].copy()
    work = work[work["minutes_used"] >= float(min_minutes)].copy()
    work = work[work["undervaluation_confidence"].fillna(0.0) >= float(min_confidence)].copy()
    work = work[work["ranking_gap_eur"].fillna(0.0) >= float(min_value_gap_eur)].copy()

    if min_age is not None:
        work = work[work["age_num"].fillna(-1.0) >= float(min_age)].copy()

    if max_age is not None:
        work = work[work["age_num"].fillna(999.0) <= float(max_age)].copy()

    if positions:
        wanted = {p.strip().upper() for p in positions if str(p).strip()}
        work = work[work["position_used"].isin(wanted)].copy()

    if role_keys:
        wanted_roles = {str(role).strip().upper() for role in role_keys if str(role).strip()}
        if wanted_roles:
            inferred_roles = work.apply(_infer_position_role_key, axis=1)
            work = work[inferred_roles.isin(wanted_roles)].copy()

    if non_big5_only:
        work = work[~work["league_norm"].isin(BIG5_LEAGUES)].copy()

    if include_leagues:
        include_norm = {str(league).strip().casefold() for league in include_leagues if str(league).strip()}
        work = work[work["league_norm"].isin(include_norm)].copy()

    if exclude_leagues:
        exclude_norm = {str(league).strip().casefold() for league in exclude_leagues if str(league).strip()}
        work = work[~work["league_norm"].isin(exclude_norm)].copy()

    if min_expected_value_eur is not None:
        work = work[work[pred_col].fillna(0.0) >= float(min_expected_value_eur)].copy()
    if max_expected_value_eur is not None:
        work = work[work[pred_col].fillna(np.inf) <= float(max_expected_value_eur)].copy()

    if max_market_value_eur is not None:
        work = work[work["market_value_eur"].fillna(np.inf) <= float(max_market_value_eur)].copy()

    if max_contract_years_left is not None and "contract_years_left" in work.columns:
        contract_years = pd.to_numeric(work["contract_years_left"], errors="coerce")
        work = work[contract_years.notna() & (contract_years <= float(max_contract_years_left))].copy()

    confidence = work["undervaluation_confidence"].clip(lower=0.0).fillna(0.0)
    minutes_factor = np.clip(work["minutes_used"] / 1800.0, 0.35, 1.25)
    age = work["age_num"].fillna(26.0)
    age_factor = np.where(age <= 20, 1.25, np.where(age <= 23, 1.12, np.where(age <= 26, 1.0, 0.82)))
    market = work["market_value_eur"].fillna(1_000_000.0).clip(lower=1_000_000.0)
    value_efficiency = (work["ranking_gap_eur"] / market).clip(lower=0.0)
    history_factor = _history_factor_series(work)
    work["scout_target_score"] = (
        (work["ranking_gap_eur"] / 1_000_000.0)
        * (1.0 + np.log1p(confidence))
        * minutes_factor
        * age_factor
        * (1.0 + 0.30 * value_efficiency)
        * history_factor
    )
    score_col = _preferred_scout_score_col(work, default="scout_target_score")

    work = work.sort_values(
        [score_col, "ranking_gap_eur", "value_gap_conservative_eur", "undervaluation_confidence"],
        ascending=False,
    )
    out = work.head(max(int(top_n), 0)).copy()
    precision = _precision_at_k(
        work,
        score_col=score_col,
        k_values=(10, 25, 50, int(top_n)),
    )
    return {
        "split": split,
        "total_candidates": int(len(work)),
        "count": int(len(out)),
        "diagnostics": {
            "ranking_basis": _ranking_basis_for_score_col(
                score_col,
                default_basis="guardrailed_gap_confidence_history_efficiency",
            ),
            "score_column": score_col,
            "precision_at_k": precision,
        },
        "items": _to_records(out),
    }


def get_player_prediction(
    player_id: str,
    split: Split = "test",
    season: str | None = None,
) -> dict[str, Any]:
    frame = _prepare_predictions_frame(get_predictions(split=split))
    if "player_id" not in frame.columns:
        raise ValueError("Prediction artifact does not include 'player_id'.")

    subset = frame[frame["player_id"].astype(str) == str(player_id)].copy()
    if season is not None and "season" in subset.columns:
        subset = subset[subset["season"].astype(str) == str(season)].copy()

    if subset.empty:
        raise ValueError(f"No prediction found for player_id={player_id!r} in split={split}.")

    if season is None:
        sort_cols = []
        if "season_end_year" in subset.columns:
            sort_cols.append("season_end_year")
        if "minutes" in subset.columns:
            sort_cols.append("minutes")
        elif "sofa_minutesPlayed" in subset.columns:
            sort_cols.append("sofa_minutesPlayed")
        if sort_cols:
            subset = subset.sort_values(sort_cols, ascending=False, na_position="last")

    row = subset.iloc[0].to_frame().T
    records = _to_records(row)
    return records[0]


def _safe_float(value: Any) -> float | None:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(parsed) else float(parsed)


def _select_player_row(frame: pd.DataFrame, player_id: str, season: str | None = None) -> pd.Series:
    if "player_id" not in frame.columns:
        raise ValueError("Prediction artifact does not include 'player_id'.")

    subset = frame[frame["player_id"].astype(str) == str(player_id)].copy()
    if season is not None and "season" in subset.columns:
        subset = subset[subset["season"].astype(str) == str(season)].copy()

    if subset.empty:
        raise ValueError(f"No prediction found for player_id={player_id!r} in split.")

    if season is None:
        sort_cols = []
        if "season_end_year" in subset.columns:
            sort_cols.append("season_end_year")
        if "minutes" in subset.columns:
            sort_cols.append("minutes")
        elif "sofa_minutesPlayed" in subset.columns:
            sort_cols.append("sofa_minutesPlayed")
        if sort_cols:
            subset = subset.sort_values(sort_cols, ascending=False, na_position="last")

    return subset.iloc[0].copy()


def _cohort_for_player(frame: pd.DataFrame, row: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
    cohort = frame.copy()
    filters: dict[str, Any] = {}

    player_position = None
    if "model_position" in row.index and pd.notna(row["model_position"]):
        player_position = str(row["model_position"]).upper().strip()
    elif "position_group" in row.index and pd.notna(row["position_group"]):
        player_position = str(row["position_group"]).upper().strip()
    if player_position:
        pos = _position_series(cohort)
        filtered = cohort[pos == player_position].copy()
        if len(filtered) >= 40:
            cohort = filtered
            filters["position"] = player_position

    player_age = _safe_float(row.get("age"))
    if player_age is not None and "age" in cohort.columns:
        ages = pd.to_numeric(cohort["age"], errors="coerce")
        for band, min_rows in ((2.0, 140), (3.0, 90), (4.0, 50)):
            filtered = cohort[(ages >= player_age - band) & (ages <= player_age + band)].copy()
            if len(filtered) >= min_rows:
                cohort = filtered
                filters["age_band"] = f"{max(int(player_age - band), 0)}-{int(player_age + band)}"
                break

    player_season = row.get("season")
    if player_season is not None and not pd.isna(player_season) and "season" in cohort.columns:
        filtered = cohort[cohort["season"].astype(str) == str(player_season)].copy()
        if len(filtered) >= 35:
            cohort = filtered
            filters["season"] = str(player_season)

    return cohort, filters


def _metric_value_corr_abs(cohort: pd.DataFrame, col: str) -> float:
    target_col = None
    if "fair_value_eur" in cohort.columns:
        target_col = "fair_value_eur"
    elif "expected_value_eur" in cohort.columns:
        target_col = "expected_value_eur"
    if target_col is None:
        return 0.0

    x = pd.to_numeric(cohort[col], errors="coerce")
    y = pd.to_numeric(cohort[target_col], errors="coerce")
    aligned = pd.concat([x, y], axis=1).dropna()
    if len(aligned) < 25:
        return 0.0
    corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    if pd.isna(corr):
        return 0.0
    return min(abs(float(corr)), 1.0)


def _normalize_position_key(raw: Any) -> str:
    if raw is None or pd.isna(raw):
        return "UNK"
    token = str(raw).strip().upper()
    if token in {"GK", "DF", "MF", "FW"}:
        return token
    if token in {"BACK", "DEFENDER"} or "DEF" in token:
        return "DF"
    if token in {"MIDFIELD", "MIDFIELDER"} or "MID" in token:
        return "MF"
    if token in {"ATTACK", "ATTACKER", "FORWARD", "STRIKER"} or "WING" in token:
        return "FW"
    if "GOAL" in token:
        return "GK"
    return token


def _safe_text_token(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _position_role_label(role_key: str) -> str:
    return ROLE_KEY_LABELS.get(role_key, role_key)


def _infer_position_role_key(row: pd.Series) -> str:
    family_key = _normalize_position_key(row.get("model_position") or row.get("position_group"))
    if family_key == "GK":
        return "GK"

    primary = " ".join(
        part
        for part in (
            _safe_text_token(row.get("position_main")),
            _safe_text_token(row.get("position")),
            _safe_text_token(row.get("position_alt")),
        )
        if part
    ).lower()

    if family_key == "DF":
        if any(token in primary for token in ("wing-back", "wing back", "fullback", "full back", "left back", "right back")):
            return "FB"
        if "defender, left" in primary or "defender, right" in primary:
            return "FB"
        if any(token in primary for token in ("centre", "center", "back")):
            return "CB"
        return "DF"

    if family_key == "MF":
        if "defensive" in primary:
            return "DM"
        if "attacking" in primary:
            return "AM"
        if "left midfield" in primary or "right midfield" in primary or "wing" in primary:
            return "W"
        if "central" in primary:
            return "CM"
        return "MF"

    if family_key == "FW":
        if "winger" in primary:
            return "W"
        if "second striker" in primary or "support forward" in primary:
            return "SS"
        if "attack, centre" in primary or "striker" in primary or "forward" in primary:
            return "ST"
        return "FW"

    return family_key


def _metric_snapshot(
    *,
    row: pd.Series,
    cohort: pd.DataFrame,
    metric: str,
    direction: int | None = None,
) -> dict[str, Any] | None:
    if metric not in row.index or metric not in cohort.columns:
        return None
    player_value = _safe_float(row.get(metric))
    if player_value is None:
        return None

    series = pd.to_numeric(cohort[metric], errors="coerce").dropna()
    if len(series) < 20:
        return None

    metric_direction = int(direction or ADVANCED_METRIC_DIRECTION.get(metric, 1))
    percentile_raw = float((series <= player_value).mean())
    quality_percentile = percentile_raw if metric_direction > 0 else (1.0 - percentile_raw)
    quality_percentile = float(np.clip(quality_percentile, 0.0, 1.0))

    q25, q50, q75 = np.nanquantile(series.to_numpy(), [0.25, 0.5, 0.75])
    return {
        "metric": metric,
        "label": ADVANCED_METRIC_LABEL.get(metric, metric),
        "direction": "higher_is_better" if metric_direction > 0 else "lower_is_better",
        "player_value": player_value,
        "cohort_p25": float(q25),
        "cohort_median": float(q50),
        "cohort_p75": float(q75),
        "percentile_raw": float(percentile_raw),
        "quality_percentile": quality_percentile,
    }


def _build_metric_snapshot_map(row: pd.Series, cohort: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for metric, _, direction in ADVANCED_METRIC_SPECS:
        snap = _metric_snapshot(row=row, cohort=cohort, metric=metric, direction=direction)
        if snap is not None:
            out[metric] = snap
    return out


def _score_template_fit(
    *,
    metric_map: dict[str, dict[str, Any]],
    targets: dict[str, float],
) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    for metric, target_pct in targets.items():
        snap = metric_map.get(metric)
        if snap is None:
            continue
        observed = float(snap["quality_percentile"])
        target = float(np.clip(target_pct, 0.0, 1.0))
        # Soft match around target percentile. 1.0 is perfect, 0.0 is very far.
        fit_score = max(0.0, 1.0 - abs(observed - target) / 0.75)
        parts.append(
            {
                "metric": metric,
                "label": snap["label"],
                "target_percentile": target,
                "observed_percentile": observed,
                "fit_score": float(fit_score),
            }
        )

    n_targets = max(len(targets), 1)
    coverage = len(parts) / n_targets
    if not parts:
        return {
            "score": 0.0,
            "coverage": 0.0,
            "parts": [],
        }
    mean_fit = float(np.mean([p["fit_score"] for p in parts]))
    score = mean_fit * (0.65 + 0.35 * coverage)
    return {
        "score": float(np.clip(score, 0.0, 1.0)),
        "coverage": float(np.clip(coverage, 0.0, 1.0)),
        "parts": parts,
    }


def _build_player_type_profile(
    *,
    row: pd.Series,
    metric_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    role_key = _infer_position_role_key(row)
    family_key = _normalize_position_key(row.get("model_position") or row.get("position_group"))
    templates = ARCHETYPE_TEMPLATES.get(role_key) or ARCHETYPE_TEMPLATES.get(family_key, ())
    if not templates:
        return {
            "position_key": role_key,
            "position_family_key": family_key,
            "position_label": _position_role_label(role_key),
            "archetype": "Unknown",
            "confidence_0_to_1": 0.0,
            "tier": "low",
            "runner_up": None,
            "candidates": [],
            "summary_text": "Not enough archetype templates for this position.",
        }

    scored: list[dict[str, Any]] = []
    for template in templates:
        fit = _score_template_fit(metric_map=metric_map, targets=template.get("targets", {}))
        scored.append(
            {
                "name": str(template.get("name") or "Unknown"),
                "description": str(template.get("description") or ""),
                "score_0_to_1": fit["score"],
                "coverage_0_to_1": fit["coverage"],
                "matched_metrics": fit["parts"],
            }
        )

    scored.sort(key=lambda x: (x["score_0_to_1"], x["coverage_0_to_1"]), reverse=True)
    best = scored[0]
    runner = scored[1] if len(scored) > 1 else None
    conf = float(best["score_0_to_1"])
    if runner is not None:
        conf = float(np.clip(conf - 0.15 * max(0.0, runner["score_0_to_1"]), 0.0, 1.0))

    tier = "low"
    if conf >= 0.72:
        tier = "high"
    elif conf >= 0.52:
        tier = "medium"

    runner_name = runner["name"] if isinstance(runner, dict) else None
    summary = f"Role lens: {_position_role_label(role_key)}. Archetype: {best['name']} ({tier} confidence)."
    if runner_name:
        summary += f" Runner-up: {runner_name}."

    return {
        "position_key": role_key,
        "position_family_key": family_key,
        "position_label": _position_role_label(role_key),
        "archetype": best["name"],
        "description": best["description"],
        "confidence_0_to_1": conf,
        "tier": tier,
        "runner_up": runner_name,
        "candidates": scored[:3],
        "summary_text": summary,
    }


def _build_formation_fit_profile(
    *,
    row: pd.Series,
    metric_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    role_key = _infer_position_role_key(row)
    family_key = _normalize_position_key(row.get("model_position") or row.get("position_group"))
    templates = FORMATION_FIT_TEMPLATES.get(role_key) or FORMATION_FIT_TEMPLATES.get(family_key, ())
    if not templates:
        return {
            "position_key": role_key,
            "position_family_key": family_key,
            "position_label": _position_role_label(role_key),
            "recommended": [],
            "summary_text": "No formation templates available for this position.",
        }

    fits: list[dict[str, Any]] = []
    for template in templates:
        fit = _score_template_fit(metric_map=metric_map, targets=template.get("targets", {}))
        score = float(fit["score"])
        tier = "low"
        if score >= 0.75:
            tier = "high"
        elif score >= 0.55:
            tier = "medium"
        fits.append(
            {
                "formation": str(template.get("formation") or ""),
                "role": str(template.get("role") or ""),
                "fit_score_0_to_1": score,
                "fit_tier": tier,
                "coverage_0_to_1": float(fit["coverage"]),
                "matched_metrics": fit["parts"],
            }
        )

    fits.sort(key=lambda x: (x["fit_score_0_to_1"], x["coverage_0_to_1"]), reverse=True)
    recommended = fits[:3]
    if recommended:
        top = recommended[0]
        summary = (
            f"Best tactical fit for {_position_role_label(role_key)}: {top['formation']} as {top['role']} "
            f"({top['fit_tier']} fit)."
        )
    else:
        summary = "No formation fit could be estimated from available metrics."
    return {
        "position_key": role_key,
        "position_family_key": family_key,
        "position_label": _position_role_label(role_key),
        "recommended": recommended,
        "summary_text": summary,
    }


def _build_radar_profile(
    *,
    row: pd.Series,
    metric_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    role_key = _infer_position_role_key(row)
    family_key = _normalize_position_key(row.get("model_position") or row.get("position_group"))
    axes_metrics = RADAR_AXES_BY_POSITION.get(role_key) or RADAR_AXES_BY_POSITION.get(family_key)
    if axes_metrics is None:
        axes_metrics = tuple(metric for metric, _, _ in PROFILE_METRIC_SPECS[:6])

    axes: list[dict[str, Any]] = []
    for metric in axes_metrics:
        snap = metric_map.get(metric)
        if snap is None:
            axes.append(
                {
                    "metric": metric,
                    "label": ADVANCED_METRIC_LABEL.get(metric, metric),
                    "available": False,
                    "normalized_0_to_100": None,
                    "quality_percentile": None,
                    "player_value": None,
                    "cohort_median": None,
                }
            )
            continue
        quality = float(snap["quality_percentile"])
        axes.append(
            {
                "metric": metric,
                "label": snap["label"],
                "available": True,
                "normalized_0_to_100": float(np.clip(quality * 100.0, 0.0, 100.0)),
                "quality_percentile": quality,
                "player_value": snap["player_value"],
                "cohort_median": snap["cohort_median"],
            }
        )

    available = [a for a in axes if a["available"]]
    coverage = float(len(available) / max(len(axes), 1))
    return {
        "position_key": role_key,
        "position_family_key": family_key,
        "position_label": _position_role_label(role_key),
        "ready_for_plot": coverage >= 0.50,
        "coverage_0_to_1": coverage,
        "axes": axes,
    }


def _build_metric_profile(
    row: pd.Series,
    cohort: pd.DataFrame,
    top_metrics: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    metrics: list[dict[str, Any]] = []
    for col, label, direction in PROFILE_METRIC_SPECS:
        if col not in cohort.columns or col not in row.index:
            continue

        player_value = _safe_float(row.get(col))
        if player_value is None:
            continue

        series = pd.to_numeric(cohort[col], errors="coerce").dropna()
        if len(series) < 30:
            continue

        percentile_raw = float((series <= player_value).mean())
        quality_percentile = percentile_raw if direction > 0 else (1.0 - percentile_raw)

        q25, q50, q75 = np.nanquantile(series.to_numpy(), [0.25, 0.5, 0.75])
        iqr = max(float(q75 - q25), 1e-9)
        if direction > 0:
            improvement_gap = max(float(q75 - player_value), 0.0)
        else:
            improvement_gap = max(float(player_value - q25), 0.0)
        improvement_gap_iqr = improvement_gap / iqr
        impact_score = improvement_gap_iqr * _metric_value_corr_abs(cohort, col)

        status = "neutral"
        if quality_percentile >= 0.67:
            status = "strength"
        elif quality_percentile <= 0.33:
            status = "weakness"

        metrics.append(
            {
                "metric": col,
                "label": label,
                "direction": "higher_is_better" if direction > 0 else "lower_is_better",
                "player_value": player_value,
                "cohort_median": float(q50),
                "cohort_p25": float(q25),
                "cohort_p75": float(q75),
                "percentile_raw": percentile_raw,
                "quality_percentile": quality_percentile,
                "status": status,
                "improvement_gap": improvement_gap,
                "improvement_gap_iqr": improvement_gap_iqr,
                "impact_score": float(impact_score),
            }
        )

    if not metrics:
        return {"strengths": [], "weaknesses": [], "development_levers": []}

    strengths = sorted(
        [m for m in metrics if m["status"] == "strength"],
        key=lambda m: m["quality_percentile"],
        reverse=True,
    )[:top_metrics]

    weaknesses = sorted(
        [m for m in metrics if m["status"] == "weakness"],
        key=lambda m: m["quality_percentile"],
    )[:top_metrics]

    development = sorted(
        [m for m in weaknesses if m["improvement_gap"] > 0.0],
        key=lambda m: (m["impact_score"], m["improvement_gap_iqr"]),
        reverse=True,
    )[: min(3, top_metrics)]

    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "development_levers": development,
    }


def _build_confidence_summary(row: pd.Series, cohort: pd.DataFrame) -> dict[str, Any]:
    pred = _safe_float(row.get("fair_value_eur"))
    if pred is None:
        pred = _safe_float(row.get("expected_value_eur"))

    low = _safe_float(row.get("expected_value_low_eur"))
    high = _safe_float(row.get("expected_value_high_eur"))
    width = None
    width_ratio = None
    if low is not None and high is not None:
        width = max(high - low, 0.0)
        if pred is not None and pred > 0:
            width_ratio = width / pred

    confidence_signal = _safe_float(row.get("undervaluation_confidence"))
    prior_mae = _safe_float(row.get("prior_mae_eur"))
    prior_ratio = None
    if prior_mae is not None and pred is not None and pred > 0:
        prior_ratio = prior_mae / pred

    score = 0.5
    if confidence_signal is not None:
        score += min(max(confidence_signal, 0.0), 2.0) / 4.0
    if width_ratio is not None:
        score -= min(max(width_ratio, 0.0), 2.0) / 2.6
    if prior_ratio is not None:
        score -= min(max(prior_ratio, 0.0), 1.0) / 3.0
    score = float(np.clip(score, 0.0, 1.0))

    label = "low"
    if score >= 0.67:
        label = "high"
    elif score >= 0.40:
        label = "medium"

    cohort_median_prior_mae = None
    if "prior_mae_eur" in cohort.columns:
        cohort_median_prior_mae = _safe_float(
            pd.to_numeric(cohort["prior_mae_eur"], errors="coerce").median()
        )

    return {
        "label": label,
        "score": score,
        "undervaluation_confidence": confidence_signal,
        "interval_low_eur": low,
        "interval_high_eur": high,
        "interval_width_eur": width,
        "interval_width_ratio": width_ratio,
        "prior_mae_eur": prior_mae,
        "prior_mae_ratio_to_prediction": prior_ratio,
        "cohort_median_prior_mae_eur": cohort_median_prior_mae,
    }


def _build_valuation_guardrails(row: pd.Series) -> dict[str, Any]:
    pred = _safe_float(row.get("fair_value_eur"))
    if pred is None:
        pred = _safe_float(row.get("expected_value_eur"))
    market = _safe_float(row.get("market_value_eur"))

    raw_gap = _safe_float(row.get("value_gap_raw_eur"))
    if raw_gap is None:
        raw_gap = _safe_float(row.get("value_gap_eur"))
    cons_gap = _safe_float(row.get("value_gap_conservative_eur"))
    if raw_gap is None and pred is not None and market is not None:
        raw_gap = pred - market
    if cons_gap is None:
        cons_gap = raw_gap

    capped_gap = _safe_float(row.get("value_gap_capped_eur"))
    cap_threshold = _safe_float(row.get("value_gap_cap_threshold_eur"))
    cap_ratio = _safe_float(row.get("value_gap_cap_ratio"))
    cap_applied_raw = row.get("value_gap_cap_applied")
    cap_applied = bool(cap_applied_raw) if cap_applied_raw is not None and not pd.isna(cap_applied_raw) else False

    prior_mae = _safe_float(row.get("prior_mae_eur"))
    prior_p75ae = _safe_float(row.get("prior_p75ae_eur"))
    prior_qae = _safe_float(row.get("prior_qae_eur"))

    if cap_threshold is None:
        cap_candidates: list[float] = []
        if prior_qae is not None and prior_qae > 0:
            cap_candidates.append(2.5 * prior_qae)
        if prior_p75ae is not None and prior_p75ae > 0:
            cap_candidates.append(3.0 * prior_p75ae)
        if prior_mae is not None and prior_mae > 0:
            cap_candidates.append(4.0 * prior_mae)
        cap_threshold = min(cap_candidates) if cap_candidates else None

    if capped_gap is None:
        capped_gap = cons_gap
        if cons_gap is not None and cap_threshold is not None and cons_gap > 0:
            capped_gap = min(cons_gap, cap_threshold)
            cap_applied = bool(capped_gap < cons_gap - 1.0)

    if cap_ratio is None and cons_gap is not None and cap_threshold is not None and cap_threshold > 0:
        cap_ratio = cons_gap / cap_threshold

    return {
        "market_value_eur": market,
        "fair_value_eur": pred,
        "value_gap_raw_eur": raw_gap,
        "value_gap_conservative_eur": cons_gap,
        "value_gap_capped_eur": capped_gap,
        "cap_threshold_eur": cap_threshold,
        "cap_applied": cap_applied,
        "cap_ratio": cap_ratio,
        "prior_mae_eur": prior_mae,
        "prior_p75ae_eur": prior_p75ae,
        "prior_qae_eur": prior_qae,
    }


def _fmt_eur(value: float | None) -> str:
    if value is None:
        return "n/a"
    sign = "-" if value < 0 else ""
    v = abs(float(value))
    if v >= 1_000_000_000:
        return f"{sign}EUR {v / 1_000_000_000:.2f}bn"
    if v >= 1_000_000:
        return f"{sign}EUR {v / 1_000_000:.1f}m"
    if v >= 1_000:
        return f"{sign}EUR {v / 1_000:.0f}k"
    return f"{sign}EUR {v:.0f}"


def _build_summary_text(
    row: pd.Series,
    strengths: list[dict[str, Any]],
    development_levers: list[dict[str, Any]],
    risk_flags: list[dict[str, str]],
    confidence: dict[str, Any],
    valuation_guardrails: dict[str, Any],
) -> str:
    name = str(row.get("name") or row.get("player_id") or "Player")
    conf_label = str(confidence.get("label", "medium"))
    market = _fmt_eur(valuation_guardrails.get("market_value_eur"))
    fair = _fmt_eur(valuation_guardrails.get("fair_value_eur"))
    gap = _fmt_eur(valuation_guardrails.get("value_gap_conservative_eur"))
    capped = _fmt_eur(valuation_guardrails.get("value_gap_capped_eur"))

    top_strengths = ", ".join(m["label"] for m in strengths[:2]) if strengths else "no standout metric edge"
    top_levers = ", ".join(m["label"] for m in development_levers[:2]) if development_levers else "no clear lever"
    history_tier = str(row.get("history_strength_tier") or "").strip()
    history_score = _safe_float(row.get("history_strength_score"))
    history_note = None
    if history_tier or history_score is not None:
        if history_score is None:
            history_note = f"History profile: {history_tier}."
        elif history_tier:
            history_note = f"History profile: {history_tier} ({history_score:.0f}/100)."
        else:
            history_note = f"History profile score: {history_score:.0f}/100."

    sentences = [
        f"{name}: {conf_label}-confidence undervaluation signal (market {market}, fair value {fair}).",
        f"Conservative value gap is {gap}; guardrailed gap is {capped}.",
        f"Top strengths: {top_strengths}. Development focus: {top_levers}.",
    ]
    if history_note:
        sentences.append(history_note)
    if risk_flags:
        top_risks = ", ".join(flag["code"] for flag in risk_flags[:2])
        sentences.append(f"Key risk flags: {top_risks}.")
    else:
        sentences.append("No major risk flag triggered by current thresholds.")
    return " ".join(sentences)


def _build_risk_flags(
    row: pd.Series,
    cohort: pd.DataFrame,
    confidence: dict[str, Any],
    valuation_guardrails: dict[str, Any],
) -> list[dict[str, str]]:
    flags: list[dict[str, str]] = []

    minutes = _safe_float(row.get("minutes"))
    if minutes is None:
        minutes = _safe_float(row.get("sofa_minutesPlayed"))
    if minutes is not None and minutes < 900:
        flags.append(
            {
                "severity": "medium",
                "code": "low_minutes",
                "message": "Low minute sample this season; performance signal is less stable.",
            }
        )
    elif minutes is not None and minutes < 1200:
        flags.append(
            {
                "severity": "low",
                "code": "low_minutes_watch",
                "message": "Minutes are below a full-season sample; monitor stability.",
            }
        )

    width_ratio = confidence.get("interval_width_ratio")
    if width_ratio is not None and width_ratio >= 1.0:
        flags.append(
            {
                "severity": "high",
                "code": "high_uncertainty",
                "message": "Prediction interval is wide relative to predicted value.",
            }
        )
    elif width_ratio is not None and width_ratio >= 0.70:
        flags.append(
            {
                "severity": "medium",
                "code": "medium_uncertainty",
                "message": "Prediction interval is moderately wide; prefer conservative valuation.",
            }
        )

    injury_burden = _safe_float(row.get("injury_days_per_1000_min"))
    if injury_burden is not None and "injury_days_per_1000_min" in cohort.columns:
        cohort_injury = pd.to_numeric(cohort["injury_days_per_1000_min"], errors="coerce").dropna()
        if len(cohort_injury) >= 20:
            p65 = float(np.nanquantile(cohort_injury.to_numpy(), 0.65))
            p85 = float(np.nanquantile(cohort_injury.to_numpy(), 0.85))
            if injury_burden >= max(p85, 60.0):
                flags.append(
                    {
                        "severity": "high",
                        "code": "injury_burden_high",
                        "message": "Injury burden is high versus cohort and may reduce reliability.",
                    }
                )
            elif injury_burden >= max(p65, 35.0):
                flags.append(
                    {
                        "severity": "medium",
                        "code": "injury_burden",
                        "message": "Injury burden is above cohort baseline.",
                    }
                )

    contract_years = _safe_float(row.get("contract_years_left"))
    if contract_years is not None and contract_years <= 0.5:
        flags.append(
            {
                "severity": "medium",
                "code": "contract_very_short",
                "message": "Very short contract horizon can strongly distort transfer valuation.",
            }
        )
    elif contract_years is not None and contract_years <= 1.0:
        flags.append(
            {
                "severity": "low",
                "code": "contract_horizon",
                "message": "Short contract horizon can distort valuation and transfer dynamics.",
            }
        )

    history_cov = _safe_float(row.get("history_strength_coverage"))
    history_score = _safe_float(row.get("history_strength_score"))
    if history_cov is not None and history_cov < 0.35:
        flags.append(
            {
                "severity": "medium",
                "code": "history_data_sparse",
                "message": "Historical signal coverage is sparse; treat development trend cautiously.",
            }
        )
    elif history_score is not None and history_score < 40.0:
        flags.append(
            {
                "severity": "low",
                "code": "history_strength_low",
                "message": "Historical stability/momentum profile is below cohort baseline.",
            }
        )

    if valuation_guardrails.get("cap_applied"):
        cap_ratio = valuation_guardrails.get("cap_ratio")
        severity = "medium"
        if isinstance(cap_ratio, (float, int)) and cap_ratio >= 1.75:
            severity = "high"
        flags.append(
            {
                "severity": severity,
                "code": "valuation_optimism_guardrail",
                "message": "Raw undervaluation is above historical error priors; capped for conservative decisions.",
            }
        )

    return flags


def get_player_report(
    player_id: str,
    split: Split = "test",
    season: str | None = None,
    top_metrics: int = 5,
) -> dict[str, Any]:
    frame = _prepare_predictions_frame(get_predictions(split=split))
    row = _select_player_row(frame=frame, player_id=player_id, season=season)
    return _build_player_report_from_row(frame=frame, row=row, top_metrics=top_metrics)


def _build_player_report_from_row(
    *,
    frame: pd.DataFrame,
    row: pd.Series,
    top_metrics: int = 5,
) -> dict[str, Any]:
    row_dict = _to_records(row.to_frame().T)[0]

    cohort, cohort_filters = _cohort_for_player(frame=frame, row=row)
    metric_map = _build_metric_snapshot_map(row=row, cohort=cohort)
    metric_profile = _build_metric_profile(row=row, cohort=cohort, top_metrics=max(int(top_metrics), 1))
    player_type = _build_player_type_profile(row=row, metric_map=metric_map)
    formation_fit = _build_formation_fit_profile(row=row, metric_map=metric_map)
    radar_profile = _build_radar_profile(row=row, metric_map=metric_map)
    confidence = _build_confidence_summary(row=row, cohort=cohort)
    valuation_guardrails = _build_valuation_guardrails(row=row)
    risk_flags = _build_risk_flags(
        row=row,
        cohort=cohort,
        confidence=confidence,
        valuation_guardrails=valuation_guardrails,
    )
    summary_text = _build_summary_text(
        row=row,
        strengths=metric_profile["strengths"],
        development_levers=metric_profile["development_levers"],
        risk_flags=risk_flags,
        confidence=confidence,
        valuation_guardrails=valuation_guardrails,
    )

    return {
        "player": row_dict,
        "cohort": {
            "size": int(len(cohort)),
            "filters": cohort_filters,
        },
        "strengths": metric_profile["strengths"],
        "weaknesses": metric_profile["weaknesses"],
        "development_levers": metric_profile["development_levers"],
        "player_type": player_type,
        "formation_fit": formation_fit,
        "radar_profile": radar_profile,
        "risk_flags": risk_flags,
        "confidence": confidence,
        "valuation_guardrails": valuation_guardrails,
        "summary_text": summary_text,
    }


def query_player_reports(
    split: Split = "test",
    season: str | None = None,
    league: str | None = None,
    club: str | None = None,
    position: str | None = None,
    min_minutes: float | None = None,
    max_age: float | None = None,
    player_ids: Sequence[str] | None = None,
    top_metrics: int = 5,
    include_history: bool = True,
    sort_by: str = "undervaluation_score",
    sort_order: Literal["asc", "desc"] = "desc",
    limit: int = 200,
    offset: int = 0,
) -> dict[str, Any]:
    frame = _prepare_predictions_frame(get_predictions(split=split))
    work = frame.copy()

    if season and "season" in work.columns:
        work = work[work["season"].astype(str) == str(season)].copy()
    if league and "league" in work.columns:
        work = work[work["league"].astype(str).str.casefold() == str(league).casefold()].copy()
    if club and "club" in work.columns:
        work = work[work["club"].astype(str).str.casefold() == str(club).casefold()].copy()
    if position:
        pos_series = _position_series(work)
        work = work[pos_series == str(position).upper()].copy()

    if min_minutes is not None:
        work = work[_minutes_series(work).fillna(0.0) >= float(min_minutes)].copy()
    if max_age is not None and "age" in work.columns:
        ages = pd.to_numeric(work["age"], errors="coerce")
        work = work[ages <= float(max_age)].copy()

    if player_ids:
        if "player_id" not in work.columns:
            raise ValueError("Prediction artifact does not include 'player_id'.")
        wanted_ids = {str(pid).strip() for pid in player_ids if str(pid).strip()}
        if wanted_ids:
            work = work[work["player_id"].astype(str).isin(wanted_ids)].copy()

    total = int(len(work))

    if sort_by not in work.columns:
        fallback = (
            "undervaluation_score"
            if "undervaluation_score" in work.columns
            else (
                "value_gap_capped_eur"
                if "value_gap_capped_eur" in work.columns
                else (
                    "value_gap_conservative_eur"
                    if "value_gap_conservative_eur" in work.columns
                    else ("value_gap_eur" if "value_gap_eur" in work.columns else work.columns[0])
                )
            )
        )
        sort_by = fallback

    ascending = sort_order == "asc"
    work = work.sort_values(sort_by, ascending=ascending, na_position="last")

    start = max(int(offset), 0)
    end = start + max(int(limit), 0)
    page = work.iloc[start:end].copy()

    items: list[dict[str, Any]] = []
    for _, row in page.iterrows():
        report = _build_player_report_from_row(frame=frame, row=row, top_metrics=top_metrics)
        item: dict[str, Any] = {
            "player_id": str(row.get("player_id") or ""),
            "season": row.get("season"),
            "report": report,
        }
        if include_history:
            item["history_strength"] = _build_history_strength_payload(row=row)
        items.append(item)

    return {
        "split": split,
        "total": total,
        "count": int(len(items)),
        "limit": int(limit),
        "offset": int(offset),
        "sort_by": sort_by,
        "sort_order": sort_order,
        "items": items,
    }


def get_player_advanced_profile(
    player_id: str,
    split: Split = "test",
    season: str | None = None,
    top_metrics: int = 6,
) -> dict[str, Any]:
    report = get_player_report(
        player_id=player_id,
        split=split,
        season=season,
        top_metrics=top_metrics,
    )
    return {
        "player": report.get("player", {}),
        "cohort": report.get("cohort", {}),
        "player_type": report.get("player_type", {}),
        "formation_fit": report.get("formation_fit", {}),
        "radar_profile": report.get("radar_profile", {}),
        "strengths": report.get("strengths", []),
        "weaknesses": report.get("weaknesses", []),
        "development_levers": report.get("development_levers", []),
        "risk_flags": report.get("risk_flags", []),
        "confidence": report.get("confidence", {}),
        "valuation_guardrails": report.get("valuation_guardrails", {}),
        "summary_text": report.get("summary_text"),
    }


def _build_history_strength_payload(row: pd.Series) -> dict[str, Any]:
    score = _safe_float(row.get("history_strength_score"))
    coverage = _safe_float(row.get("history_strength_coverage"))
    tier = row.get("history_strength_tier")
    tier_text = str(tier).strip() if tier is not None and not pd.isna(tier) else "uncertain"

    components: list[dict[str, Any]] = []
    for key in HISTORY_COMPONENT_COLUMNS:
        value = _safe_float(row.get(key))
        weight = float(HISTORY_COMPONENT_WEIGHTS.get(key, 0.0))
        label = HISTORY_COMPONENT_LABELS.get(key, key)
        weighted_points = None if value is None else float(value * weight * 100.0)
        components.append(
            {
                "key": key,
                "label": label,
                "value_0_to_1": value,
                "value_0_to_100": None if value is None else float(value * 100.0),
                "weight": weight,
                "weighted_points_0_to_100": weighted_points,
                "missing": value is None,
            }
        )

    components_sorted = sorted(
        components,
        key=lambda x: x["weighted_points_0_to_100"] if isinstance(x["weighted_points_0_to_100"], (float, int)) else -1.0,
        reverse=True,
    )
    strongest = [c for c in components_sorted if not c["missing"]][:3]

    weakest = sorted(
        [c for c in components if not c["missing"]],
        key=lambda x: x["value_0_to_1"] if isinstance(x["value_0_to_1"], (float, int)) else 1.0,
    )[:3]

    if score is None:
        narrative = "History strength score is unavailable because required history components are missing."
    else:
        narrative = f"History strength is {score:.1f}/100 ({tier_text})."
        if strongest:
            narrative += " Strongest components: " + ", ".join(c["label"] for c in strongest[:2]) + "."
        if weakest:
            narrative += " Development focus: " + ", ".join(c["label"] for c in weakest[:2]) + "."

    return {
        "score_0_to_100": score,
        "coverage_0_to_1": coverage,
        "tier": tier_text,
        "components": components,
        "strongest_components": strongest,
        "improvement_components": weakest,
        "summary_text": narrative,
    }


def _humanize_profile_field(key: str) -> str:
    raw = (
        str(key)
        .replace("sofa_", "")
        .replace("clubctx_", "club ")
        .replace("history_", "history ")
        .replace("prior_", "prior ")
        .replace("_", " ")
        .strip()
    )
    raw = " ".join(raw.split())
    if not raw:
        return str(key)
    words = raw.split(" ")
    return " ".join(word.upper() if word.lower() in {"eur", "xg", "xa"} else word.capitalize() for word in words)


def _is_profile_stat_value(key: str, value: Any) -> bool:
    key_str = str(key)
    if key_str in PROFILE_STAT_SKIP_FIELDS or key_str.startswith("_"):
        return False
    if isinstance(value, bool):
        return True
    if isinstance(value, str):
        return key_str in PROFILE_CONTEXT_FIELDS and bool(value.strip())
    return _safe_float(value) is not None


def _profile_stat_kind(key: str, value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, str):
        return "text"
    key_str = str(key).lower()
    if key_str.endswith("_eur"):
        return "currency"
    if key_str.endswith("_0_to_1") or "percentile" in key_str:
        return "fraction"
    if "percentage" in key_str:
        return "percentage"
    if "minutes" in key_str or key_str.endswith("_n") or "count" in key_str or "caps" in key_str:
        return "integer"
    return "number"


def _format_profile_stat_value(key: str, value: Any) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, str):
        return value
    num = _safe_float(value)
    if num is None:
        return "n/a"
    kind = _profile_stat_kind(key, value)
    if kind == "currency":
        return _fmt_eur(num)
    if kind == "fraction":
        return f"{num * 100.0:.1f}%"
    if kind == "percentage":
        return f"{num:.1f}%"
    if kind == "integer":
        return f"{int(round(num)):,}"
    return f"{num:.2f}" if abs(num) < 1000 else f"{num:,.0f}"


def _classify_profile_stat_group(key: str) -> str:
    key_str = str(key)
    key_lower = key_str.lower()
    if key_str in PROFILE_CONTEXT_FIELDS:
        return "Profile & Context"
    if key_lower.startswith("sb_"):
        return "External Tactical"
    if key_lower.startswith("fixture_") or key_lower.startswith("odds_"):
        return "Schedule & Market"
    if key_lower.startswith("avail_"):
        return "Availability & Physical"
    if any(tok in key_lower for tok in ("market_value", "expected_value", "fair_value", "gap", "confidence", "interval", "calibration", "prior_")):
        return "Value & Model"
    if any(tok in key_lower for tok in ("goal", "assist", "shot", "xg", "xa", "dribble", "bigchance", "penalty")):
        return "Attacking"
    if any(tok in key_lower for tok in ("pass", "cross", "throughball", "longball", "keypass", "progressive", "chancecreated")):
        return "Passing & Progression"
    if any(tok in key_lower for tok in ("tackle", "interception", "clearance", "blocked", "duel", "aerial", "recovery", "possessionwon")):
        return "Defending & Duels"
    if any(tok in key_lower for tok in ("save", "highclaim", "runout", "goalsprevented", "cleansheet")):
        return "Goalkeeping"
    if any(tok in key_lower for tok in ("age", "minutes", "injury", "height", "weight", "contract", "foot")):
        return "Availability & Physical"
    if key_lower.startswith("clubctx_") or key_lower.startswith("history_") or "coeff" in key_lower:
        return "History & Context"
    return "Other Metrics"


def _build_profile_stat_groups(row_payload: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in PROFILE_STAT_GROUP_ORDER}
    for key, value in row_payload.items():
        if not _is_profile_stat_value(key, value):
            continue
        group_name = _classify_profile_stat_group(key)
        grouped[group_name].append(
            {
                "key": str(key),
                "label": _humanize_profile_field(str(key)),
                "value": value,
                "display_value": _format_profile_stat_value(str(key), value),
                "kind": _profile_stat_kind(str(key), value),
            }
        )

    out: list[dict[str, Any]] = []
    for group_name in PROFILE_STAT_GROUP_ORDER:
        items = sorted(grouped[group_name], key=lambda item: str(item["label"]))
        if not items:
            continue
        out.append(
            {
                "group": group_name,
                "count": int(len(items)),
                "items": items,
            }
        )
    return out


def _load_similar_player_matches(player_id: str, top_k: int) -> list[dict[str, Any]]:
    from scouting_ml.services import find_similar_players

    return list(find_similar_players(player_id, top_k=top_k))


def _build_similar_players_payload(
    *,
    frame: pd.DataFrame,
    row: pd.Series,
    top_k: int = 5,
) -> dict[str, Any]:
    player_id = str(row.get("player_id") or "").strip()
    if not player_id:
        return {"available": False, "reason": "missing_player_id", "items": []}

    try:
        raw_matches = _load_similar_player_matches(player_id, top_k=max(int(top_k), 1))
    except Exception as exc:  # pragma: no cover - defensive
        return {"available": False, "reason": str(exc), "items": []}

    enriched: list[dict[str, Any]] = []
    lookup = frame.copy()
    if "player_id" in lookup.columns:
        lookup = lookup.set_index("player_id", drop=False)
    for match in raw_matches:
        candidate_id = str(match.get("player_id") or "").strip()
        item = {
            "player_id": candidate_id,
            "score": _safe_float(match.get("score")),
            "justification": str(match.get("justification") or ""),
        }
        if candidate_id and candidate_id in lookup.index:
            candidate = lookup.loc[candidate_id]
            if isinstance(candidate, pd.DataFrame):
                candidate = candidate.iloc[0]
            item.update(
                {
                    "name": candidate.get("name"),
                    "club": candidate.get("club"),
                    "league": candidate.get("league"),
                    "season": candidate.get("season"),
                    "position": candidate.get("model_position") or candidate.get("position_group"),
                    "market_value_eur": _safe_float(candidate.get("market_value_eur")),
                    "expected_value_eur": _safe_float(candidate.get("expected_value_eur")),
                }
            )
        enriched.append(item)

    return {"available": True, "reason": None, "items": enriched}


def _build_external_tactical_context(row: pd.Series) -> dict[str, Any]:
    formations: list[dict[str, Any]] = []
    for key in row.index:
        key_str = str(key)
        if not key_str.startswith("sb_minutes_in_"):
            continue
        minutes = _safe_float(row.get(key_str))
        if minutes is None or minutes <= 0:
            continue
        formations.append(
            {
                "formation": key_str.replace("sb_minutes_in_", ""),
                "minutes": minutes,
            }
        )
    formations.sort(key=lambda item: float(item["minutes"]), reverse=True)

    metrics = [
        ("Progressive passes/90", _safe_float(row.get("sb_progressive_passes_per90"))),
        ("Progressive carries/90", _safe_float(row.get("sb_progressive_carries_per90"))),
        ("Passes into box/90", _safe_float(row.get("sb_passes_into_box_per90"))),
        ("Shot assists/90", _safe_float(row.get("sb_shot_assists_per90"))),
        ("Pressures/90", _safe_float(row.get("sb_pressures_per90"))),
        ("Counterpressures/90", _safe_float(row.get("sb_counterpressures_per90"))),
        ("High regains/90", _safe_float(row.get("sb_high_regains_per90"))),
        ("Duel win rate", _safe_float(row.get("sb_duel_win_rate"))),
        ("Aerial win rate", _safe_float(row.get("sb_aerial_win_rate"))),
    ]
    signals = [
        {"label": label, "value": value, "display_value": f"{value * 100.0:.1f}%" if "rate" in label.lower() else f"{value:.2f}"}
        for label, value in metrics
        if value is not None
    ]
    if not formations and not signals:
        return {"available": False, "summary_text": "No external tactical provider signals available.", "preferred_formations": [], "signals": []}

    if formations:
        top = formations[0]
        summary = f"StatsBomb profile leans toward {top['formation']} ({top['minutes']:.0f} tracked minutes)."
    else:
        summary = "StatsBomb tactical metrics available without formation exposure."
    return {
        "available": True,
        "summary_text": summary,
        "preferred_formations": formations[:4],
        "signals": signals[:6],
    }


def _build_availability_context(row: pd.Series) -> dict[str, Any]:
    reports = _safe_float(row.get("avail_reports"))
    start_share = _safe_float(row.get("avail_start_share"))
    bench_share = _safe_float(row.get("avail_bench_share"))
    injury_count = _safe_float(row.get("avail_injury_count"))
    suspension_count = _safe_float(row.get("avail_suspension_count"))
    expected_start_rate = _safe_float(row.get("avail_expected_start_rate"))

    signals = []
    for label, value, fmt in [
        ("Availability reports", reports, "count"),
        ("Start share", start_share, "pct"),
        ("Bench share", bench_share, "pct"),
        ("Injury reports", injury_count, "count"),
        ("Suspension reports", suspension_count, "count"),
        ("Expected start rate", expected_start_rate, "pct"),
    ]:
        if value is None:
            continue
        display = f"{value * 100.0:.1f}%" if fmt == "pct" else f"{value:.0f}"
        signals.append({"label": label, "value": value, "display_value": display})

    if not signals:
        return {"available": False, "summary_text": "No provider availability signals available.", "signals": []}

    summary_parts: list[str] = []
    if start_share is not None:
        summary_parts.append(f"start share {start_share * 100.0:.0f}%")
    if injury_count is not None:
        summary_parts.append(f"{injury_count:.0f} injury reports")
    if expected_start_rate is not None:
        summary_parts.append(f"expected-start rate {expected_start_rate * 100.0:.0f}%")
    summary = "Availability context: " + ", ".join(summary_parts) + "." if summary_parts else "Availability provider signals available."
    return {
        "available": True,
        "summary_text": summary,
        "signals": signals,
    }


def _build_market_context_payload(row: pd.Series) -> dict[str, Any]:
    signals = []
    for label, key, fmt in [
        ("Fixture matches", "fixture_matches", "count"),
        ("Mean rest days", "fixture_mean_rest_days", "num"),
        ("Congestion share", "fixture_congestion_share", "pct"),
        ("Opponent strength", "fixture_opponent_strength", "num"),
        ("Team market strength", "odds_implied_team_strength", "pct"),
        ("Opponent market strength", "odds_implied_opponent_strength", "pct"),
        ("Upset probability", "odds_upset_probability", "pct"),
        ("Expected total goals", "odds_expected_total_goals", "num"),
    ]:
        value = _safe_float(row.get(key))
        if value is None:
            continue
        if fmt == "pct":
            display = f"{value * 100.0:.1f}%"
        elif fmt == "count":
            display = f"{value:.0f}"
        else:
            display = f"{value:.2f}"
        signals.append({"label": label, "value": value, "display_value": display})

    if not signals:
        return {"available": False, "summary_text": "No fixture or market-context signals available.", "signals": []}

    congestion = _safe_float(row.get("fixture_congestion_share"))
    team_strength = _safe_float(row.get("odds_implied_team_strength"))
    summary_bits: list[str] = []
    if congestion is not None:
        summary_bits.append(f"schedule congestion {congestion * 100.0:.0f}%")
    if team_strength is not None:
        summary_bits.append(f"market win strength {team_strength * 100.0:.0f}%")
    summary = "Schedule + market context: " + ", ".join(summary_bits) + "." if summary_bits else "Fixture and market signals available."
    return {
        "available": True,
        "summary_text": summary,
        "signals": signals,
    }


def _build_provider_coverage(row: pd.Series) -> dict[str, Any]:
    tactical = _build_external_tactical_context(row)
    availability = _build_availability_context(row)
    market = _build_market_context_payload(row)
    return {
        "statsbomb": bool(tactical.get("available")),
        "availability_provider": bool(availability.get("available")),
        "market_provider": bool(market.get("available")),
    }


def _build_provider_risk_flags(row: pd.Series) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    injury_count = _safe_float(row.get("avail_injury_count"))
    if injury_count is not None and injury_count >= 3:
        flags.append(
            {
                "severity": "medium",
                "code": "provider_injury_load",
                "message": f"External availability feed shows {injury_count:.0f} injury reports this season.",
            }
        )
    start_share = _safe_float(row.get("avail_start_share"))
    reports = _safe_float(row.get("avail_reports"))
    if reports is not None and reports >= 3 and start_share is not None and start_share < 0.5:
        flags.append(
            {
                "severity": "medium",
                "code": "provider_rotation_risk",
                "message": f"External lineup feed shows only {start_share * 100.0:.0f}% starts across tracked reports.",
            }
        )
    congestion = _safe_float(row.get("fixture_congestion_share"))
    if congestion is not None and congestion >= 0.4:
        flags.append(
            {
                "severity": "low",
                "code": "provider_schedule_congestion",
                "message": f"Fixture context indicates high congestion ({congestion * 100.0:.0f}% of matches on short rest).",
            }
        )
    return flags


def get_player_history_strength(
    player_id: str,
    split: Split = "test",
    season: str | None = None,
) -> dict[str, Any]:
    frame = _prepare_predictions_frame(get_predictions(split=split))
    row = _select_player_row(frame=frame, player_id=player_id, season=season)
    row_dict = _to_records(row.to_frame().T)[0]
    return {
        "player": row_dict,
        "history_strength": _build_history_strength_payload(row=row),
    }


def get_player_profile(
    player_id: str,
    split: Split = "test",
    season: str | None = None,
    top_metrics: int = 6,
    similar_top_k: int = 5,
) -> dict[str, Any]:
    frame = _prepare_predictions_frame(get_predictions(split=split))
    row = _select_player_row(frame=frame, player_id=player_id, season=season)
    report = _build_player_report_from_row(frame=frame, row=row, top_metrics=top_metrics)
    history_strength = _build_history_strength_payload(row=row)
    player_payload = dict(report.get("player", {}))
    tactical_context = _build_external_tactical_context(row)
    availability_context = _build_availability_context(row)
    market_context = _build_market_context_payload(row)
    combined_risk_flags = list(report.get("risk_flags", [])) + _build_provider_risk_flags(row)
    return {
        "player": player_payload,
        "cohort": report.get("cohort", {}),
        "strengths": report.get("strengths", []),
        "weaknesses": report.get("weaknesses", []),
        "development_levers": report.get("development_levers", []),
        "player_type": report.get("player_type", {}),
        "formation_fit": report.get("formation_fit", {}),
        "radar_profile": report.get("radar_profile", {}),
        "risk_flags": combined_risk_flags,
        "confidence": report.get("confidence", {}),
        "valuation_guardrails": report.get("valuation_guardrails", {}),
        "history_strength": history_strength,
        "summary_text": report.get("summary_text"),
        "external_tactical_context": tactical_context,
        "availability_context": availability_context,
        "market_context": market_context,
        "provider_coverage": _build_provider_coverage(row),
        "stat_groups": _build_profile_stat_groups(player_payload),
        "similar_players": _build_similar_players_payload(
            frame=frame,
            row=row,
            top_k=similar_top_k,
        ),
    }


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_watchlist_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _write_watchlist_records(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def list_watchlist(
    *,
    split: Split | None = None,
    tag: str | None = None,
    player_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    path = _watchlist_path()
    records = _read_watchlist_records(path)

    out: list[dict[str, Any]] = []
    for item in records:
        if split and str(item.get("split", "")).lower() != str(split):
            continue
        if tag and str(item.get("tag", "")).strip().casefold() != str(tag).strip().casefold():
            continue
        if player_id and str(item.get("player_id", "")) != str(player_id):
            continue
        out.append(item)

    out.sort(key=lambda x: str(x.get("created_at_utc", "")), reverse=True)
    total = len(out)
    start = max(int(offset), 0)
    end = start + max(int(limit), 0)
    page = out[start:end]

    return {
        "path": str(path),
        "total": int(total),
        "count": int(len(page)),
        "limit": int(limit),
        "offset": int(offset),
        "items": page,
    }


def add_watchlist_item(
    *,
    player_id: str,
    split: Split = "test",
    season: str | None = None,
    tag: str | None = None,
    notes: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    row = get_player_prediction(player_id=player_id, split=split, season=season)
    report = get_player_report(player_id=player_id, split=split, season=season, top_metrics=5)
    guardrails = report.get("valuation_guardrails", {})
    confidence = report.get("confidence", {})
    risk_flags = report.get("risk_flags", [])

    record = {
        "watch_id": uuid.uuid4().hex,
        "created_at_utc": _now_utc_iso(),
        "split": split,
        "season": str(row.get("season") or season or ""),
        "player_id": str(row.get("player_id") or player_id),
        "name": row.get("name"),
        "league": row.get("league"),
        "club": row.get("club"),
        "position": row.get("model_position") or row.get("position_group"),
        "tag": tag or "",
        "notes": notes or "",
        "source": source or "manual",
        "market_value_eur": _safe_float(row.get("market_value_eur")),
        "fair_value_eur": _safe_float(row.get("fair_value_eur") or row.get("expected_value_eur")),
        "value_gap_capped_eur": _safe_float(guardrails.get("value_gap_capped_eur")),
        "value_gap_conservative_eur": _safe_float(guardrails.get("value_gap_conservative_eur")),
        "undervaluation_confidence": _safe_float(row.get("undervaluation_confidence")),
        "confidence_label": confidence.get("label"),
        "risk_codes": [str(flag.get("code")) for flag in risk_flags if isinstance(flag, dict) and flag.get("code")],
        "summary_text": report.get("summary_text"),
    }

    path = _watchlist_path()
    records = _read_watchlist_records(path)
    records.append(record)
    _write_watchlist_records(path, records)
    return record


def delete_watchlist_item(watch_id: str) -> dict[str, Any]:
    path = _watchlist_path()
    records = _read_watchlist_records(path)
    keep = [row for row in records if str(row.get("watch_id")) != str(watch_id)]
    deleted = len(keep) != len(records)
    if deleted:
        _write_watchlist_records(path, keep)
    return {
        "path": str(path),
        "watch_id": str(watch_id),
        "deleted": bool(deleted),
    }


def get_model_manifest() -> dict[str, Any]:
    test_path = _resolve_path(*SPLIT_TO_PATH["test"])
    val_path = _resolve_path(*SPLIT_TO_PATH["val"])
    metrics_path = _resolve_path(METRICS_ENV, DEFAULT_METRICS)
    manifest_env_raw = os.getenv(MODEL_MANIFEST_ENV, "").strip()
    manifest_path = _manifest_path()

    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["source"] = str(payload.get("source") or "file")
        payload["_meta"] = {
            "source": payload["source"],
            "path": str(manifest_path),
            "sha256": _sha256_file(manifest_path),
            "mtime_utc": _file_meta(manifest_path).get("mtime_utc"),
        }
        if (
            manifest_env_raw
            or _manifest_role_section(payload, "valuation") is not None
            or _manifest_role_section(payload, "future_shortlist") is not None
            or _manifest_targets_active_artifacts(
            payload,
            test_path=test_path,
            val_path=val_path,
            metrics_path=metrics_path,
        )):
            return payload

    valuation_paths = _resolve_role_artifact_paths("valuation")
    future_paths = _resolve_role_artifact_paths("future_shortlist")
    out: dict[str, Any] = {
        "registry_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "derived",
        "legacy_default_role": "future_shortlist",
        "artifacts": {
            "test_predictions": {
                "path": str(test_path),
                **_file_meta(test_path),
                "sha256": _sha256_file(test_path),
            },
            "val_predictions": {
                "path": str(val_path),
                **_file_meta(val_path),
                "sha256": _sha256_file(val_path),
            },
            "metrics": {
                "path": str(metrics_path),
                **_file_meta(metrics_path),
                "sha256": _sha256_file(metrics_path),
            },
        },
        "config": {},
        "summary": {},
    }

    try:
        metrics = get_metrics(role="future_shortlist")
        out["config"] = {
            "dataset": metrics.get("dataset"),
            "val_season": metrics.get("val_season"),
            "test_season": metrics.get("test_season"),
            "trials_per_position": metrics.get("trials_per_position"),
            "recency_half_life": metrics.get("recency_half_life"),
            "optimize_metric": metrics.get("optimize_metric"),
            "interval_q": metrics.get("interval_q"),
            "two_stage_band_model": metrics.get("two_stage_band_model"),
            "band_min_samples": metrics.get("band_min_samples"),
            "band_blend_alpha": metrics.get("band_blend_alpha"),
            "strict_leakage_guard": metrics.get("strict_leakage_guard"),
        }
        summary = {
            "overall": metrics.get("overall"),
            "segments": metrics.get("segments"),
            "holdout": metrics.get("holdout"),
            "artifacts": metrics.get("artifacts"),
        }
        out["summary"] = summary
    except Exception as exc:
        out["metrics_error"] = str(exc)

    for role, role_paths in (("valuation", valuation_paths), ("future_shortlist", future_paths)):
        try:
            role_metrics = get_metrics(role=role)
            role_config = {
                "dataset": role_metrics.get("dataset"),
                "val_season": role_metrics.get("val_season"),
                "test_season": role_metrics.get("test_season"),
                "trials_per_position": role_metrics.get("trials_per_position"),
                "recency_half_life": role_metrics.get("recency_half_life"),
                "optimize_metric": role_metrics.get("optimize_metric"),
                "interval_q": role_metrics.get("interval_q"),
                "two_stage_band_model": role_metrics.get("two_stage_band_model"),
                "band_min_samples": role_metrics.get("band_min_samples"),
                "band_blend_alpha": role_metrics.get("band_blend_alpha"),
                "strict_leakage_guard": role_metrics.get("strict_leakage_guard"),
            }
            role_summary = {
                "overall": role_metrics.get("overall"),
                "segments": role_metrics.get("segments"),
                "holdout": role_metrics.get("holdout"),
                "artifacts": role_metrics.get("artifacts"),
            }
        except Exception as exc:
            role_config = {}
            role_summary = {"metrics_error": str(exc)}

        out[ROLE_TO_MANIFEST_KEY[role]] = {
            "role": role,
            "label": role,
            "artifacts": {
                "test_predictions": {
                    "path": str(role_paths["test_predictions"]),
                    **_file_meta(role_paths["test_predictions"]),
                    "sha256": _sha256_file(role_paths["test_predictions"]),
                },
                "val_predictions": {
                    "path": str(role_paths["val_predictions"]),
                    **_file_meta(role_paths["val_predictions"]),
                    "sha256": _sha256_file(role_paths["val_predictions"]),
                },
                "metrics": {
                    "path": str(role_paths["metrics"]),
                    **_file_meta(role_paths["metrics"]),
                    "sha256": _sha256_file(role_paths["metrics"]),
                },
            },
            "config": role_config,
            "summary": role_summary,
        }

    return out


def health_payload() -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": "ok",
        "artifacts": {},
        "strict_artifacts": _env_flag("SCOUTING_STRICT_ARTIFACTS", default=False),
        "strict_artifacts_error": None,
    }
    test_path = _resolve_path(*SPLIT_TO_PATH["test"])
    val_path = _resolve_path(*SPLIT_TO_PATH["val"])
    metrics_path = _resolve_path(METRICS_ENV, DEFAULT_METRICS)
    test_meta = _file_meta(test_path)
    val_meta = _file_meta(val_path)
    metrics_meta = _file_meta(metrics_path)
    out["artifacts"] = {
        "test_predictions_path": str(test_path),
        "val_predictions_path": str(val_path),
        "metrics_path": str(metrics_path),
        "test_predictions_exists": test_meta["exists"],
        "val_predictions_exists": val_meta["exists"],
        "metrics_exists": metrics_meta["exists"],
        "test_predictions_size_bytes": test_meta["size_bytes"],
        "val_predictions_size_bytes": val_meta["size_bytes"],
        "metrics_size_bytes": metrics_meta["size_bytes"],
        "test_predictions_mtime_utc": test_meta["mtime_utc"],
        "val_predictions_mtime_utc": val_meta["mtime_utc"],
        "metrics_mtime_utc": metrics_meta["mtime_utc"],
    }
    if out["strict_artifacts"]:
        try:
            validate_strict_artifact_env()
        except Exception as exc:
            out["strict_artifacts_error"] = str(exc)
    try:
        out["test_rows"] = int(len(get_predictions("test")))
    except Exception as exc:
        out["test_rows"] = None
        out["test_error"] = str(exc)
    try:
        out["val_rows"] = int(len(get_predictions("val")))
    except Exception as exc:
        out["val_rows"] = None
        out["val_error"] = str(exc)
    try:
        metrics = get_metrics()
        out["metrics_loaded"] = True
        out["metrics_dataset"] = metrics.get("dataset")
        out["metrics_test_season"] = metrics.get("test_season")
        out["metrics_val_season"] = metrics.get("val_season")
    except Exception as exc:
        out["metrics_loaded"] = False
        out["metrics_error"] = str(exc)
    if (
        out["strict_artifacts_error"]
        or not test_meta["exists"]
        or not val_meta["exists"]
        or not metrics_meta["exists"]
        or out.get("test_error")
        or out.get("val_error")
        or out.get("metrics_error")
    ):
        out["status"] = "error"
    return out


__all__ = [
    "ArtifactNotFoundError",
    "Split",
    "add_watchlist_item",
    "delete_watchlist_item",
    "get_active_artifacts",
    "get_model_manifest",
    "get_metrics",
    "get_player_advanced_profile",
    "get_player_profile",
    "get_player_history_strength",
    "get_player_report",
    "get_player_prediction",
    "get_predictions",
    "get_resolved_artifact_paths",
    "health_payload",
    "list_watchlist",
    "query_player_reports",
    "query_predictions",
    "query_scout_targets",
    "query_shortlist",
    "validate_strict_artifact_env",
]
