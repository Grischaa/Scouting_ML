from __future__ import annotations

from typing import Any


SUPPORTED_ACTIVE_LANES: tuple[str, ...] = ("valuation", "future_shortlist")
SUPPORTED_TRUST_SCOPES: tuple[str, ...] = ("trusted_only", "trusted_and_watch", "all")


def _slot(slot_key: str, slot_label: str, default_role_key: str) -> dict[str, str]:
    return {
        "slot_key": slot_key,
        "slot_label": slot_label,
        "default_role_key": default_role_key,
    }


SYSTEM_PROFILES: dict[str, dict[str, Any]] = {
    "high_press_433": {
        "template_key": "high_press_433",
        "label": "High-Press 4-3-3",
        "formation": "4-3-3",
        "press_intensity": "high",
        "defensive_line": "high",
        "build_up_style": "mixed",
        "transition_tempo": "high",
        "width_preference": "wide",
        "possession_preference": "balanced",
        "style_metric_weights": {
            "sb_pressures_per90": 1.0,
            "sofa_tackles_per90": 0.85,
            "sofa_interceptions_per90": 0.80,
            "sb_duel_win_rate": 0.70,
            "sb_progressive_passes_per90": 0.60,
            "sb_progressive_carries_per90": 0.55,
            "sofa_successfulDribbles_per90": 0.40,
            "sofa_expectedGoals_per90": 0.35,
        },
        "supported_lanes": list(SUPPORTED_ACTIVE_LANES),
        "supported_trust_scopes": list(SUPPORTED_TRUST_SCOPES),
        "slots": [
            _slot("GK", "Goalkeeper", "sweeper_gk"),
            _slot("RB", "Right Back", "overlapping_fb"),
            _slot("RCB", "Right Centre-Back", "ball_playing_cb"),
            _slot("LCB", "Left Centre-Back", "cover_cb"),
            _slot("LB", "Left Back", "inverted_fb"),
            _slot("DM", "Holding Midfielder", "anchor_6"),
            _slot("RCM", "Right No. 8", "runner_8"),
            _slot("LCM", "Left No. 8", "controller_8"),
            _slot("RW", "Right Winger", "touchline_winger"),
            _slot("LW", "Left Winger", "inside_winger"),
            _slot("ST", "Striker", "pressing_9"),
        ],
    },
    "transition_4231": {
        "template_key": "transition_4231",
        "label": "Transition 4-2-3-1",
        "formation": "4-2-3-1",
        "press_intensity": "medium",
        "defensive_line": "mid",
        "build_up_style": "direct",
        "transition_tempo": "high",
        "width_preference": "wide",
        "possession_preference": "reactive",
        "style_metric_weights": {
            "sb_progressive_carries_per90": 1.0,
            "sofa_successfulDribbles_per90": 0.90,
            "sb_passes_into_box_per90": 0.80,
            "sofa_expectedGoals_per90": 0.75,
            "sb_shot_assists_per90": 0.70,
            "sb_progressive_passes_per90": 0.55,
            "sb_pressures_per90": 0.30,
        },
        "supported_lanes": list(SUPPORTED_ACTIVE_LANES),
        "supported_trust_scopes": list(SUPPORTED_TRUST_SCOPES),
        "slots": [
            _slot("GK", "Goalkeeper", "shot_stopper_gk"),
            _slot("RB", "Right Back", "overlapping_fb"),
            _slot("RCB", "Right Centre-Back", "cover_cb"),
            _slot("LCB", "Left Centre-Back", "ball_playing_cb"),
            _slot("LB", "Left Back", "balanced_fb"),
            _slot("RDM", "Right Pivot", "ball_winner_6"),
            _slot("LDM", "Left Pivot", "deep_playmaker_6"),
            _slot("AM", "Attacking Midfielder", "chance_creator_10"),
            _slot("RW", "Right Winger", "inverted_winger"),
            _slot("LW", "Left Winger", "transition_winger"),
            _slot("ST", "Striker", "channel_runner_9"),
        ],
    },
}


def get_system_profile(template_key: str) -> dict[str, Any]:
    token = str(template_key or "").strip()
    if token not in SYSTEM_PROFILES:
        raise KeyError(f"Unknown system profile: {token}")
    return SYSTEM_PROFILES[token]


def list_system_profiles() -> list[dict[str, Any]]:
    return [get_system_profile(key) for key in SYSTEM_PROFILES]


__all__ = [
    "SUPPORTED_ACTIVE_LANES",
    "SUPPORTED_TRUST_SCOPES",
    "SYSTEM_PROFILES",
    "get_system_profile",
    "list_system_profiles",
]
