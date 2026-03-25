from __future__ import annotations

from typing import Any


def _role_template(
    *,
    role_key: str,
    label: str,
    allowed_position_families: tuple[str, ...],
    family_weights: dict[str, float],
    metric_weights: dict[str, float],
    minimum_confidence: float,
    budget_bias: float,
) -> dict[str, Any]:
    return {
        "role_key": role_key,
        "label": label,
        "allowed_position_families": list(allowed_position_families),
        "family_weights": dict(family_weights),
        "metric_weights": dict(metric_weights),
        "minimum_confidence": float(minimum_confidence),
        "budget_bias": float(budget_bias),
    }


ROLE_TEMPLATES: dict[str, dict[str, Any]] = {
    "sweeper_gk": _role_template(
        role_key="sweeper_gk",
        label="Sweeper GK",
        allowed_position_families=("GK",),
        family_weights={"impact": 0.12, "technical": 0.18, "tactical": 0.24, "physical": 0.22, "context": 0.10, "trajectory": 0.14},
        metric_weights={
            "sofa_successfulRunsOut": 1.0,
            "sofa_accuratePassesPercentage": 0.9,
            "sofa_accurateLongBallsPercentage": 0.8,
            "sofa_highClaims": 0.6,
        },
        minimum_confidence=50.0,
        budget_bias=0.05,
    ),
    "shot_stopper_gk": _role_template(
        role_key="shot_stopper_gk",
        label="Shot-Stopper GK",
        allowed_position_families=("GK",),
        family_weights={"impact": 0.18, "technical": 0.10, "tactical": 0.24, "physical": 0.20, "context": 0.12, "trajectory": 0.16},
        metric_weights={
            "sofa_saves": 1.0,
            "sofa_highClaims": 0.7,
            "sofa_successfulRunsOut": 0.4,
            "sofa_accuratePassesPercentage": 0.3,
        },
        minimum_confidence=50.0,
        budget_bias=0.0,
    ),
    "ball_playing_cb": _role_template(
        role_key="ball_playing_cb",
        label="Ball-Playing CB",
        allowed_position_families=("CB", "FB"),
        family_weights={"impact": 0.18, "technical": 0.18, "tactical": 0.28, "physical": 0.14, "context": 0.10, "trajectory": 0.12},
        metric_weights={
            "sb_progressive_passes_per90": 1.0,
            "sofa_accuratePassesPercentage": 0.9,
            "sofa_interceptions_per90": 0.6,
            "sb_duel_win_rate": 0.5,
        },
        minimum_confidence=52.0,
        budget_bias=0.05,
    ),
    "cover_cb": _role_template(
        role_key="cover_cb",
        label="Cover CB",
        allowed_position_families=("CB", "FB"),
        family_weights={"impact": 0.16, "technical": 0.10, "tactical": 0.30, "physical": 0.22, "context": 0.12, "trajectory": 0.10},
        metric_weights={
            "sofa_interceptions_per90": 0.9,
            "sofa_clearances_per90": 0.8,
            "sb_duel_win_rate": 0.8,
            "sb_aerial_win_rate": 0.7,
        },
        minimum_confidence=52.0,
        budget_bias=0.0,
    ),
    "overlapping_fb": _role_template(
        role_key="overlapping_fb",
        label="Overlapping FB",
        allowed_position_families=("FB", "W"),
        family_weights={"impact": 0.20, "technical": 0.18, "tactical": 0.16, "physical": 0.22, "context": 0.10, "trajectory": 0.14},
        metric_weights={
            "sb_progressive_carries_per90": 1.0,
            "sofa_accurateCrossesPercentage": 0.8,
            "sofa_keyPasses_per90": 0.7,
            "sofa_tackles_per90": 0.6,
        },
        minimum_confidence=48.0,
        budget_bias=0.0,
    ),
    "inverted_fb": _role_template(
        role_key="inverted_fb",
        label="Inverted FB",
        allowed_position_families=("FB", "CM"),
        family_weights={"impact": 0.16, "technical": 0.20, "tactical": 0.24, "physical": 0.16, "context": 0.10, "trajectory": 0.14},
        metric_weights={
            "sb_progressive_passes_per90": 1.0,
            "sofa_accuratePassesPercentage": 0.9,
            "sofa_interceptions_per90": 0.7,
            "sofa_tackles_per90": 0.6,
        },
        minimum_confidence=48.0,
        budget_bias=0.05,
    ),
    "balanced_fb": _role_template(
        role_key="balanced_fb",
        label="Balanced FB",
        allowed_position_families=("FB",),
        family_weights={"impact": 0.18, "technical": 0.16, "tactical": 0.20, "physical": 0.20, "context": 0.12, "trajectory": 0.14},
        metric_weights={
            "sofa_tackles_per90": 0.8,
            "sofa_interceptions_per90": 0.8,
            "sb_progressive_carries_per90": 0.6,
            "sofa_accuratePassesPercentage": 0.5,
        },
        minimum_confidence=48.0,
        budget_bias=0.0,
    ),
    "anchor_6": _role_template(
        role_key="anchor_6",
        label="Anchor 6",
        allowed_position_families=("DM", "CM"),
        family_weights={"impact": 0.14, "technical": 0.12, "tactical": 0.30, "physical": 0.18, "context": 0.12, "trajectory": 0.14},
        metric_weights={
            "sofa_interceptions_per90": 1.0,
            "sofa_tackles_per90": 0.9,
            "sb_duel_win_rate": 0.8,
            "sofa_accuratePassesPercentage": 0.5,
        },
        minimum_confidence=52.0,
        budget_bias=0.0,
    ),
    "ball_winner_6": _role_template(
        role_key="ball_winner_6",
        label="Ball-Winner 6",
        allowed_position_families=("DM", "CM"),
        family_weights={"impact": 0.16, "technical": 0.10, "tactical": 0.28, "physical": 0.20, "context": 0.10, "trajectory": 0.16},
        metric_weights={
            "sofa_tackles_per90": 1.0,
            "sofa_interceptions_per90": 1.0,
            "sb_duel_win_rate": 0.8,
            "sb_pressures_per90": 0.7,
        },
        minimum_confidence=50.0,
        budget_bias=-0.05,
    ),
    "deep_playmaker_6": _role_template(
        role_key="deep_playmaker_6",
        label="Deep Playmaker 6",
        allowed_position_families=("DM", "CM", "AM"),
        family_weights={"impact": 0.16, "technical": 0.22, "tactical": 0.24, "physical": 0.10, "context": 0.10, "trajectory": 0.18},
        metric_weights={
            "sb_progressive_passes_per90": 1.0,
            "sofa_accuratePassesPercentage": 1.0,
            "sofa_accurateFinalThirdPasses_per90": 0.7,
            "sofa_interceptions_per90": 0.4,
        },
        minimum_confidence=50.0,
        budget_bias=0.05,
    ),
    "runner_8": _role_template(
        role_key="runner_8",
        label="Runner 8",
        allowed_position_families=("CM", "AM"),
        family_weights={"impact": 0.18, "technical": 0.16, "tactical": 0.20, "physical": 0.18, "context": 0.10, "trajectory": 0.18},
        metric_weights={
            "sb_progressive_carries_per90": 1.0,
            "sofa_tackles_per90": 0.8,
            "sofa_keyPasses_per90": 0.6,
            "sofa_expectedGoals_per90": 0.5,
        },
        minimum_confidence=46.0,
        budget_bias=-0.05,
    ),
    "controller_8": _role_template(
        role_key="controller_8",
        label="Controller 8",
        allowed_position_families=("CM", "DM", "AM"),
        family_weights={"impact": 0.16, "technical": 0.22, "tactical": 0.24, "physical": 0.08, "context": 0.12, "trajectory": 0.18},
        metric_weights={
            "sofa_accuratePassesPercentage": 1.0,
            "sb_progressive_passes_per90": 0.9,
            "sofa_keyPasses_per90": 0.6,
            "sofa_interceptions_per90": 0.3,
        },
        minimum_confidence=46.0,
        budget_bias=0.0,
    ),
    "chance_creator_10": _role_template(
        role_key="chance_creator_10",
        label="Chance-Creator 10",
        allowed_position_families=("AM", "CM", "W"),
        family_weights={"impact": 0.24, "technical": 0.22, "tactical": 0.18, "physical": 0.06, "context": 0.10, "trajectory": 0.20},
        metric_weights={
            "sofa_keyPasses_per90": 1.0,
            "sb_shot_assists_per90": 1.0,
            "sb_passes_into_box_per90": 0.9,
            "sofa_successfulDribbles_per90": 0.5,
        },
        minimum_confidence=48.0,
        budget_bias=0.05,
    ),
    "touchline_winger": _role_template(
        role_key="touchline_winger",
        label="Touchline Winger",
        allowed_position_families=("W", "AM"),
        family_weights={"impact": 0.20, "technical": 0.24, "tactical": 0.12, "physical": 0.14, "context": 0.10, "trajectory": 0.20},
        metric_weights={
            "sofa_successfulDribbles_per90": 1.0,
            "sb_progressive_carries_per90": 0.9,
            "sofa_accurateCrossesPercentage": 0.8,
            "sb_shot_assists_per90": 0.7,
        },
        minimum_confidence=46.0,
        budget_bias=-0.05,
    ),
    "inside_winger": _role_template(
        role_key="inside_winger",
        label="Inside Winger",
        allowed_position_families=("W", "AM", "ST"),
        family_weights={"impact": 0.28, "technical": 0.20, "tactical": 0.14, "physical": 0.08, "context": 0.10, "trajectory": 0.20},
        metric_weights={
            "sofa_expectedGoals_per90": 1.0,
            "sb_passes_into_box_per90": 0.8,
            "sofa_successfulDribbles_per90": 0.8,
            "sofa_goalConversionPercentage": 0.6,
        },
        minimum_confidence=46.0,
        budget_bias=0.0,
    ),
    "inverted_winger": _role_template(
        role_key="inverted_winger",
        label="Inverted Winger",
        allowed_position_families=("W", "AM", "ST"),
        family_weights={"impact": 0.26, "technical": 0.20, "tactical": 0.16, "physical": 0.08, "context": 0.10, "trajectory": 0.20},
        metric_weights={
            "sofa_expectedGoals_per90": 0.9,
            "sb_passes_into_box_per90": 0.8,
            "sofa_successfulDribbles_per90": 0.7,
            "sofa_keyPasses_per90": 0.5,
        },
        minimum_confidence=46.0,
        budget_bias=0.0,
    ),
    "transition_winger": _role_template(
        role_key="transition_winger",
        label="Transition Winger",
        allowed_position_families=("W", "AM", "ST"),
        family_weights={"impact": 0.22, "technical": 0.16, "tactical": 0.12, "physical": 0.20, "context": 0.10, "trajectory": 0.20},
        metric_weights={
            "sb_progressive_carries_per90": 1.0,
            "sofa_successfulDribbles_per90": 0.9,
            "sofa_expectedGoals_per90": 0.8,
            "sb_passes_into_box_per90": 0.7,
        },
        minimum_confidence=46.0,
        budget_bias=-0.05,
    ),
    "pressing_9": _role_template(
        role_key="pressing_9",
        label="Pressing 9",
        allowed_position_families=("ST", "W", "AM"),
        family_weights={"impact": 0.26, "technical": 0.18, "tactical": 0.14, "physical": 0.14, "context": 0.10, "trajectory": 0.18},
        metric_weights={
            "sofa_expectedGoals_per90": 1.0,
            "sb_pressures_per90": 0.8,
            "sofa_totalShots_per90": 0.7,
            "sofa_successfulDribbles_per90": 0.4,
        },
        minimum_confidence=48.0,
        budget_bias=0.05,
    ),
    "channel_runner_9": _role_template(
        role_key="channel_runner_9",
        label="Channel-Runner 9",
        allowed_position_families=("ST", "W"),
        family_weights={"impact": 0.28, "technical": 0.16, "tactical": 0.10, "physical": 0.18, "context": 0.10, "trajectory": 0.18},
        metric_weights={
            "sofa_expectedGoals_per90": 1.0,
            "sb_progressive_carries_per90": 0.7,
            "sofa_totalShots_per90": 0.8,
            "sofa_successfulDribbles_per90": 0.5,
        },
        minimum_confidence=48.0,
        budget_bias=0.05,
    ),
}


def get_role_template(role_key: str) -> dict[str, Any]:
    token = str(role_key or "").strip()
    if token not in ROLE_TEMPLATES:
        raise KeyError(f"Unknown role template: {token}")
    return ROLE_TEMPLATES[token]


def list_role_templates(role_keys: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    keys = role_keys or tuple(ROLE_TEMPLATES.keys())
    return [get_role_template(key) for key in keys]


__all__ = ["ROLE_TEMPLATES", "get_role_template", "list_role_templates"]
