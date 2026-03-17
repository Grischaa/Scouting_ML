"""NLP utilities and configuration with lazy heavy-model imports."""

from __future__ import annotations

from typing import Any, Dict

from .config import NLPConfig, load_nlp_config
from .prompts import (
    build_embedding_profile_text,
    build_role_classification_text,
    build_scouting_report_prompt,
)

ROLE_LABELS = [
    "Ball-playing centre-back",
    "Defensive stopper",
    "Box-to-box midfielder",
    "Deep-lying playmaker",
    "Inverted winger",
    "Pressing forward",
    "Target man",
]


def generate_scouting_report(player_data: Dict[str, Any]) -> str:
    from .summarizer import generate_scouting_report as _generate_scouting_report

    return _generate_scouting_report(player_data)


def classify_player_role(player_data: Dict[str, Any]) -> Dict[str, Any]:
    from .role_classifier import classify_player_role as _classify_player_role

    return _classify_player_role(player_data)


__all__ = [
    "NLPConfig",
    "load_nlp_config",
    "build_scouting_report_prompt",
    "build_embedding_profile_text",
    "build_role_classification_text",
    "generate_scouting_report",
    "classify_player_role",
    "ROLE_LABELS",
]
