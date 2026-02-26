"""NLP utilities and configuration."""

from .config import NLPConfig, load_nlp_config
from .prompts import (
    build_embedding_profile_text,
    build_role_classification_text,
    build_scouting_report_prompt,
)
from .role_classifier import ROLE_LABELS, classify_player_role
from .summarizer import generate_scouting_report

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
