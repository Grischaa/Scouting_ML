"""System-fit templates and scoring for scouting workflows."""

from scouting_ml.scouting.role_templates import ROLE_TEMPLATES, get_role_template, list_role_templates
from scouting_ml.scouting.system_fit import (
    build_lane_posture,
    list_system_fit_templates,
    rank_system_fit_slots,
)
from scouting_ml.scouting.system_profiles import (
    SUPPORTED_ACTIVE_LANES,
    SUPPORTED_TRUST_SCOPES,
    SYSTEM_PROFILES,
    get_system_profile,
)

__all__ = [
    "ROLE_TEMPLATES",
    "SUPPORTED_ACTIVE_LANES",
    "SUPPORTED_TRUST_SCOPES",
    "SYSTEM_PROFILES",
    "build_lane_posture",
    "get_role_template",
    "get_system_profile",
    "list_role_templates",
    "list_system_fit_templates",
    "rank_system_fit_slots",
]
