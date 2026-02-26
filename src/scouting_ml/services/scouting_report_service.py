"""Service wrapper around the scouting report summarizer."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Mapping

from scouting_ml.nlp import generate_scouting_report

logger = logging.getLogger(__name__)

_REPORT_CACHE: Dict[str, str] = {}


def _validate_player_data(player_data: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and normalize incoming player data for report generation."""
    if not isinstance(player_data, Mapping):
        raise ValueError("player_data must be a mapping.")

    data = dict(player_data)
    if not data:
        raise ValueError("player_data is required and cannot be empty.")

    if not any(key in data for key in ("name", "player_id", "id")):
        raise ValueError("player_data must include a player identifier (name or id).")

    return data


def _cache_key(player_data: Dict[str, Any]) -> str:
    """Create a stable cache key from player data."""
    try:
        return json.dumps(player_data, sort_keys=True, default=str)
    except TypeError:
        # Fallback to repr when data contains non-JSON-serializable values.
        return repr(sorted(player_data.items()))


def get_scouting_report(player_data: Mapping[str, Any]) -> str:
    """Validate input, leverage caching, and return a scouting report."""
    data = _validate_player_data(player_data)
    key = _cache_key(data)

    if key in _REPORT_CACHE:
        logger.info("Returning cached scouting report", extra={"cache_hit": True})
        return _REPORT_CACHE[key]

    logger.info("Generating new scouting report", extra={"cache_hit": False})
    report = generate_scouting_report(data)
    _REPORT_CACHE[key] = report
    return report


__all__ = ["get_scouting_report"]
