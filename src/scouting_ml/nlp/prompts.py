"""Prompt builders for football scouting NLP tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


def _safe_text(value: Any, fallback: str) -> str:
    """Return a clean string or a fallback if the value is missing."""
    if value is None:
        return fallback
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or fallback
    return str(value)


def _to_float(value: Any) -> Optional[float]:
    """Best-effort float parsing; returns None when parsing is not possible."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _to_int(value: Any) -> Optional[int]:
    """Best-effort integer parsing; returns None when parsing is not possible."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(float(text))
        except ValueError:
            return None
    return None


def _format_stats(stats: Any) -> str:
    """Format key performance stats into a readable string."""
    if isinstance(stats, Mapping):
        items = [f"{k}: {v}" for k, v in stats.items() if v is not None]
        return "; ".join(items) if items else "not provided"
    if isinstance(stats, (list, tuple)):
        items = [str(item) for item in stats if item is not None]
        return "; ".join(items) if items else "not provided"
    if stats:
        return str(stats)
    return "not provided"


def build_scouting_report_prompt(player_data: Dict[str, Any]) -> str:
    """Build a neutral, structured scouting narrative for summarization or reporting."""
    name = _safe_text(player_data.get("name"), "The player")
    age_value = _to_int(player_data.get("age"))

    position = _safe_text(player_data.get("position"), "")
    league = _safe_text(player_data.get("league"), "")

    minutes_value = _to_int(
        player_data.get("minutes_played", player_data.get("minutes"))
    )
    minutes_phrase = (
        f"{minutes_value:,} minutes" if minutes_value is not None else "minutes not reported"
    )

    stats_text = _format_stats(
        player_data.get("key_performance_stats", player_data.get("stats"))
    )

    predicted_raw = player_data.get("predicted_market_value")
    current_raw = player_data.get("current_market_value")
    predicted_value = _to_float(predicted_raw)
    current_value = _to_float(current_raw)

    predicted_text = _safe_text(predicted_raw, "not provided")
    current_text = _safe_text(current_raw, "not provided")

    market_sentence: str
    if predicted_value is not None and current_value is not None:
        diff = predicted_value - current_value
        if abs(diff) < 1e-9:
            relation = "aligned with"
            diff_text = "with no notable difference"
        elif diff > 0:
            relation = "above"
            diff_text = f"by {abs(diff):.2f}"
        else:
            relation = "below"
            diff_text = f"by {abs(diff):.2f}"
        market_sentence = (
            f"Predicted market value ({predicted_value:.2f}) is {relation} the current estimate "
            f"({current_value:.2f}) {diff_text}."
        )
    else:
        market_sentence = (
            f"Predicted market value: {predicted_text}. Current market value: {current_text}."
            " Comparative insight is limited by missing values."
        )

    descriptor_parts = []
    if age_value is not None:
        descriptor_parts.append(f"{age_value}-year-old")
    if position:
        descriptor_parts.append(position)
    if league:
        descriptor_parts.append(f"competing in {league}")

    if descriptor_parts:
        article = "a " if age_value is not None or position else ""
        intro_sentence = f"{name} is {article}" + " ".join(descriptor_parts) + "."
    else:
        intro_sentence = (
            f"{name} profile summary focuses on playing time, performance, and market value."
        )

    minutes_sentence = f"The player has logged {minutes_phrase} this season."
    stats_sentence = (
        f"Key performance stats include {stats_text}."
        if stats_text != "not provided"
        else "Key performance stats are not provided."
    )

    return " ".join([intro_sentence, minutes_sentence, stats_sentence, market_sentence])


def build_embedding_profile_text(player_data: Dict[str, Any]) -> str:
    """Create a compact descriptive paragraph intended for semantic embeddings."""
    role_tendencies = _safe_text(
        player_data.get("role_tendencies", player_data.get("role")),
        "role tendencies not specified",
    )
    technical = _safe_text(
        player_data.get("technical_profile", player_data.get("technical")),
        "technical traits not specified",
    )
    physical = _safe_text(
        player_data.get("physical_profile", player_data.get("physical")),
        "physical traits not specified",
    )
    tactical = _safe_text(
        player_data.get("tactical_contribution", player_data.get("tactical")),
        "tactical contribution not specified",
    )
    league = _safe_text(player_data.get("league"), "league not specified")
    minutes_value = _to_int(
        player_data.get("minutes_played", player_data.get("minutes"))
    )
    minutes_phrase = (
        f"{minutes_value:,} minutes" if minutes_value is not None else "minutes not reported"
    )

    return (
        f"Role tendencies: {role_tendencies}. "
        f"Technical and physical profile: {technical}; {physical}. "
        f"Tactical contribution: {tactical}. "
        f"Context: {league} with {minutes_phrase}."
    )


def build_role_classification_text(player_data: Dict[str, Any]) -> str:
    """Generate a short behavioural profile for zero-shot role classification prompts."""
    tendencies = _safe_text(
        player_data.get("role_tendencies", player_data.get("role")),
        "role behaviours not specified",
    )
    on_ball = _safe_text(
        player_data.get("on_ball_actions"),
        "on-ball tendencies not detailed",
    )
    off_ball = _safe_text(
        player_data.get("off_ball_actions"),
        "off-ball work rate not detailed",
    )
    league = _safe_text(player_data.get("league"), "league not specified")
    minutes_value = _to_int(
        player_data.get("minutes_played", player_data.get("minutes"))
    )
    minutes_phrase = (
        f"{minutes_value:,} minutes" if minutes_value is not None else "limited minute data"
    )

    return (
        f"Behaviours: {tendencies}. "
        f"On-ball focus: {on_ball}. "
        f"Off-ball focus: {off_ball}. "
        f"Current context: operates in {league} with {minutes_phrase}."
    )


__all__ = [
    "build_scouting_report_prompt",
    "build_embedding_profile_text",
    "build_role_classification_text",
]
