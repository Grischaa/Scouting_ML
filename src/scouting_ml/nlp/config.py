"""Centralized NLP configuration for Hugging Face usage."""

from __future__ import annotations

import os
from dataclasses import dataclass

# Defaults used when corresponding environment variables are not set.
DEFAULT_HF_DEVICE = "cpu"
DEFAULT_SUMMARIZATION_MODEL = "facebook/bart-large-cnn"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
DEFAULT_MAX_SUMMARY_TOKENS = 180
DEFAULT_ROLE_CONFIDENCE_THRESHOLD = 0.55
DEFAULT_ENABLE_ROLE_CLASSIFICATION = True


@dataclass(frozen=True)
class NLPConfig:
    """Resolved NLP settings for downstream components."""

    hf_device: str = DEFAULT_HF_DEVICE
    summarization_model: str = DEFAULT_SUMMARIZATION_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    zero_shot_model: str = DEFAULT_ZERO_SHOT_MODEL
    max_summary_tokens: int = DEFAULT_MAX_SUMMARY_TOKENS
    role_confidence_threshold: float = DEFAULT_ROLE_CONFIDENCE_THRESHOLD
    enable_role_classification: bool = DEFAULT_ENABLE_ROLE_CLASSIFICATION


def _read_str(name: str, default: str) -> str:
    """Return a non-empty string from the environment or the provided default."""
    value = os.getenv(name, default).strip()
    return value or default


def _read_positive_int(name: str, default: int) -> int:
    """Parse a positive integer from the environment variable named ``name``."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def _read_probability(name: str, default: float) -> float:
    """Parse a probability-like float between 0 and 1 (inclusive)."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}.") from exc
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1, got {value}.")
    return value


def _read_bool(name: str, default: bool) -> bool:
    """Parse a boolean flag from the environment variable named ``name``."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean-like value, got {raw!r}.")


def load_nlp_config() -> NLPConfig:
    """Load and validate NLP configuration from environment variables."""
    return NLPConfig(
        hf_device=_read_str("HF_DEVICE", DEFAULT_HF_DEVICE),
        summarization_model=_read_str("SUMMARIZATION_MODEL", DEFAULT_SUMMARIZATION_MODEL),
        embedding_model=_read_str("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        zero_shot_model=_read_str("ZERO_SHOT_MODEL", DEFAULT_ZERO_SHOT_MODEL),
        max_summary_tokens=_read_positive_int(
            "MAX_SUMMARY_TOKENS", DEFAULT_MAX_SUMMARY_TOKENS
        ),
        role_confidence_threshold=_read_probability(
            "ROLE_CONFIDENCE_THRESHOLD", DEFAULT_ROLE_CONFIDENCE_THRESHOLD
        ),
        enable_role_classification=_read_bool(
            "ENABLE_ROLE_CLASSIFICATION", DEFAULT_ENABLE_ROLE_CLASSIFICATION
        ),
    )


__all__ = ["NLPConfig", "load_nlp_config"]
