"""Hugging Face-based summarization for scouting reports."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from transformers import Pipeline, pipeline

from .config import NLPConfig, load_nlp_config
from .prompts import build_scouting_report_prompt

logger = logging.getLogger(__name__)

_SUMMARIZER: Optional[Pipeline] = None
_MODEL_ID: Optional[str] = None
_MAX_INPUT_WORDS = 1024  # approximate safeguard to keep prompts within model limits


def _resolve_device(device: str) -> int:
    """Convert a device string (cpu, cuda:0) to the pipeline device index."""
    normalized = (device or "cpu").lower()
    if normalized in {"cpu", "-1"}:
        return -1
    if normalized.startswith("cuda"):
        parts = normalized.split(":")
        if len(parts) == 2 and parts[1].isdigit():
            return int(parts[1])
        return 0
    return -1


def _get_summarizer(cfg: NLPConfig) -> Pipeline:
    """Lazy-load and cache the summarization pipeline for the configured model."""
    global _SUMMARIZER, _MODEL_ID
    if _SUMMARIZER is not None and _MODEL_ID == cfg.summarization_model:
        return _SUMMARIZER

    device_index = _resolve_device(cfg.hf_device)
    logger.info(
        "Loading summarization pipeline",
        extra={"model": cfg.summarization_model, "device": cfg.hf_device},
    )
    _SUMMARIZER = pipeline(
        "summarization",
        model=cfg.summarization_model,
        device=device_index,
    )
    _MODEL_ID = cfg.summarization_model
    return _SUMMARIZER


def _truncate_prompt(prompt: str, max_words: int) -> str:
    """Trim prompt length to reduce the chance of exceeding model token limits."""
    words = prompt.split()
    if len(words) <= max_words:
        return prompt
    logger.warning(
        "Prompt truncated for length safeguard",
        extra={"original_words": len(words), "kept_words": max_words},
    )
    return " ".join(words[:max_words])


def _clean_summary(text: str) -> str:
    """Normalize whitespace in the generated summary."""
    return " ".join(text.strip().split())


def generate_scouting_report(player_data: Dict[str, Any]) -> str:
    """
    Generate a concise, human-readable scouting report via summarization.

    This builds a structured prompt from player data, runs the Hugging Face
    summarization pipeline, and returns cleaned summary text. Errors are surfaced
    with clear messages to aid debugging.
    """
    cfg = load_nlp_config()
    prompt = build_scouting_report_prompt(player_data)
    prompt = _truncate_prompt(prompt, _MAX_INPUT_WORDS)

    summarizer = _get_summarizer(cfg)
    max_len = max(cfg.max_summary_tokens, 32)
    min_len = max(20, int(max_len * 0.4))

    try:
        logger.info(
            "Generating scouting summary",
            extra={"max_length": max_len, "min_length": min_len},
        )
        results: List[Dict[str, Any]] = summarizer(
            prompt,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,
            truncation=True,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Summarization failed")
        raise RuntimeError(f"Failed to generate scouting report: {exc}") from exc

    if not results:
        raise RuntimeError("No summary returned from the summarization pipeline.")

    summary_text = results[0].get("summary_text")
    if not isinstance(summary_text, str):
        raise RuntimeError("Unexpected summarization output format.")

    return _clean_summary(summary_text)


__all__ = ["generate_scouting_report"]
