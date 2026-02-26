"""Zero-shot player role classification using Hugging Face pipelines.

Assumptions:
- Input ``player_data`` is a dict with descriptive fields used by the prompt builder.
- The configured zero-shot model supports English inputs and the provided labels.

Limitations:
- Results depend on prompt quality and model generalization; no calibration is applied.
- Classification is skipped via explicit error if disabled in configuration.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from transformers import Pipeline, pipeline

from .config import NLPConfig, load_nlp_config
from .prompts import build_role_classification_text

logger = logging.getLogger(__name__)

_CLASSIFIER: Optional[Pipeline] = None
_MODEL_ID: Optional[str] = None

ROLE_LABELS: List[str] = [
    "Ball-playing centre-back",
    "Defensive stopper",
    "Box-to-box midfielder",
    "Deep-lying playmaker",
    "Inverted winger",
    "Pressing forward",
    "Target man",
]


def _resolve_device(device: str) -> int:
    """Convert a device string (cpu, cuda:0) to a transformers device index."""
    normalized = (device or "cpu").lower()
    if normalized in {"cpu", "-1"}:
        return -1
    if normalized.startswith("cuda"):
        parts = normalized.split(":")
        if len(parts) == 2 and parts[1].isdigit():
            return int(parts[1])
        return 0
    return -1


def _get_classifier(cfg: NLPConfig) -> Pipeline:
    """Lazy-load and cache the zero-shot classifier."""
    global _CLASSIFIER, _MODEL_ID
    if _CLASSIFIER is not None and _MODEL_ID == cfg.zero_shot_model:
        return _CLASSIFIER

    device_index = _resolve_device(cfg.hf_device)
    logger.info(
        "Loading zero-shot role classifier",
        extra={"model": cfg.zero_shot_model, "device": cfg.hf_device},
    )
    _CLASSIFIER = pipeline(
        "zero-shot-classification",
        model=cfg.zero_shot_model,
        device=device_index,
    )
    _MODEL_ID = cfg.zero_shot_model
    return _CLASSIFIER


def classify_player_role(player_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify a player's role using zero-shot learning.

    Returns a dict with ``role`` (or None if below threshold) and the associated
    confidence score. Raises explicit errors when classification is disabled or
    when inference fails.
    """
    cfg = load_nlp_config()
    if not cfg.enable_role_classification:
        logger.error("Role classification requested but disabled in configuration.")
        raise RuntimeError("Role classification is disabled by configuration.")

    prompt = build_role_classification_text(player_data)
    classifier = _get_classifier(cfg)

    try:
        logger.info(
            "Running zero-shot role classification",
            extra={"candidate_labels": ROLE_LABELS},
        )
        result: Dict[str, Any] = classifier(
            prompt,
            candidate_labels=ROLE_LABELS,
            multi_class=False,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Zero-shot classification failed")
        raise RuntimeError(f"Failed to classify player role: {exc}") from exc

    labels: List[str] = result.get("labels", [])
    scores: List[float] = result.get("scores", [])
    if not labels or not scores:
        raise RuntimeError("Unexpected zero-shot output format (missing labels or scores).")

    top_label = labels[0]
    top_score = float(scores[0])

    if top_score < cfg.role_confidence_threshold:
        logger.info(
            "Top score below confidence threshold",
            extra={
                "top_label": top_label,
                "top_score": top_score,
                "threshold": cfg.role_confidence_threshold,
            },
        )
        return {"role": None, "confidence": top_score}

    return {"role": top_label, "confidence": top_score}


__all__ = ["classify_player_role", "ROLE_LABELS"]
