"""Advisory kNN proxy estimates for sparse player-detail views."""

from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd

from scouting_ml.services.similarity_service import SimilarityService, get_similarity_service


logger = logging.getLogger(__name__)

PROXY_K = 8
PROXY_MIN_NEIGHBORS = 5
PROXY_MIN_MEAN_SIMILARITY = 0.60
PROXY_METRIC_SPECS: tuple[tuple[str, str], ...] = (
    ("sofa_goals_per90", "Goals / 90"),
    ("sofa_assists_per90", "Assists / 90"),
    ("sofa_expectedGoals_per90", "xG / 90"),
    ("sofa_keyPasses_per90", "Key passes / 90"),
    ("sofa_successfulDribbles_per90", "Successful dribbles / 90"),
    ("sb_progressive_passes_per90", "Progressive passes / 90"),
    ("sb_progressive_carries_per90", "Progressive carries / 90"),
    ("sofa_accuratePassesPercentage", "Pass accuracy %"),
    ("sofa_totalDuelsWonPercentage", "Duel win %"),
    ("sofa_aerialDuelsWonPercentage", "Aerial duel win %"),
    ("sb_pressures_per90", "Pressures / 90"),
)


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return float(parsed)


def _league_strength_bucket(value: Any) -> int:
    score = _safe_float(value)
    if score is None:
        return 1
    if score < 0.15:
        return 0
    if score < 0.30:
        return 1
    if score < 0.45:
        return 2
    return 3


def _support_label(*, mean_similarity: float, neighbor_count: int) -> str:
    if neighbor_count >= 7 and mean_similarity >= 0.72:
        return "strong"
    if neighbor_count >= 6 and mean_similarity >= 0.66:
        return "moderate"
    return "weak"


def _league_trust_tier(row: pd.Series) -> str:
    league = str(row.get("league") or "").strip().casefold()
    season = str(row.get("season") or "").strip().casefold()
    if not league:
        return "unknown"
    try:
        from scouting_ml.services.market_value_service import _league_trust_maps

        exact, latest = _league_trust_maps()
    except Exception:
        return "unknown"
    if league and season and (league, season) in exact:
        return str(exact[(league, season)] or "unknown")
    return str(latest.get(league, "unknown") or "unknown")


class ProxyEstimateService:
    """Serve advisory proxy metrics derived from comparable-player neighbors."""

    def __init__(self, similarity_service: SimilarityService | None = None) -> None:
        self._similarity_service = similarity_service or get_similarity_service()
        self.dataset_path: Path = self._similarity_service.dataset_path

    def _eligible_neighbors(self, row: pd.Series, neighbors: pd.DataFrame) -> pd.DataFrame:
        if neighbors.empty:
            return neighbors.copy()
        work = neighbors.copy()
        query_age = _safe_float(row.get("age"))
        if query_age is not None and "age" in work.columns:
            ages = pd.to_numeric(work["age"], errors="coerce")
            work = work[(ages >= query_age - 3.0) & (ages <= query_age + 3.0)].copy()

        query_minutes = _safe_float(row.get("minutes")) or _safe_float(row.get("sofa_minutesPlayed"))
        if query_minutes is not None and query_minutes > 0:
            minutes = pd.to_numeric(work.get("minutes", work.get("sofa_minutesPlayed")), errors="coerce")
            work = work[(minutes >= query_minutes * 0.5) & (minutes <= query_minutes * 2.0)].copy()

        query_bucket = _league_strength_bucket(row.get("leaguectx_league_strength_index"))
        if "leaguectx_league_strength_index" in work.columns:
            work["_league_strength_bucket"] = work["leaguectx_league_strength_index"].map(_league_strength_bucket)
            work = work[(work["_league_strength_bucket"] - query_bucket).abs() <= 1].copy()

        if "league_trust_tier" in work.columns:
            tier = work["league_trust_tier"].astype(str).str.strip().str.lower()
        else:
            tier = work.apply(_league_trust_tier, axis=1).astype(str).str.strip().str.lower()
        work = work[tier != "blocked"].copy()

        return work.sort_values("_similarity_score", ascending=False).head(PROXY_K).copy()

    def get_proxy_estimates(self, player_id: str, *, season: str | None = None) -> dict[str, Any]:
        """Return advisory proxy estimates for missing secondary metrics."""
        row = self._similarity_service.get_player_row(player_id=player_id, season=season)
        _, neighbors, meta = self._similarity_service.get_neighbor_frame(
            player_id=player_id,
            season=season,
            same_position=True,
            exclude_big5=False,
            n=40,
        )
        neighbors = self._eligible_neighbors(row, neighbors)
        if neighbors.empty or len(neighbors) < PROXY_MIN_NEIGHBORS:
            return {
                "available": False,
                "summary": "Comparable-player support is too thin to estimate missing metrics safely.",
                "metrics": [],
                "position_group": meta.get("position_group"),
            }

        weights = pd.to_numeric(neighbors["_similarity_score"], errors="coerce").fillna(0.0).clip(lower=0.0)
        mean_similarity = float(weights.mean()) if not weights.empty else 0.0
        if mean_similarity < PROXY_MIN_MEAN_SIMILARITY:
            return {
                "available": False,
                "summary": "Comparable-player similarity is too weak to estimate missing metrics safely.",
                "metrics": [],
                "position_group": meta.get("position_group"),
            }

        metrics: list[dict[str, Any]] = []
        for metric_key, label in PROXY_METRIC_SPECS:
            if metric_key not in row.index or _safe_float(row.get(metric_key)) is not None:
                continue
            if metric_key not in neighbors.columns:
                continue
            series = pd.to_numeric(neighbors[metric_key], errors="coerce")
            available = series.notna()
            if int(available.sum()) < PROXY_MIN_NEIGHBORS:
                continue
            metric_weights = weights.loc[available].copy()
            if metric_weights.empty or float(metric_weights.sum()) <= 0.0:
                continue
            value = float(np.average(series.loc[available], weights=metric_weights))
            metric_mean_similarity = float(metric_weights.mean())
            metrics.append(
                {
                    "metric_key": metric_key,
                    "label": label,
                    "estimated_value": round(value, 4),
                    "neighbor_count": int(available.sum()),
                    "mean_similarity": round(metric_mean_similarity, 4),
                    "support_label": _support_label(
                        mean_similarity=metric_mean_similarity,
                        neighbor_count=int(available.sum()),
                    ),
                }
            )

        if not metrics:
            return {
                "available": False,
                "summary": "This player already has enough direct metric coverage for the current advisory proxy set.",
                "metrics": [],
                "position_group": meta.get("position_group"),
            }

        metrics.sort(key=lambda item: (item["neighbor_count"], item["mean_similarity"]), reverse=True)
        summary = (
            f"{len(metrics)} proxy-estimated metrics from {len(neighbors)} comparable-player neighbors. "
            "Treat them as advisory, not observed data."
        )
        return {
            "available": True,
            "summary": summary,
            "metrics": metrics,
            "position_group": meta.get("position_group"),
        }


_PROXY_ESTIMATE_SERVICE: ProxyEstimateService | None = None
_PROXY_ESTIMATE_LOCK = Lock()


def get_proxy_estimate_service(force_reload: bool = False) -> ProxyEstimateService:
    """Return the singleton proxy-estimate service."""
    global _PROXY_ESTIMATE_SERVICE
    if _PROXY_ESTIMATE_SERVICE is not None and not force_reload:
        return _PROXY_ESTIMATE_SERVICE
    with _PROXY_ESTIMATE_LOCK:
        if _PROXY_ESTIMATE_SERVICE is None or force_reload:
            _PROXY_ESTIMATE_SERVICE = ProxyEstimateService()
            logger.info("Loaded proxy estimate service from %s", _PROXY_ESTIMATE_SERVICE.dataset_path)
    return _PROXY_ESTIMATE_SERVICE


def get_player_proxy_estimates(player_id: str, *, season: str | None = None) -> dict[str, Any]:
    """Convenience wrapper for singleton-backed proxy estimates."""
    return get_proxy_estimate_service().get_proxy_estimates(player_id=player_id, season=season)


__all__ = [
    "ProxyEstimateService",
    "get_player_proxy_estimates",
    "get_proxy_estimate_service",
]
