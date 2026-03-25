"""Player value trajectory service backed by the clean parquet artifact."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd

from scouting_ml.core.runtime_config import PRODUCTION_PIPELINE_DEFAULTS


logger = logging.getLogger(__name__)

SCOUTING_CLEAN_DATASET_ENV = "SCOUTING_CLEAN_DATASET_PATH"
MODEL_ARTIFACTS_DIR_ENV = "MODEL_ARTIFACTS_DIR"
TRAJECTORY_METRIC_CANDIDATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("minutes", ("minutes", "sofa_minutesPlayed")),
    ("goals", ("sofa_goals", "goals")),
    ("assists", ("sofa_assists", "assists")),
    ("xg", ("sofa_expectedGoals", "sb_xg", "xg")),
    ("progressive_passes", ("sb_progressive_passes", "progressive_passes")),
    ("progressive_carries", ("sb_progressive_carries", "progressive_carries")),
)


def _safe_float(value: Any) -> float | None:
    """Return a float when possible, otherwise ``None``."""
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_clean_dataset_path() -> Path:
    """Resolve the clean parquet path from env vars or production defaults."""
    explicit = os.getenv(SCOUTING_CLEAN_DATASET_ENV, "").strip()
    if explicit:
        return Path(explicit)
    artifacts_dir = os.getenv(MODEL_ARTIFACTS_DIR_ENV, "").strip()
    if artifacts_dir:
        return Path(artifacts_dir) / "champion_players_clean.parquet"
    return Path(PRODUCTION_PIPELINE_DEFAULTS.clean_output)


def _season_sort_value(frame: pd.DataFrame) -> pd.Series:
    """Build a sortable season-end proxy from the available season columns."""
    if "season_end_year" in frame.columns:
        parsed = pd.to_numeric(frame["season_end_year"], errors="coerce")
        if parsed.notna().any():
            return parsed.fillna(-1)
    season = frame.get("season", pd.Series("", index=frame.index)).astype(str)
    return pd.to_numeric(season.str.extract(r"(\d{4})", expand=False), errors="coerce").fillna(-1)


def _prediction_lookup() -> dict[tuple[str, str], float]:
    """Load the active prediction value for each player-season where available."""
    from scouting_ml.services.market_value_service import get_predictions

    lookup: dict[tuple[str, str], float] = {}
    for split in ("test", "val"):
        try:
            frame = get_predictions(split=split)
        except Exception:
            continue
        if frame.empty or "player_id" not in frame.columns:
            continue
        work = frame.copy()
        if "season" not in work.columns:
            work["season"] = ""
        predicted = pd.to_numeric(work.get("fair_value_eur", work.get("expected_value_eur")), errors="coerce")
        for row, value in zip(work.to_dict(orient="records"), predicted.tolist()):
            key = (str(row.get("player_id") or "").strip(), str(row.get("season") or "").strip())
            if not key[0]:
                continue
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            lookup[key] = float(value)
    return lookup


@dataclass(frozen=True)
class _TrajectoryData:
    frame: pd.DataFrame


class TrajectoryService:
    """Load player-season history once and compute simple value trajectories."""

    def __init__(self, dataset_path: Path | None = None) -> None:
        self.dataset_path = dataset_path or _resolve_clean_dataset_path()
        self._data = self._load_data(self.dataset_path)

    @staticmethod
    def _load_data(path: Path) -> _TrajectoryData:
        """Load the clean parquet and enrich it with active predicted values."""
        if not path.exists():
            raise FileNotFoundError(f"Clean dataset not found: {path}")
        frame = pd.read_parquet(path)
        if frame.empty:
            raise ValueError(f"Clean dataset is empty: {path}")
        frame = frame.copy()
        if "player_id" not in frame.columns:
            raise ValueError("Clean dataset must include player_id for trajectory lookup.")
        frame["player_id"] = frame["player_id"].astype(str)
        if "season" in frame.columns:
            frame["season"] = frame["season"].astype(str)
        else:
            frame["season"] = ""
        frame["season_sort"] = _season_sort_value(frame)

        predicted_lookup = _prediction_lookup()
        predicted_values: list[float | None] = []
        predicted_sources: list[str] = []
        for row in frame.to_dict(orient="records"):
            key = (str(row.get("player_id") or "").strip(), str(row.get("season") or "").strip())
            predicted = predicted_lookup.get(key)
            if predicted is not None:
                predicted_values.append(float(predicted))
                predicted_sources.append("model")
            else:
                fallback = _safe_float(row.get("market_value_eur"))
                predicted_values.append(fallback)
                predicted_sources.append("market_fallback" if fallback is not None else "missing")
        frame["predicted_value"] = predicted_values
        frame["predicted_value_source"] = predicted_sources
        return _TrajectoryData(frame=frame)

    def _resolve_query_row(self, player_id: str, season: str | None = None) -> pd.Series:
        """Resolve the player-season row that anchors a trajectory request."""
        work = self._data.frame[self._data.frame["player_id"].astype(str) == str(player_id)].copy()
        if season is not None:
            work = work[work["season"].astype(str) == str(season)].copy()
        if work.empty:
            raise ValueError(f"Player {player_id!r} not found in clean dataset.")
        work = work.sort_values("season_sort", ascending=False, na_position="last")
        return work.iloc[0].copy()

    @staticmethod
    def _history_mask(frame: pd.DataFrame, row: pd.Series) -> pd.Series:
        """Return a mask that matches the most stable available player identity."""
        if pd.notna(row.get("player_id")):
            mask = frame["player_id"].astype(str) == str(row.get("player_id"))
            if bool(mask.any()):
                return mask
        for stable_key in ("transfermarkt_id", "sb_transfermarkt_id"):
            if stable_key in frame.columns and pd.notna(row.get(stable_key)):
                mask = frame[stable_key].astype(str) == str(row.get(stable_key))
                if bool(mask.any()):
                    return mask
        if "dob" in frame.columns and pd.notna(row.get("dob")):
            mask = (
                frame.get("name", pd.Series("", index=frame.index)).astype(str).str.casefold()
                == str(row.get("name") or "").casefold()
            ) & (frame["dob"].astype(str) == str(row.get("dob")))
            if bool(mask.any()):
                return mask
        return frame.get("name", pd.Series("", index=frame.index)).astype(str).str.casefold() == str(
            row.get("name") or ""
        ).casefold()

    @staticmethod
    def _season_rows(work: pd.DataFrame) -> list[dict[str, Any]]:
        """Serialize season rows and attach season-over-season deltas."""
        rows: list[dict[str, Any]] = []
        for _, row in work.iterrows():
            payload: dict[str, Any] = {
                "season": str(row.get("season") or ""),
                "club": str(row.get("club") or ""),
                "league": str(row.get("league") or ""),
                "market_value_eur": _safe_float(row.get("market_value_eur")),
                "predicted_value": _safe_float(row.get("predicted_value")),
                "predicted_value_source": str(row.get("predicted_value_source") or ""),
            }
            for label, candidates in TRAJECTORY_METRIC_CANDIDATES:
                value = None
                for candidate in candidates:
                    value = _safe_float(row.get(candidate))
                    if value is not None:
                        break
                payload[label] = value
            rows.append(payload)
        prior: dict[str, Any] | None = None
        delta_fields = (
            "predicted_value",
            "market_value_eur",
            "minutes",
            "goals",
            "assists",
            "xg",
            "progressive_passes",
            "progressive_carries",
        )
        for payload in rows:
            for field in delta_fields:
                current = _safe_float(payload.get(field))
                previous = _safe_float(prior.get(field)) if isinstance(prior, dict) else None
                payload[f"delta_{field}"] = (
                    float(current - previous)
                    if current is not None and previous is not None
                    else None
                )
            prior = payload
        return rows

    @staticmethod
    def _fit_value_trend(seasons: list[dict[str, Any]]) -> tuple[float, float, float | None]:
        """Fit a simple linear trend over predicted value across seasons."""
        values = [item.get("predicted_value") for item in seasons]
        usable = [(idx, float(value)) for idx, value in enumerate(values) if value is not None]
        if len(usable) < 2:
            only_value = usable[0][1] if usable else None
            return 0.0, 0.0, only_value
        x = np.array([idx for idx, _ in usable], dtype=float)
        y = np.array([value for _, value in usable], dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        fitted = (slope * x) + intercept
        residual = np.sum((y - fitted) ** 2)
        total = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - residual / total if total > 0 else 1.0
        base = float(abs(y[0])) if y[0] else float(max(abs(y).max(), 1.0))
        slope_pct = float((slope / max(base, 1.0)) * 100.0)
        projected_next = float((slope * len(usable)) + intercept)
        return slope_pct, float(np.clip(r2, -1.0, 1.0)), max(projected_next, 0.0)

    @staticmethod
    def _trajectory_label(slope_pct: float) -> str:
        """Map slope percentage to a simple trajectory label."""
        if slope_pct > 5.0:
            return "ascending"
        if slope_pct < -5.0:
            return "declining"
        return "stable"

    def get_player_trajectory(self, player_id: str, season: str | None = None) -> dict[str, Any]:
        """Return a compact multi-season trajectory view for one player."""
        query_row = self._resolve_query_row(player_id, season=season)
        mask = self._history_mask(self._data.frame, query_row)
        work = self._data.frame.loc[mask].copy()
        if work.empty:
            raise ValueError(f"No trajectory history available for player {player_id!r}.")
        work = work.sort_values("season_sort", ascending=True, na_position="last")
        seasons = self._season_rows(work)
        slope_pct, r2, projected_next_value = self._fit_value_trend(seasons)
        predicted_values = [item.get("predicted_value") for item in seasons if item.get("predicted_value") is not None]
        if predicted_values:
            peak_idx = max(
                range(len(seasons)),
                key=lambda idx: seasons[idx].get("predicted_value") if seasons[idx].get("predicted_value") is not None else -1,
            )
            peak_season = seasons[peak_idx].get("season")
        else:
            peak_season = seasons[-1].get("season")
        return {
            "player_id": str(query_row.get("player_id") or player_id),
            "trajectory_label": self._trajectory_label(slope_pct),
            "slope_pct": slope_pct,
            "r2": r2,
            "projected_next_value": projected_next_value,
            "peak_season": peak_season,
            "seasons": seasons,
        }


_TRAJECTORY_SERVICE: TrajectoryService | None = None
_TRAJECTORY_LOCK = Lock()


def get_trajectory_service(force_reload: bool = False) -> TrajectoryService:
    """Return the singleton trajectory service."""
    global _TRAJECTORY_SERVICE
    if _TRAJECTORY_SERVICE is not None and not force_reload:
        return _TRAJECTORY_SERVICE
    with _TRAJECTORY_LOCK:
        if _TRAJECTORY_SERVICE is None or force_reload:
            _TRAJECTORY_SERVICE = TrajectoryService()
            logger.info("Loaded trajectory service from %s", _TRAJECTORY_SERVICE.dataset_path)
    return _TRAJECTORY_SERVICE


def get_player_trajectory(player_id: str, season: str | None = None) -> dict[str, Any]:
    """Convenience wrapper for singleton-backed trajectory lookup."""
    return get_trajectory_service().get_player_trajectory(player_id=player_id, season=season)


__all__ = [
    "TrajectoryService",
    "get_player_trajectory",
    "get_trajectory_service",
]
