"""Position-aware player similarity service backed by the clean parquet artifact."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from scouting_ml.core.runtime_config import PRODUCTION_PIPELINE_DEFAULTS


logger = logging.getLogger(__name__)

SCOUTING_CLEAN_DATASET_ENV = "SCOUTING_CLEAN_DATASET_PATH"
MODEL_ARTIFACTS_DIR_ENV = "MODEL_ARTIFACTS_DIR"
BIG5_LEAGUES = {
    "english premier league",
    "premier league",
    "spanish la liga",
    "la liga",
    "laliga",
    "italian serie a",
    "serie a",
    "german bundesliga",
    "bundesliga",
    "french ligue 1",
    "ligue 1",
}
EXCLUDED_NUMERIC_EXACT = {
    "market_value_eur",
    "predicted_value",
    "fair_value_eur",
    "expected_value_eur",
    "expected_value_low_eur",
    "expected_value_high_eur",
    "value_gap_eur",
    "value_gap_conservative_eur",
    "value_gap_raw_eur",
    "undervaluation_score",
    "undervaluation_confidence",
    "current_level_score",
    "current_level_confidence",
    "future_potential_score",
    "future_potential_confidence",
    "future_growth_probability",
    "future_scout_blend_score",
    "future_scout_score",
    "future_family_weighted_score",
    "next_market_value_eur",
    "value_growth_next_season_eur",
}
EXCLUDED_NUMERIC_PREFIXES = (
    "confidence_",
    "future_",
    "value_gap_",
)
SIMILARITY_MIN_GROUP_COVERAGE = 0.35
SIMILARITY_MAX_FEATURES = 32
SIMILARITY_MIN_FEATURES = 10
SIMILARITY_GLOBAL_MAX_FEATURES = 48
SIMILARITY_GROUP_TOKENS: dict[str, tuple[str, ...]] = {
    "GK": ("save", "clean", "cross", "claim", "launch", "throw", "keeper", "sweeper", "pass", "distribution", "shot"),
    "DF": ("tackle", "interception", "clearance", "block", "aerial", "duel", "progressive", "carry", "cross", "pass", "dribble", "press", "recovery"),
    "MF": ("assist", "chance", "keypass", "xg", "xa", "pass", "progressive", "carry", "dribble", "touch", "duel", "interception", "press"),
    "FW": ("goal", "shot", "assist", "xg", "xa", "keypass", "dribble", "carry", "touch", "box", "aerial", "duel", "press", "progressive"),
}
SIMILARITY_EXCLUDED_SUBSTRINGS = (
    "market_value",
    "fair_value",
    "expected_value",
    "value_gap",
    "undervalu",
    "confidence",
    "contract",
    "injury",
    "avail_",
    "availability",
    "fixture_",
    "odds_",
    "freshness",
    "trust",
    "leaguectx_",
    "uefa_",
    "history_",
    "snapshot",
    "retrieved",
    "source_",
    "season_end",
    "next_market",
    "growth",
    "player_id",
    "transfermarkt",
)
SIMILARITY_EXCLUDED_EXACT = {
    "age",
    "minutes",
    "minutes_sort",
    "season_sort",
    "contract_years_left",
    "injury_days_per_1000_min",
    "avail_reports",
    "avail_start_share",
    "avail_injury_count",
    "fixture_matches",
    "fixture_mean_rest_days",
    "fixture_congestion_share",
    "odds_implied_team_strength",
    "odds_upset_probability",
}


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


def _normalize_league_name(value: Any) -> str:
    """Return a normalized league key for comparisons and filters."""
    return str(value or "").strip().casefold()


def _position_group(frame: pd.DataFrame) -> pd.Series:
    """Infer a coarse position group for every row in the clean dataset."""
    if "model_position" in frame.columns:
        series = frame["model_position"].astype(str).str.upper().str.strip()
        return series.where(series.isin({"GK", "DF", "MF", "FW"}), "UNK")
    if "position_group" in frame.columns:
        series = frame["position_group"].astype(str).str.upper().str.strip()
        return series.where(series.isin({"GK", "DF", "MF", "FW"}), "UNK")
    main = frame.get("position_main", pd.Series("", index=frame.index)).astype(str).str.casefold()
    out = pd.Series("UNK", index=frame.index, dtype="object")
    out = out.mask(main.str.contains("keeper"), "GK")
    out = out.mask(main.str.contains("back|defen"), "DF")
    out = out.mask(main.str.contains("mid"), "MF")
    out = out.mask(main.str.contains("wing|striker|forward|attack"), "FW")
    return out


def _season_sort_value(frame: pd.DataFrame) -> pd.Series:
    """Build a sortable season-end proxy from the available season columns."""
    if "season_end_year" in frame.columns:
        values = pd.to_numeric(frame["season_end_year"], errors="coerce")
        if values.notna().any():
            return values.fillna(-1)
    season = frame.get("season", pd.Series("", index=frame.index)).astype(str)
    parsed = pd.to_numeric(season.str.extract(r"(\d{4})", expand=False), errors="coerce")
    return parsed.fillna(-1)


def _prediction_lookup() -> dict[tuple[str, str], dict[str, Any]]:
    """Load predicted and market values from active prediction artifacts."""
    from scouting_ml.services.market_value_service import get_predictions

    lookup: dict[tuple[str, str], dict[str, Any]] = {}
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
        if "predicted_value" not in work.columns:
            work["predicted_value"] = pd.to_numeric(
                work.get("fair_value_eur", work.get("expected_value_eur")),
                errors="coerce",
            )
        for row in work.to_dict(orient="records"):
            key = (str(row.get("player_id") or "").strip(), str(row.get("season") or "").strip())
            if not key[0]:
                continue
            lookup[key] = {
                "predicted_value": _safe_float(row.get("predicted_value")),
                "market_value_eur": _safe_float(row.get("market_value_eur")),
            }
    return lookup


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return numeric similarity features after excluding pricing outputs."""
    numeric_cols = frame.select_dtypes(include=[np.number]).columns.tolist()
    out: list[str] = []
    for col in numeric_cols:
        if col in EXCLUDED_NUMERIC_EXACT:
            continue
        if any(col.startswith(prefix) for prefix in EXCLUDED_NUMERIC_PREFIXES):
            continue
        if col.endswith("_confidence"):
            continue
        out.append(col)
    return out


def _normalized_metric_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().casefold()).strip("_")


def _is_similarity_feature_allowed(column: str) -> bool:
    key = _normalized_metric_key(column)
    if not key:
        return False
    if key in SIMILARITY_EXCLUDED_EXACT:
        return False
    if key in EXCLUDED_NUMERIC_EXACT:
        return False
    if any(key.startswith(prefix.rstrip("_")) for prefix in EXCLUDED_NUMERIC_PREFIXES):
        return False
    if any(part in key for part in SIMILARITY_EXCLUDED_SUBSTRINGS):
        return False
    if key.endswith("_confidence"):
        return False
    return True


def _coverage_filtered_columns(frame: pd.DataFrame, columns: list[str], *, limit: int) -> list[str]:
    if not columns:
        return []
    numeric = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    coverage = numeric.notna().mean(axis=0)
    variance = numeric.var(axis=0, skipna=True).fillna(0.0)
    kept: list[tuple[str, float]] = []
    for col in columns:
        if coverage.get(col, 0.0) < SIMILARITY_MIN_GROUP_COVERAGE:
            continue
        if float(variance.get(col, 0.0)) <= 0.0:
            continue
        kept.append((col, float(variance.get(col, 0.0))))
    kept.sort(key=lambda item: item[1], reverse=True)
    return [col for col, _ in kept[:limit]]


def _group_feature_columns(frame: pd.DataFrame, group: str) -> list[str]:
    numeric_cols = [col for col in frame.select_dtypes(include=[np.number]).columns if _is_similarity_feature_allowed(col)]
    if not numeric_cols:
        return []
    group_frame = frame[frame["position_group_norm"].astype(str) == str(group)].copy()
    if group_frame.empty:
        return []
    group_tokens = SIMILARITY_GROUP_TOKENS.get(str(group), ())
    token_matches = [col for col in numeric_cols if any(token in _normalized_metric_key(col) for token in group_tokens)]
    selected = _coverage_filtered_columns(group_frame, token_matches, limit=SIMILARITY_MAX_FEATURES)
    if len(selected) >= SIMILARITY_MIN_FEATURES:
        return selected
    return _coverage_filtered_columns(group_frame, numeric_cols, limit=SIMILARITY_MAX_FEATURES)


def _global_feature_columns(frame: pd.DataFrame) -> list[str]:
    numeric_cols = [col for col in frame.select_dtypes(include=[np.number]).columns if _is_similarity_feature_allowed(col)]
    return _coverage_filtered_columns(frame, numeric_cols, limit=SIMILARITY_GLOBAL_MAX_FEATURES)


@dataclass(frozen=True)
class _SimilarityIndex:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    position_feature_columns: dict[str, tuple[str, ...]]
    global_vectors: np.ndarray
    position_vectors: dict[str, np.ndarray]
    position_indices: dict[str, np.ndarray]


class SimilarityService:
    """Load the clean dataset once and serve cosine-similarity comparables."""

    def __init__(self, dataset_path: Path | None = None) -> None:
        self.dataset_path = dataset_path or _resolve_clean_dataset_path()
        self._index = self._build_index(self.dataset_path)

    @staticmethod
    def _prepare_base_frame(path: Path) -> pd.DataFrame:
        """Load and normalize the clean parquet rows used for similarity lookup."""
        if not path.exists():
            raise FileNotFoundError(f"Clean dataset not found: {path}")
        frame = pd.read_parquet(path)
        if frame.empty:
            raise ValueError(f"Clean dataset is empty: {path}")
        frame = frame.copy()
        if "player_id" not in frame.columns:
            raise ValueError("Clean dataset must include player_id for similarity lookup.")
        frame["player_id"] = frame["player_id"].astype(str)
        if "season" in frame.columns:
            frame["season"] = frame["season"].astype(str)
        else:
            frame["season"] = ""
        frame["position_group_norm"] = _position_group(frame)
        frame["season_sort"] = _season_sort_value(frame)
        if "minutes" in frame.columns:
            minutes_source = frame["minutes"]
        elif "sofa_minutesPlayed" in frame.columns:
            minutes_source = frame["sofa_minutesPlayed"]
        else:
            minutes_source = pd.Series(0.0, index=frame.index, dtype=float)
        minutes = pd.to_numeric(minutes_source, errors="coerce").fillna(0.0)
        frame["minutes_sort"] = minutes
        frame = frame.sort_values(["player_id", "season_sort", "minutes_sort"], ascending=[True, False, False])
        frame = frame.drop_duplicates(subset=["player_id", "season"], keep="first").reset_index(drop=True)

        pred_lookup = _prediction_lookup()
        predicted_values: list[float | None] = []
        market_values: list[float | None] = []
        for row in frame.to_dict(orient="records"):
            key = (str(row.get("player_id") or "").strip(), str(row.get("season") or "").strip())
            payload = pred_lookup.get(key, {})
            predicted_values.append(_safe_float(payload.get("predicted_value")))
            market_values.append(
                _safe_float(payload.get("market_value_eur")) or _safe_float(row.get("market_value_eur"))
            )
        frame["predicted_value"] = predicted_values
        frame["market_value_enriched_eur"] = market_values
        return frame

    @classmethod
    def _scaled_unit_vectors(cls, values: pd.DataFrame) -> np.ndarray:
        """Scale feature values and project them to unit-length vectors."""
        if values.empty:
            return np.zeros((0, 0), dtype=np.float32)
        filled = values.copy()
        medians = filled.median(axis=0, numeric_only=True)
        filled = filled.fillna(medians).fillna(0.0)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(filled).astype(np.float32, copy=False)
        norms = np.linalg.norm(scaled, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return (scaled / norms).astype(np.float32, copy=False)

    @classmethod
    def _build_index(cls, path: Path) -> _SimilarityIndex:
        """Build global and position-scoped cosine-similarity indices."""
        frame = cls._prepare_base_frame(path)
        feature_columns = tuple(_global_feature_columns(frame))
        if not feature_columns:
            raise ValueError("No numeric similarity features available in clean dataset.")

        feature_frame = frame.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
        global_vectors = cls._scaled_unit_vectors(feature_frame)
        position_vectors: dict[str, np.ndarray] = {}
        position_indices: dict[str, np.ndarray] = {}
        position_feature_columns: dict[str, tuple[str, ...]] = {}
        for position in ("GK", "DF", "MF", "FW"):
            idx = np.flatnonzero(frame["position_group_norm"].to_numpy() == position)
            if idx.size == 0:
                continue
            position_indices[position] = idx
            group_columns = tuple(_group_feature_columns(frame, position))
            if len(group_columns) < SIMILARITY_MIN_FEATURES:
                group_columns = feature_columns
            position_feature_columns[position] = group_columns
            group_frame = frame.iloc[idx].loc[:, list(group_columns)].apply(pd.to_numeric, errors="coerce")
            position_vectors[position] = cls._scaled_unit_vectors(group_frame)
        return _SimilarityIndex(
            frame=frame,
            feature_columns=feature_columns,
            position_feature_columns=position_feature_columns,
            global_vectors=global_vectors,
            position_vectors=position_vectors,
            position_indices=position_indices,
        )

    def _resolve_query_index(self, player_id: str, season: str | None = None) -> int:
        """Resolve the canonical row index for a player-season query."""
        work = self._index.frame[self._index.frame["player_id"].astype(str) == str(player_id)].copy()
        if season is not None:
            work = work[work["season"].astype(str) == str(season)].copy()
        if work.empty:
            raise ValueError(f"Player {player_id!r} not found in clean dataset.")
        work = work.sort_values(["season_sort", "minutes_sort"], ascending=False, na_position="last")
        return int(work.index[0])

    def get_player_row(self, player_id: str, season: str | None = None) -> pd.Series:
        """Return the canonical clean-dataset row for one player query."""
        return self._index.frame.loc[self._resolve_query_index(player_id, season=season)]

    def get_neighbor_frame(
        self,
        player_id: str,
        *,
        n: int = 25,
        same_position: bool = True,
        exclude_big5: bool = False,
        season: str | None = None,
    ) -> tuple[pd.Series, pd.DataFrame, dict[str, Any]]:
        """Return a query row plus ranked neighbor frame for internal consumers."""
        if int(n) <= 0:
            raise ValueError("n must be a positive integer.")
        query_idx = self._resolve_query_index(player_id, season=season)
        query_row = self._index.frame.loc[query_idx]
        query_position = str(query_row.get("position_group_norm") or "UNK")

        if same_position and query_position in self._index.position_vectors:
            candidate_indices = self._index.position_indices[query_position]
            candidate_vectors = self._index.position_vectors[query_position]
            local_query_pos = int(np.where(candidate_indices == query_idx)[0][0])
            query_vector = candidate_vectors[local_query_pos]
            active_columns = list(self._index.position_feature_columns.get(query_position, self._index.feature_columns))
        else:
            candidate_indices = np.arange(len(self._index.frame), dtype=int)
            candidate_vectors = self._index.global_vectors
            query_vector = candidate_vectors[query_idx]
            active_columns = list(self._index.feature_columns)

        scores = candidate_vectors @ query_vector
        order = np.argsort(scores)[::-1]
        seen_players: set[str] = {str(query_row.get("player_id") or "").strip()}
        rows: list[dict[str, Any]] = []
        for local_idx in order.tolist():
            global_idx = int(candidate_indices[local_idx])
            candidate = self._index.frame.iloc[global_idx]
            candidate_player_id = str(candidate.get("player_id") or "").strip()
            if not candidate_player_id or candidate_player_id in seen_players:
                continue
            if exclude_big5 and _normalize_league_name(candidate.get("league")) in BIG5_LEAGUES:
                continue
            similarity_score = float(np.clip(scores[local_idx], -1.0, 1.0))
            normalized_score = float(np.clip((similarity_score + 1.0) / 2.0, 0.0, 1.0))
            row = candidate.to_dict()
            row["_similarity_score"] = normalized_score
            rows.append(row)
            seen_players.add(candidate_player_id)
            if len(rows) >= int(n):
                break
        return query_row, pd.DataFrame(rows), {
            "position_group": query_position,
            "feature_count_used": int(len(active_columns)),
            "feature_columns_used": active_columns,
        }

    def similarity_payload(
        self,
        player_id: str,
        n: int = 5,
        same_position: bool = True,
        exclude_big5: bool = False,
        *,
        season: str | None = None,
    ) -> dict[str, Any]:
        """Return similar-player comparisons plus feature metadata."""
        query_row, neighbor_frame, metadata = self.get_neighbor_frame(
            player_id=player_id,
            n=n,
            same_position=same_position,
            exclude_big5=exclude_big5,
            season=season,
        )
        rows: list[dict[str, Any]] = []
        for match in neighbor_frame.to_dict(orient="records"):
            rows.append(
                {
                    "player_id": str(match.get("player_id") or "").strip(),
                    "name": str(match.get("name") or match.get("player_id") or ""),
                    "club": str(match.get("club") or ""),
                    "league": str(match.get("league") or ""),
                    "season": str(match.get("season") or ""),
                    "position": str(match.get("position_group_norm") or ""),
                    "similarity_score": _safe_float(match.get("_similarity_score")),
                    "predicted_value": _safe_float(match.get("predicted_value")),
                    "market_value_eur": _safe_float(match.get("market_value_enriched_eur")),
                }
            )
        return {
            "player_id": str(query_row.get("player_id") or player_id),
            "position_group": metadata["position_group"],
            "feature_count_used": metadata["feature_count_used"],
            "feature_columns_used": metadata["feature_columns_used"],
            "comparisons": rows,
        }

    def find_similar_players(
        self,
        player_id: str,
        n: int = 5,
        same_position: bool = True,
        exclude_big5: bool = False,
        *,
        season: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the nearest comparable player rows using cosine similarity."""
        return list(
            self.similarity_payload(
                player_id=player_id,
                n=n,
                same_position=same_position,
                exclude_big5=exclude_big5,
                season=season,
            ).get("comparisons", [])
        )


_SIMILARITY_SERVICE: SimilarityService | None = None
_SIMILARITY_LOCK = Lock()


def get_similarity_service(force_reload: bool = False) -> SimilarityService:
    """Return the singleton similarity service."""
    global _SIMILARITY_SERVICE
    if _SIMILARITY_SERVICE is not None and not force_reload:
        return _SIMILARITY_SERVICE
    with _SIMILARITY_LOCK:
        if _SIMILARITY_SERVICE is None or force_reload:
            _SIMILARITY_SERVICE = SimilarityService()
            logger.info("Loaded similarity service from %s", _SIMILARITY_SERVICE.dataset_path)
    return _SIMILARITY_SERVICE


def find_similar_players(
    player_id: str,
    n: int = 5,
    same_position: bool = True,
    exclude_big5: bool = False,
    *,
    season: str | None = None,
) -> list[dict[str, Any]]:
    """Convenience wrapper for singleton-backed similar-player lookup."""
    return get_similarity_service().find_similar_players(
        player_id=player_id,
        n=n,
        same_position=same_position,
        exclude_big5=exclude_big5,
        season=season,
    )


def get_similarity_payload(
    player_id: str,
    n: int = 5,
    same_position: bool = True,
    exclude_big5: bool = False,
    *,
    season: str | None = None,
) -> dict[str, Any]:
    """Convenience wrapper returning comparison rows plus feature metadata."""
    return get_similarity_service().similarity_payload(
        player_id=player_id,
        n=n,
        same_position=same_position,
        exclude_big5=exclude_big5,
        season=season,
    )


__all__ = [
    "SimilarityService",
    "find_similar_players",
    "get_similarity_payload",
    "get_similarity_service",
]
