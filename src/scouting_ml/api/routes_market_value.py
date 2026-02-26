"""FastAPI routes for market-value model artifacts and retrieval."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from scouting_ml.services.market_value_service import (
    ArtifactNotFoundError,
    add_watchlist_item,
    delete_watchlist_item,
    get_active_artifacts,
    get_model_manifest,
    get_metrics,
    get_player_history_strength,
    get_player_report,
    get_player_prediction,
    health_payload,
    list_watchlist,
    query_predictions,
    query_scout_targets,
    query_shortlist,
)

router = APIRouter(prefix="/market-value", tags=["market_value"])


class HealthResponse(BaseModel):
    status: str
    artifacts: Dict[str, Any]
    test_rows: Optional[int] = None
    val_rows: Optional[int] = None
    metrics_loaded: Optional[bool] = None
    metrics_dataset: Optional[str] = None
    metrics_test_season: Optional[str] = None
    metrics_val_season: Optional[str] = None
    test_error: Optional[str] = None
    val_error: Optional[str] = None
    metrics_error: Optional[str] = None


class MetricsResponse(BaseModel):
    payload: Dict[str, Any]


class ModelManifestResponse(BaseModel):
    payload: Dict[str, Any]


class ActiveArtifactsResponse(BaseModel):
    payload: Dict[str, Any]


class PredictionListResponse(BaseModel):
    split: str
    total: int
    count: int
    limit: int
    offset: int
    sort_by: str
    sort_order: str
    items: List[Dict[str, Any]]


class ShortlistResponse(BaseModel):
    split: str
    total_candidates: int
    count: int
    diagnostics: Optional[Dict[str, Any]] = None
    items: List[Dict[str, Any]]


class ScoutTargetsResponse(BaseModel):
    split: str
    total_candidates: int
    count: int
    diagnostics: Optional[Dict[str, Any]] = None
    items: List[Dict[str, Any]]


class PlayerPredictionResponse(BaseModel):
    split: str
    item: Dict[str, Any]


class PlayerReportResponse(BaseModel):
    split: str
    report: Dict[str, Any]


class PlayerHistoryStrengthResponse(BaseModel):
    split: str
    breakdown: Dict[str, Any]


class WatchlistAddRequest(BaseModel):
    player_id: str
    split: Literal["test", "val"] = "test"
    season: Optional[str] = None
    tag: Optional[str] = None
    notes: Optional[str] = None
    source: Optional[str] = "manual"


class WatchlistListResponse(BaseModel):
    path: str
    total: int
    count: int
    limit: int
    offset: int
    items: List[Dict[str, Any]]


class WatchlistItemResponse(BaseModel):
    item: Dict[str, Any]


class WatchlistDeleteResponse(BaseModel):
    path: str
    watch_id: str
    deleted: bool


@router.get("/health", response_model=HealthResponse, summary="Check market-value artifact health")
def market_value_health() -> HealthResponse:
    """Return backend readiness and artifact status for market-value endpoints."""
    return HealthResponse(**health_payload())


@router.get("/metrics", response_model=MetricsResponse, summary="Get training/evaluation metrics")
def market_value_metrics() -> MetricsResponse:
    """Return JSON metrics payload produced by train_market_value_full."""
    try:
        payload = get_metrics()
    except ArtifactNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to load metrics: {exc}") from exc
    return MetricsResponse(payload=payload)


@router.get("/model-manifest", response_model=ModelManifestResponse, summary="Get model registry manifest")
def market_value_model_manifest() -> ModelManifestResponse:
    """Return model artifact manifest (file-backed or derived from current artifacts)."""
    try:
        payload = get_model_manifest()
    except ArtifactNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to load model manifest: {exc}") from exc
    return ModelManifestResponse(payload=payload)


@router.get(
    "/active-artifacts",
    response_model=ActiveArtifactsResponse,
    summary="Get resolved artifact paths and hashes",
)
def market_value_active_artifacts() -> ActiveArtifactsResponse:
    """Return currently active predictions/metrics artifact paths and SHA256 hashes."""
    try:
        payload = get_active_artifacts()
    except ArtifactNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to load active artifacts: {exc}") from exc
    return ActiveArtifactsResponse(payload=payload)


@router.get("/predictions", response_model=PredictionListResponse, summary="Query prediction rows")
def market_value_predictions(
    split: Literal["test", "val"] = Query("test"),
    season: Optional[str] = Query(None),
    league: Optional[str] = Query(None),
    club: Optional[str] = Query(None),
    position: Optional[str] = Query(None, description="GK/DF/MF/FW"),
    min_minutes: Optional[float] = Query(None, ge=0),
    max_age: Optional[float] = Query(None, ge=0),
    undervalued_only: bool = Query(False),
    min_confidence: Optional[float] = Query(None),
    min_value_gap_eur: Optional[float] = Query(None),
    sort_by: str = Query("value_gap_capped_eur"),
    sort_order: Literal["asc", "desc"] = Query("desc"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    columns: Optional[str] = Query(
        None,
        description="Optional comma-separated columns to return.",
    ),
) -> PredictionListResponse:
    """Filter/sort/paginate model predictions for API clients."""
    try:
        selected_cols = [c.strip() for c in columns.split(",")] if columns else None
        payload = query_predictions(
            split=split,
            season=season,
            league=league,
            club=club,
            position=position,
            min_minutes=min_minutes,
            max_age=max_age,
            undervalued_only=undervalued_only,
            min_confidence=min_confidence,
            min_value_gap_eur=min_value_gap_eur,
            sort_by=sort_by,
            sort_order=sort_order,
            limit=limit,
            offset=offset,
            columns=selected_cols,
        )
    except ArtifactNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to query predictions: {exc}") from exc
    return PredictionListResponse(**payload)


@router.get("/shortlist", response_model=ShortlistResponse, summary="Get confidence-aware scouting shortlist")
def market_value_shortlist(
    split: Literal["test", "val"] = Query("test"),
    top_n: int = Query(100, ge=1, le=1000),
    min_minutes: float = Query(900, ge=0),
    max_age: int = Query(25, description="Set -1 to disable."),
    positions: Optional[str] = Query(None, description="Comma-separated positions, e.g. FW,MF"),
) -> ShortlistResponse:
    """Return ranked shortlist of conservatively undervalued players."""
    try:
        pos_list = [p.strip().upper() for p in positions.split(",") if p.strip()] if positions else None
        payload = query_shortlist(
            split=split,
            top_n=top_n,
            min_minutes=min_minutes,
            max_age=None if max_age < 0 else float(max_age),
            positions=pos_list,
        )
    except ArtifactNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to build shortlist: {exc}") from exc
    return ShortlistResponse(**payload)


@router.get(
    "/scout-targets",
    response_model=ScoutTargetsResponse,
    summary="Get top undervalued scouting targets (supports non-Big5 filter)",
)
def market_value_scout_targets(
    split: Literal["test", "val"] = Query("test"),
    top_n: int = Query(100, ge=1, le=1000),
    min_minutes: float = Query(900, ge=0),
    max_age: int = Query(23, description="Set -1 to disable."),
    min_confidence: float = Query(0.50, ge=0.0),
    min_value_gap_eur: float = Query(1_000_000.0, ge=0.0),
    min_expected_value_eur: Optional[float] = Query(None, ge=0.0),
    max_expected_value_eur: Optional[float] = Query(None, ge=0.0),
    non_big5_only: bool = Query(True),
    positions: Optional[str] = Query(None, description="Comma-separated positions, e.g. FW,MF"),
    include_leagues: Optional[str] = Query(
        None,
        description="Comma-separated league names to force-include.",
    ),
    exclude_leagues: Optional[str] = Query(
        None,
        description="Comma-separated league names to exclude.",
    ),
) -> ScoutTargetsResponse:
    """Return production-ready scouting targets ranked by confidence-weighted undervaluation."""
    try:
        pos_list = [p.strip().upper() for p in positions.split(",") if p.strip()] if positions else None
        include_list = [l.strip() for l in include_leagues.split(",") if l.strip()] if include_leagues else None
        exclude_list = [l.strip() for l in exclude_leagues.split(",") if l.strip()] if exclude_leagues else None
        payload = query_scout_targets(
            split=split,
            top_n=top_n,
            min_minutes=min_minutes,
            max_age=None if max_age < 0 else float(max_age),
            min_confidence=min_confidence,
            min_value_gap_eur=min_value_gap_eur,
            positions=pos_list,
            non_big5_only=non_big5_only,
            include_leagues=include_list,
            exclude_leagues=exclude_list,
            min_expected_value_eur=min_expected_value_eur,
            max_expected_value_eur=max_expected_value_eur,
        )
    except ArtifactNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to build scout targets: {exc}") from exc
    return ScoutTargetsResponse(**payload)


@router.get(
    "/watchlist",
    response_model=WatchlistListResponse,
    summary="List saved watchlist entries",
)
def market_value_watchlist(
    split: Optional[Literal["test", "val"]] = Query(None),
    tag: Optional[str] = Query(None),
    player_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> WatchlistListResponse:
    """Return persisted watchlist entries with optional filtering."""
    try:
        payload = list_watchlist(
            split=split,
            tag=tag,
            player_id=player_id,
            limit=limit,
            offset=offset,
        )
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to load watchlist: {exc}") from exc
    return WatchlistListResponse(**payload)


@router.post(
    "/watchlist/items",
    response_model=WatchlistItemResponse,
    summary="Add a player to persisted watchlist",
)
def market_value_watchlist_add(payload: WatchlistAddRequest) -> WatchlistItemResponse:
    """Persist a watchlist entry with player snapshot and summary text."""
    try:
        item = add_watchlist_item(
            player_id=payload.player_id,
            split=payload.split,
            season=payload.season,
            tag=payload.tag,
            notes=payload.notes,
            source=payload.source,
        )
    except ArtifactNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to add watchlist item: {exc}") from exc
    return WatchlistItemResponse(item=item)


@router.delete(
    "/watchlist/items/{watch_id}",
    response_model=WatchlistDeleteResponse,
    summary="Delete watchlist entry by id",
)
def market_value_watchlist_delete(watch_id: str) -> WatchlistDeleteResponse:
    """Delete one watchlist item by watch_id."""
    try:
        payload = delete_watchlist_item(watch_id=watch_id)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to delete watchlist item: {exc}") from exc
    return WatchlistDeleteResponse(**payload)


@router.get("/player/{player_id}", response_model=PlayerPredictionResponse, summary="Get a player prediction")
def market_value_player(
    player_id: str,
    split: Literal["test", "val"] = Query("test"),
    season: Optional[str] = Query(None),
) -> PlayerPredictionResponse:
    """Return one prediction row for a specific player and optional season."""
    try:
        item = get_player_prediction(player_id=player_id, split=split, season=season)
    except ArtifactNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to load player prediction: {exc}") from exc
    return PlayerPredictionResponse(split=split, item=item)


@router.get(
    "/player/{player_id}/report",
    response_model=PlayerReportResponse,
    summary="Get scouting report for a player (strengths, risks, development levers)",
)
def market_value_player_report(
    player_id: str,
    split: Literal["test", "val"] = Query("test"),
    season: Optional[str] = Query(None),
    top_metrics: int = Query(5, ge=1, le=10),
) -> PlayerReportResponse:
    """Return a player scouting profile built from prediction artifacts and cohort context."""
    try:
        report = get_player_report(
            player_id=player_id,
            split=split,
            season=season,
            top_metrics=top_metrics,
        )
    except ArtifactNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to build player report: {exc}") from exc
    return PlayerReportResponse(split=split, report=report)


@router.get(
    "/player/{player_id}/history-strength",
    response_model=PlayerHistoryStrengthResponse,
    summary="Get history-strength breakdown for a player",
)
def market_value_player_history_strength(
    player_id: str,
    split: Literal["test", "val"] = Query("test"),
    season: Optional[str] = Query(None),
) -> PlayerHistoryStrengthResponse:
    """Return weighted history-strength components for one player."""
    try:
        breakdown = get_player_history_strength(
            player_id=player_id,
            split=split,
            season=season,
        )
    except ArtifactNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to build history strength breakdown: {exc}") from exc
    return PlayerHistoryStrengthResponse(split=split, breakdown=breakdown)


__all__ = ["router"]
