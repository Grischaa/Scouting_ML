"""FastAPI routes for market-value model artifacts and retrieval."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException, Query, Request, Response
from starlette.status import HTTP_503_SERVICE_UNAVAILABLE
from pydantic import BaseModel, Field

from scouting_ml.services.market_value_service import (
    ArtifactNotFoundError,
    add_watchlist_item,
    build_player_memo_pdf,
    delete_watchlist_item,
    get_active_artifacts,
    get_benchmark_report,
    get_model_manifest,
    get_metrics,
    get_operator_health,
    get_player_advanced_profile,
    get_player_profile,
    get_player_history_strength,
    get_player_report,
    get_player_similar,
    get_player_prediction,
    get_player_trajectory_view,
    get_system_fit_templates,
    get_ui_bootstrap,
    health_payload,
    list_player_decisions,
    list_watchlist,
    query_player_reports,
    query_predictions,
    query_scout_targets,
    query_shortlist,
    query_system_fit,
    save_scout_decision,
)
from scouting_ml.team.service import (
    get_latest_team_decision,
    maybe_apply_preferences_to_rows,
    session_cookie_name,
)

router = APIRouter(prefix="/market-value", tags=["market_value"])


def _team_session_token(request: Request) -> str | None:
    return request.cookies.get(session_cookie_name())


def _team_workspace_id(request: Request) -> str | None:
    return request.headers.get("X-ScoutML-Workspace") or request.query_params.get("workspace_id")


def _apply_team_preferences(
    request: Request,
    rows: List[Dict[str, Any]],
    *,
    apply_preferences: bool,
    preference_profile_id: Optional[str],
    mode: str,
    active_lane: Optional[str] = None,
) -> List[Dict[str, Any]]:
    return maybe_apply_preferences_to_rows(
        rows,
        session_token=_team_session_token(request),
        workspace_id=_team_workspace_id(request),
        preference_profile_id=preference_profile_id,
        apply_preferences=apply_preferences,
        mode=mode,
        active_lane=active_lane,
    )


def _maybe_override_latest_decision(
    request: Request,
    payload: Dict[str, Any],
    *,
    player_id: str,
    split: str,
    season: Optional[str],
) -> Dict[str, Any]:
    latest = get_latest_team_decision(
        session_token=_team_session_token(request),
        workspace_id=_team_workspace_id(request),
        player_id=player_id,
        split=split,
        season=season,
    )
    if latest is not None:
        payload["latest_decision"] = latest
    return payload


class HealthResponse(BaseModel):
    status: str
    artifacts: Dict[str, Any]
    strict_artifacts: Optional[bool] = None
    strict_artifacts_error: Optional[str] = None
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


class BenchmarkReportResponse(BaseModel):
    payload: Dict[str, Any]


class OperatorHealthResponse(BaseModel):
    payload: Dict[str, Any]


class UIBootstrapResponse(BaseModel):
    split: str
    seasons: List[str]
    leagues: List[str]
    coverage_rows: List[Dict[str, Any]]
    generated_at_utc: str


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


class SystemFitTemplatesResponse(BaseModel):
    default_template_key: Optional[str] = None
    supported_active_lanes: List[str]
    supported_trust_scopes: List[str]
    templates: List[Dict[str, Any]]


class SystemFitQueryFilters(BaseModel):
    season: Optional[str] = None
    include_leagues: Optional[List[str]] = None
    exclude_leagues: Optional[List[str]] = None
    min_age: Optional[float] = None
    max_age: Optional[float] = None
    min_minutes: Optional[float] = None
    max_market_value_eur: Optional[float] = None
    max_contract_years_left: Optional[float] = None
    min_confidence: Optional[float] = None
    non_big5_only: bool = False
    budget_eur: Optional[float] = None


class SystemFitQueryRequest(BaseModel):
    template_key: str
    split: Literal["test", "val"] = "test"
    active_lane: Literal["valuation", "future_shortlist"] = "valuation"
    top_n_per_slot: int = 10
    slot_role_overrides: Optional[Dict[str, str]] = None
    filters: Optional[SystemFitQueryFilters] = None
    trust_scope: Literal["trusted_only", "trusted_and_watch", "all"] = "trusted_and_watch"


class SystemFitQueryResponse(BaseModel):
    system_profile: Dict[str, Any]
    split: str
    active_lane: str
    lane_posture: Dict[str, Any]
    trust_scope: str
    filters_applied: Dict[str, Any]
    slots: List[Dict[str, Any]]


class PlayerPredictionResponse(BaseModel):
    split: str
    item: Dict[str, Any]


class PlayerReportResponse(BaseModel):
    split: str
    report: Dict[str, Any]


class PlayerAdvancedProfileResponse(BaseModel):
    split: str
    profile: Dict[str, Any]


class PlayerProfileResponse(BaseModel):
    split: str
    profile: Dict[str, Any]


class PlayerReportsResponse(BaseModel):
    split: str
    total: int
    count: int
    limit: int
    offset: int
    sort_by: str
    sort_order: str
    items: List[Dict[str, Any]]


class PlayerHistoryStrengthResponse(BaseModel):
    split: str
    breakdown: Dict[str, Any]


class PlayerSimilarResponse(BaseModel):
    player_id: str
    position_group: Optional[str] = None
    feature_count_used: Optional[int] = None
    feature_columns_used: List[str] = Field(default_factory=list)
    comparisons: List[Dict[str, Any]]


class PlayerTrajectoryResponse(BaseModel):
    player_id: str
    trajectory_label: str
    slope_pct: float
    r2: float
    projected_next_value: Optional[float] = None
    peak_season: Optional[str] = None
    seasons: List[Dict[str, Any]]


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


class ScoutDecisionRequest(BaseModel):
    player_id: str
    split: Literal["test", "val"] = "test"
    season: Optional[str] = None
    action: Literal["shortlist", "watch_live", "request_report", "pass"]
    reason_tags: List[str] = Field(default_factory=list)
    note: Optional[str] = None
    actor: Optional[str] = "local"
    source_surface: Optional[str] = "detail"
    ranking_context: Optional[Dict[str, Any]] = None


class ScoutDecisionResponse(BaseModel):
    decision: Dict[str, Any]
    latest_decision: Optional[Dict[str, Any]] = None
    watchlist_item: Optional[Dict[str, Any]] = None


class PlayerDecisionsResponse(BaseModel):
    player_id: str
    latest_decision: Optional[Dict[str, Any]] = None
    events: List[Dict[str, Any]]


@router.get("/health", response_model=HealthResponse, summary="Check market-value artifact health")
async def market_value_health(response: Response) -> HealthResponse:
    """Return backend readiness and artifact status for market-value endpoints."""
    payload = health_payload()
    if payload.get("status") != "ok":
        response.status_code = HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(**payload)


@router.get("/metrics", response_model=MetricsResponse, summary="Get training/evaluation metrics")
async def market_value_metrics() -> MetricsResponse:
    """Return JSON metrics payload produced by train_market_value_full."""
    try:
        payload = get_metrics()
    except ArtifactNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to load metrics: {exc}") from exc
    return MetricsResponse(payload=payload)


@router.get("/model-manifest", response_model=ModelManifestResponse, summary="Get model registry manifest")
async def market_value_model_manifest() -> ModelManifestResponse:
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
async def market_value_active_artifacts() -> ActiveArtifactsResponse:
    """Return currently active predictions/metrics artifact paths and SHA256 hashes."""
    try:
        payload = get_active_artifacts()
    except ArtifactNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to load active artifacts: {exc}") from exc
    return ActiveArtifactsResponse(payload=payload)


@router.get(
    "/benchmarks",
    response_model=BenchmarkReportResponse,
    summary="Get multi-league benchmark summary",
)
async def market_value_benchmarks() -> BenchmarkReportResponse:
    """Return aggregated benchmark/reporting payload for holdouts, onboarding, and ablations."""
    try:
        payload = get_benchmark_report()
    except ArtifactNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to load benchmark report: {exc}") from exc
    return BenchmarkReportResponse(payload=payload)


@router.get(
    "/operator-health",
    response_model=OperatorHealthResponse,
    summary="Get operator reliability dashboard payload",
)
async def market_value_operator_health() -> OperatorHealthResponse:
    """Return aggregated operator health across ingestion, artifact lanes, holdouts, and live overlay posture."""
    try:
        payload = get_operator_health()
    except ArtifactNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to load operator health: {exc}") from exc
    return OperatorHealthResponse(payload=payload)


@router.get(
    "/ui-bootstrap",
    response_model=UIBootstrapResponse,
    summary="Get lightweight split-level UI bootstrap metadata",
)
async def market_value_ui_bootstrap(
    split: Literal["test", "val"] = Query("test"),
) -> UIBootstrapResponse:
    """Return seasons, leagues, and coverage rows for fast board bootstrapping."""
    try:
        payload = get_ui_bootstrap(split=split)
    except ArtifactNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to build UI bootstrap payload: {exc}") from exc
    return UIBootstrapResponse(**payload)


@router.get(
    "/system-fit/templates",
    response_model=SystemFitTemplatesResponse,
    summary="List backend-owned system-fit templates",
)
async def market_value_system_fit_templates() -> SystemFitTemplatesResponse:
    """Return the static system-fit templates supported by the backend."""
    try:
        payload = get_system_fit_templates()
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to load system-fit templates: {exc}") from exc
    return SystemFitTemplatesResponse(**payload)


@router.post(
    "/system-fit/query",
    response_model=SystemFitQueryResponse,
    summary="Rank players by slot-level system fit",
)
async def market_value_system_fit_query(
    request: Request,
    payload: SystemFitQueryRequest,
    preference_profile_id: Optional[str] = Query(None),
    apply_preferences: bool = Query(False),
) -> SystemFitQueryResponse:
    """Return independent ranked candidate lists for each slot in a named system template."""
    try:
        result = query_system_fit(
            split=payload.split,
            template_key=payload.template_key,
            active_lane=payload.active_lane,
            top_n_per_slot=payload.top_n_per_slot,
            slot_role_overrides=payload.slot_role_overrides,
            filters=payload.filters.model_dump() if payload.filters else None,
            trust_scope=payload.trust_scope,
        )
    except ArtifactNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to query system fit: {exc}") from exc
    for slot in result.get("slots") or []:
        slot["items"] = _apply_team_preferences(
            request,
            slot.get("items") or [],
            apply_preferences=apply_preferences,
            preference_profile_id=preference_profile_id,
            mode="system_fit",
            active_lane=payload.active_lane,
        )
    return SystemFitQueryResponse(**result)


@router.get("/predictions", response_model=PredictionListResponse, summary="Query prediction rows")
async def market_value_predictions(
    request: Request,
    split: Literal["test", "val"] = Query("test"),
    season: Optional[str] = Query(None),
    league: Optional[str] = Query(None),
    club: Optional[str] = Query(None),
    position: Optional[str] = Query(None, description="GK/DF/MF/FW"),
    role_keys: Optional[str] = Query(None, description="Comma-separated role keys, e.g. CB,DM,W,ST"),
    min_minutes: Optional[float] = Query(None, ge=0),
    min_age: Optional[float] = Query(None, ge=0),
    max_age: Optional[float] = Query(None, ge=0),
    max_market_value_eur: Optional[float] = Query(None, ge=0),
    max_contract_years_left: Optional[float] = Query(None, ge=0),
    non_big5_only: bool = Query(False),
    undervalued_only: bool = Query(False),
    min_confidence: Optional[float] = Query(None),
    min_value_gap_eur: Optional[float] = Query(None),
    sort_by: str = Query("value_gap_capped_eur"),
    sort_order: Literal["asc", "desc"] = Query("desc"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    preference_profile_id: Optional[str] = Query(None),
    apply_preferences: bool = Query(False),
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
            role_keys=[r.strip().upper() for r in role_keys.split(",") if r.strip()] if role_keys else None,
            min_minutes=min_minutes,
            min_age=min_age,
            max_age=max_age,
            max_market_value_eur=max_market_value_eur,
            max_contract_years_left=max_contract_years_left,
            non_big5_only=non_big5_only,
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
    payload["items"] = _apply_team_preferences(
        request,
        payload.get("items") or [],
        apply_preferences=apply_preferences,
        preference_profile_id=preference_profile_id,
        mode="predictions",
    )
    return PredictionListResponse(**payload)


@router.get("/shortlist", response_model=ShortlistResponse, summary="Get confidence-aware scouting shortlist")
async def market_value_shortlist(
    request: Request,
    split: Literal["test", "val"] = Query("test"),
    top_n: int = Query(100, ge=1, le=1000),
    min_minutes: float = Query(900, ge=0),
    min_age: Optional[float] = Query(None, ge=0),
    max_age: int = Query(25, description="Set -1 to disable."),
    positions: Optional[str] = Query(None, description="Comma-separated positions, e.g. FW,MF"),
    role_keys: Optional[str] = Query(None, description="Comma-separated role keys, e.g. CB,DM,W,ST"),
    non_big5_only: bool = Query(False),
    max_market_value_eur: Optional[float] = Query(None, ge=0),
    max_contract_years_left: Optional[float] = Query(None, ge=0),
    preference_profile_id: Optional[str] = Query(None),
    apply_preferences: bool = Query(False),
) -> ShortlistResponse:
    """Return ranked shortlist of conservatively undervalued players."""
    try:
        pos_list = [p.strip().upper() for p in positions.split(",") if p.strip()] if positions else None
        payload = query_shortlist(
            split=split,
            top_n=top_n,
            min_minutes=min_minutes,
            min_age=min_age,
            max_age=None if max_age < 0 else float(max_age),
            positions=pos_list,
            role_keys=[r.strip().upper() for r in role_keys.split(",") if r.strip()] if role_keys else None,
            non_big5_only=non_big5_only,
            max_market_value_eur=max_market_value_eur,
            max_contract_years_left=max_contract_years_left,
        )
    except ArtifactNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to build shortlist: {exc}") from exc
    payload["items"] = _apply_team_preferences(
        request,
        payload.get("items") or [],
        apply_preferences=apply_preferences,
        preference_profile_id=preference_profile_id,
        mode="shortlist",
    )
    return ShortlistResponse(**payload)


@router.get(
    "/scout-targets",
    response_model=ScoutTargetsResponse,
    summary="Get top undervalued scouting targets (supports non-Big5 filter)",
)
async def market_value_scout_targets(
    request: Request,
    split: Literal["test", "val"] = Query("test"),
    top_n: int = Query(100, ge=1, le=1000),
    min_minutes: float = Query(900, ge=0),
    min_age: Optional[float] = Query(None, ge=0),
    max_age: int = Query(23, description="Set -1 to disable."),
    min_confidence: float = Query(0.50, ge=0.0),
    min_value_gap_eur: float = Query(1_000_000.0, ge=0.0),
    min_expected_value_eur: Optional[float] = Query(None, ge=0.0),
    max_expected_value_eur: Optional[float] = Query(None, ge=0.0),
    max_market_value_eur: Optional[float] = Query(None, ge=0.0),
    max_contract_years_left: Optional[float] = Query(None, ge=0.0),
    non_big5_only: bool = Query(True),
    positions: Optional[str] = Query(None, description="Comma-separated positions, e.g. FW,MF"),
    role_keys: Optional[str] = Query(None, description="Comma-separated role keys, e.g. CB,DM,W,ST"),
    include_leagues: Optional[str] = Query(
        None,
        description="Comma-separated league names to force-include.",
    ),
    exclude_leagues: Optional[str] = Query(
        None,
        description="Comma-separated league names to exclude.",
    ),
    preference_profile_id: Optional[str] = Query(None),
    apply_preferences: bool = Query(False),
) -> ScoutTargetsResponse:
    """Return production-ready scouting targets ranked by confidence-weighted undervaluation."""
    try:
        pos_list = [p.strip().upper() for p in positions.split(",") if p.strip()] if positions else None
        include_list = (
            [league_name.strip() for league_name in include_leagues.split(",") if league_name.strip()]
            if include_leagues
            else None
        )
        exclude_list = (
            [league_name.strip() for league_name in exclude_leagues.split(",") if league_name.strip()]
            if exclude_leagues
            else None
        )
        payload = query_scout_targets(
            split=split,
            top_n=top_n,
            min_minutes=min_minutes,
            min_age=min_age,
            max_age=None if max_age < 0 else float(max_age),
            min_confidence=min_confidence,
            min_value_gap_eur=min_value_gap_eur,
            positions=pos_list,
            role_keys=[r.strip().upper() for r in role_keys.split(",") if r.strip()] if role_keys else None,
            non_big5_only=non_big5_only,
            include_leagues=include_list,
            exclude_leagues=exclude_list,
            min_expected_value_eur=min_expected_value_eur,
            max_expected_value_eur=max_expected_value_eur,
            max_market_value_eur=max_market_value_eur,
            max_contract_years_left=max_contract_years_left,
        )
    except ArtifactNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to build scout targets: {exc}") from exc
    payload["items"] = _apply_team_preferences(
        request,
        payload.get("items") or [],
        apply_preferences=apply_preferences,
        preference_profile_id=preference_profile_id,
        mode="scout_targets",
    )
    return ScoutTargetsResponse(**payload)


@router.get(
    "/watchlist",
    response_model=WatchlistListResponse,
    summary="List saved watchlist entries",
)
async def market_value_watchlist(
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
async def market_value_watchlist_add(payload: WatchlistAddRequest) -> WatchlistItemResponse:
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
async def market_value_watchlist_delete(watch_id: str) -> WatchlistDeleteResponse:
    """Delete one watchlist item by watch_id."""
    try:
        payload = delete_watchlist_item(watch_id=watch_id)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to delete watchlist item: {exc}") from exc
    return WatchlistDeleteResponse(**payload)


@router.post(
    "/decisions",
    response_model=ScoutDecisionResponse,
    summary="Save a scout decision event for one player",
)
async def market_value_scout_decision(payload: ScoutDecisionRequest) -> ScoutDecisionResponse:
    """Persist one scout decision and sync positive actions into the watchlist."""
    try:
        response = save_scout_decision(
            player_id=payload.player_id,
            split=payload.split,
            season=payload.season,
            action=payload.action,
            reason_tags=payload.reason_tags,
            note=payload.note,
            actor=payload.actor,
            source_surface=payload.source_surface,
            ranking_context=payload.ranking_context,
        )
    except ArtifactNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to save scout decision: {exc}") from exc
    return ScoutDecisionResponse(**response)


@router.get(
    "/player-reports",
    response_model=PlayerReportsResponse,
    summary="Get precise scouting breakdowns for many players",
)
async def market_value_player_reports(
    split: Literal["test", "val"] = Query("test"),
    season: Optional[str] = Query(None),
    league: Optional[str] = Query(None),
    club: Optional[str] = Query(None),
    position: Optional[str] = Query(None, description="GK/DF/MF/FW"),
    min_minutes: Optional[float] = Query(None, ge=0),
    max_age: Optional[float] = Query(None, ge=0),
    player_ids: Optional[str] = Query(None, description="Optional comma-separated player_ids filter."),
    top_metrics: int = Query(5, ge=1, le=10),
    include_history: bool = Query(True),
    sort_by: str = Query("undervaluation_score"),
    sort_order: Literal["asc", "desc"] = Query("desc"),
    limit: int = Query(200, ge=1, le=5000),
    offset: int = Query(0, ge=0),
) -> PlayerReportsResponse:
    """Return detailed report payloads (and optional history-strength) for a player page."""
    try:
        selected_ids = [tok.strip() for tok in player_ids.split(",") if tok.strip()] if player_ids else None
        payload = query_player_reports(
            split=split,
            season=season,
            league=league,
            club=club,
            position=position,
            min_minutes=min_minutes,
            max_age=max_age,
            player_ids=selected_ids,
            top_metrics=top_metrics,
            include_history=include_history,
            sort_by=sort_by,
            sort_order=sort_order,
            limit=limit,
            offset=offset,
        )
    except ArtifactNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to build player reports: {exc}") from exc
    return PlayerReportsResponse(**payload)


@router.get("/player/{player_id}", response_model=PlayerPredictionResponse, summary="Get a player prediction")
async def market_value_player(
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
    "/player/{player_id}/profile",
    response_model=PlayerProfileResponse,
    summary="Get unified player profile for UI detail views",
)
async def market_value_player_profile(
    request: Request,
    player_id: str,
    split: Literal["test", "val"] = Query("test"),
    season: Optional[str] = Query(None),
    top_metrics: int = Query(6, ge=1, le=10),
    similar_top_k: int = Query(5, ge=1, le=10),
) -> PlayerProfileResponse:
    """Return a unified player profile payload with grouped stats, history, and similar players."""
    try:
        profile = get_player_profile(
            player_id=player_id,
            split=split,
            season=season,
            top_metrics=top_metrics,
            similar_top_k=similar_top_k,
        )
    except ArtifactNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to build player profile: {exc}") from exc
    profile = _maybe_override_latest_decision(
        request,
        profile,
        player_id=player_id,
        split=split,
        season=season or profile.get("season"),
    )
    return PlayerProfileResponse(split=split, profile=profile)


@router.get(
    "/player/{player_id}/advanced-profile",
    response_model=PlayerAdvancedProfileResponse,
    summary="Get advanced player profile (type, radar, formation fit)",
)
async def market_value_player_advanced_profile(
    player_id: str,
    split: Literal["test", "val"] = Query("test"),
    season: Optional[str] = Query(None),
    top_metrics: int = Query(6, ge=1, le=10),
) -> PlayerAdvancedProfileResponse:
    """Return tactical/archetype profile with radar payload and formation recommendations."""
    try:
        profile = get_player_advanced_profile(
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
        raise HTTPException(status_code=500, detail=f"Failed to build advanced profile: {exc}") from exc
    return PlayerAdvancedProfileResponse(split=split, profile=profile)


@router.get(
    "/player/{player_id}/report",
    response_model=PlayerReportResponse,
    summary="Get scouting report for a player (strengths, risks, development levers)",
)
async def market_value_player_report(
    request: Request,
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
    report = _maybe_override_latest_decision(
        request,
        report,
        player_id=player_id,
        split=split,
        season=season or report.get("season"),
    )
    return PlayerReportResponse(split=split, report=report)


@router.get(
    "/player/{player_id}/decisions",
    response_model=PlayerDecisionsResponse,
    summary="Get saved scout decisions for a player",
)
async def market_value_player_decisions(
    player_id: str,
    split: Literal["test", "val"] = Query("test"),
    season: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
) -> PlayerDecisionsResponse:
    """Return the latest scout decision plus ordered decision history for one player."""
    try:
        payload = list_player_decisions(
            player_id=player_id,
            split=split,
            season=season,
            limit=limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to load player decisions: {exc}") from exc
    return PlayerDecisionsResponse(**payload)


@router.get(
    "/player/{player_id}/similar",
    response_model=PlayerSimilarResponse,
    summary="Get nearest comparable players",
)
async def market_value_player_similar(
    player_id: str,
    n: int = Query(5, ge=1, le=20),
    same_position: bool = Query(True),
    exclude_big5: bool = Query(False),
    split: Literal["test", "val"] = Query("test"),
    season: Optional[str] = Query(None),
) -> PlayerSimilarResponse:
    """Return statistically similar players using normalized position-aware vectors."""
    try:
        payload = get_player_similar(
            player_id=player_id,
            split=split,
            season=season,
            n=n,
            same_position=same_position,
            exclude_big5=exclude_big5,
        )
    except ArtifactNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to load similar players: {exc}") from exc
    return PlayerSimilarResponse(**payload)


@router.get(
    "/player/{player_id}/trajectory",
    response_model=PlayerTrajectoryResponse,
    summary="Get player trajectory across seasons",
)
async def market_value_player_trajectory(
    player_id: str,
    split: Literal["test", "val"] = Query("test"),
    season: Optional[str] = Query(None),
) -> PlayerTrajectoryResponse:
    """Return a compact multi-season trajectory summary for one player."""
    try:
        payload = get_player_trajectory_view(player_id=player_id, split=split, season=season)
    except ArtifactNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to build trajectory: {exc}") from exc
    return PlayerTrajectoryResponse(**payload)


@router.get(
    "/player/{player_id}/memo.pdf",
    summary="Export a one-page PDF player memo",
)
async def market_value_player_memo_pdf(
    player_id: str,
    split: Literal["test", "val"] = Query("test"),
    season: Optional[str] = Query(None),
    include_trajectory: bool = Query(True),
    include_similar: bool = Query(True),
) -> Response:
    """Generate and stream a professional scouting memo PDF."""
    try:
        payload = build_player_memo_pdf(
            player_id=player_id,
            split=split,
            season=season,
            include_trajectory=include_trajectory,
            include_similar=include_similar,
        )
    except ArtifactNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to generate player memo: {exc}") from exc
    filename = str(payload.get("filename") or "player_scout_memo.pdf")
    content = bytes(payload.get("content") or b"")
    return Response(
        content=content,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get(
    "/player/{player_id}/history-strength",
    response_model=PlayerHistoryStrengthResponse,
    summary="Get history-strength breakdown for a player",
)
async def market_value_player_history_strength(
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
