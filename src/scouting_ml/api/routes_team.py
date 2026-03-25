"""FastAPI routes for ScoutML Team Edition shared workspaces."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

from scouting_ml.team.service import (
    TeamAuthError,
    TeamModeDisabledError,
    accept_workspace_invite,
    add_player_to_compare_list,
    add_team_player_comment,
    add_team_watchlist_item,
    bootstrap_admin,
    create_team_assignment,
    create_team_compare_list,
    create_workspace_for_current_user,
    create_workspace_invite,
    delete_team_compare_list,
    delete_team_watchlist_item,
    get_auth_me_payload,
    get_team_preferences_me,
    get_workspace_state,
    list_team_activity,
    list_team_assignments,
    list_team_compare_lists,
    list_team_player_comments,
    list_team_player_decisions,
    list_team_watchlist,
    login_user,
    logout_user,
    put_team_preferences_me,
    remove_player_from_compare_list,
    save_team_decision,
    session_cookie_name,
    session_cookie_settings,
    team_mode_enabled,
    update_team_assignment,
    update_team_compare_list,
    update_team_watchlist_item,
)


router = APIRouter(tags=["team"])


class LoginRequest(BaseModel):
    email: str
    password: str
    workspace_id: Optional[str] = None


class BootstrapAdminRequest(BaseModel):
    email: str
    password: str
    full_name: Optional[str] = None
    workspace_name: Optional[str] = None


class WorkspaceCreateRequest(BaseModel):
    name: str


class WorkspaceInviteRequest(BaseModel):
    email: Optional[str] = None
    role: str = "scout"


class InviteAcceptRequest(BaseModel):
    email: str
    password: str
    full_name: Optional[str] = None


class TeamWatchlistAddRequest(BaseModel):
    player_id: str
    split: str = "test"
    season: Optional[str] = None
    tag: Optional[str] = None
    notes: Optional[str] = None
    source: Optional[str] = "manual"


class TeamWatchlistPatchRequest(BaseModel):
    tag: Optional[str] = None
    notes: Optional[str] = None


class TeamDecisionRequest(BaseModel):
    player_id: str
    split: str = "test"
    season: Optional[str] = None
    action: str
    reason_tags: List[str] = Field(default_factory=list)
    note: Optional[str] = None
    actor: Optional[str] = None
    source_surface: Optional[str] = "detail"
    ranking_context: Optional[Dict[str, Any]] = None


class TeamAssignmentRequest(BaseModel):
    player_id: str
    split: str = "test"
    season: Optional[str] = None
    assignee_user_id: Optional[str] = None
    status: str = "to_watch"
    due_date: Optional[str] = None
    note: Optional[str] = None


class TeamAssignmentPatchRequest(BaseModel):
    assignee_user_id: Optional[str] = None
    status: Optional[str] = None
    due_date: Optional[str] = None
    note: Optional[str] = None


class TeamCommentRequest(BaseModel):
    split: str = "test"
    season: Optional[str] = None
    body: str


class CompareListCreateRequest(BaseModel):
    name: str
    notes: Optional[str] = None


class CompareListPatchRequest(BaseModel):
    name: Optional[str] = None
    notes: Optional[str] = None


class CompareListPlayerRequest(BaseModel):
    player_id: str
    split: str = "test"
    season: Optional[str] = None
    pinned: bool = False
    notes: Optional[str] = None


class PreferenceProfileRequest(BaseModel):
    name: Optional[str] = None
    target_age_min: Optional[int] = None
    target_age_max: Optional[int] = None
    budget_posture: Optional[str] = None
    trusted_league_posture: Optional[str] = None
    role_priorities: Dict[str, float] = Field(default_factory=dict)
    system_template_default: Optional[str] = None
    must_have_tags: List[str] = Field(default_factory=list)
    avoid_tags: List[str] = Field(default_factory=list)
    risk_tolerance: Optional[str] = None
    active_lane_preference: Optional[str] = None
    apply_by_default: bool = True


def _session_token(request: Request) -> str | None:
    return request.cookies.get(session_cookie_name())


def _workspace_id(request: Request) -> str | None:
    return request.headers.get("X-ScoutML-Workspace") or request.query_params.get("workspace_id")


def _raise_team_http_error(exc: Exception) -> None:
    if isinstance(exc, TeamModeDisabledError):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if isinstance(exc, TeamAuthError):
        message = str(exc)
        code = 401 if "Authentication" in message or "Session" in message else 403
        raise HTTPException(status_code=code, detail=message) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    raise HTTPException(status_code=500, detail=str(exc)) from exc


def _set_session_cookie(response: Response, session_token: str) -> None:
    response.set_cookie(session_cookie_name(), session_token, **session_cookie_settings())


@router.get("/auth/me")
async def auth_me(request: Request) -> dict[str, Any]:
    """Return current auth and workspace state for the team UI."""
    try:
        return get_auth_me_payload(session_token=_session_token(request), workspace_id=_workspace_id(request))
    except Exception as exc:  # pragma: no cover - defensive
        _raise_team_http_error(exc)


@router.post("/auth/bootstrap-admin")
async def auth_bootstrap_admin(payload: BootstrapAdminRequest, response: Response) -> dict[str, Any]:
    """Create the first admin and workspace, then start a session."""
    try:
        result = bootstrap_admin(
            email=payload.email,
            password=payload.password,
            full_name=payload.full_name,
            workspace_name=payload.workspace_name,
        )
        token = str(result.pop("session_token"))
        _set_session_cookie(response, token)
        return result
    except Exception as exc:  # pragma: no cover - defensive
        _raise_team_http_error(exc)


@router.post("/auth/login")
async def auth_login(payload: LoginRequest, response: Response) -> dict[str, Any]:
    """Authenticate one team user and issue a cookie-backed session."""
    try:
        result = login_user(email=payload.email, password=payload.password, workspace_id=payload.workspace_id)
        token = str(result.pop("session_token"))
        _set_session_cookie(response, token)
        return result
    except Exception as exc:  # pragma: no cover - defensive
        _raise_team_http_error(exc)


@router.post("/auth/logout")
async def auth_logout(request: Request, response: Response) -> dict[str, Any]:
    """Clear the current auth session."""
    try:
        payload = logout_user(session_token=_session_token(request))
        response.delete_cookie(session_cookie_name(), path="/")
        return payload
    except Exception as exc:  # pragma: no cover - defensive
        _raise_team_http_error(exc)


@router.get("/workspaces/me")
async def workspaces_me(request: Request) -> dict[str, Any]:
    """Return the current authenticated workspace state."""
    try:
        return get_workspace_state(session_token=_session_token(request) or "", workspace_id=_workspace_id(request))
    except Exception as exc:  # pragma: no cover - defensive
        _raise_team_http_error(exc)


@router.post("/workspaces")
async def workspaces_create(payload: WorkspaceCreateRequest, request: Request) -> dict[str, Any]:
    """Create a new workspace for the current user."""
    try:
        return create_workspace_for_current_user(
            session_token=_session_token(request) or "",
            workspace_id=_workspace_id(request),
            name=payload.name,
        )
    except Exception as exc:  # pragma: no cover - defensive
        _raise_team_http_error(exc)


@router.post("/workspaces/{workspace_id}/invites")
async def workspaces_invite(workspace_id: str, payload: WorkspaceInviteRequest, request: Request) -> dict[str, Any]:
    """Create one workspace invite token."""
    try:
        return create_workspace_invite(
            session_token=_session_token(request) or "",
            workspace_id=workspace_id,
            email=payload.email,
            role=payload.role,
        )
    except Exception as exc:  # pragma: no cover - defensive
        _raise_team_http_error(exc)


@router.post("/invites/{token}/accept")
async def invites_accept(token: str, payload: InviteAcceptRequest, response: Response) -> dict[str, Any]:
    """Accept an invite token and log the new member in."""
    try:
        result = accept_workspace_invite(
            token=token,
            email=payload.email,
            password=payload.password,
            full_name=payload.full_name,
        )
        session_token = str(result.pop("session_token"))
        _set_session_cookie(response, session_token)
        return result
    except Exception as exc:  # pragma: no cover - defensive
        _raise_team_http_error(exc)


@router.get("/team/watchlist")
async def team_watchlist(
    request: Request,
    split: Optional[str] = Query(None),
    tag: Optional[str] = Query(None),
    player_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """List shared watchlist items for the active workspace."""
    try:
        return list_team_watchlist(
            session_token=_session_token(request) or "",
            workspace_id=_workspace_id(request),
            split=split,
            tag=tag,
            player_id=player_id,
            limit=limit,
            offset=offset,
        )
    except Exception as exc:  # pragma: no cover - defensive
        _raise_team_http_error(exc)


@router.post("/team/watchlist/items")
async def team_watchlist_add(payload: TeamWatchlistAddRequest, request: Request) -> dict[str, Any]:
    """Add or update a shared watchlist row."""
    try:
        return {
            "item": add_team_watchlist_item(
                session_token=_session_token(request) or "",
                workspace_id=_workspace_id(request),
                player_id=payload.player_id,
                split=payload.split,
                season=payload.season,
                tag=payload.tag,
                notes=payload.notes,
                source=payload.source,
            )
        }
    except Exception as exc:  # pragma: no cover - defensive
        _raise_team_http_error(exc)


@router.patch("/team/watchlist/items/{item_id}")
async def team_watchlist_patch(item_id: str, payload: TeamWatchlistPatchRequest, request: Request) -> dict[str, Any]:
    """Patch a shared watchlist row."""
    try:
        return {
            "item": update_team_watchlist_item(
                session_token=_session_token(request) or "",
                workspace_id=_workspace_id(request),
                item_id=item_id,
                tag=payload.tag,
                notes=payload.notes,
            )
        }
    except Exception as exc:  # pragma: no cover - defensive
        _raise_team_http_error(exc)


@router.delete("/team/watchlist/items/{item_id}")
async def team_watchlist_delete(item_id: str, request: Request) -> dict[str, Any]:
    """Delete one shared watchlist row."""
    try:
        return delete_team_watchlist_item(
            session_token=_session_token(request) or "",
            workspace_id=_workspace_id(request),
            item_id=item_id,
        )
    except Exception as exc:  # pragma: no cover - defensive
        _raise_team_http_error(exc)


@router.post("/team/decisions")
async def team_decisions_save(payload: TeamDecisionRequest, request: Request) -> dict[str, Any]:
    """Save one shared scout decision event."""
    try:
        return save_team_decision(
            session_token=_session_token(request) or "",
            workspace_id=_workspace_id(request),
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
    except Exception as exc:  # pragma: no cover - defensive
        _raise_team_http_error(exc)


@router.get("/team/player/{player_id}/decisions")
async def team_player_decisions(
    player_id: str,
    request: Request,
    split: Optional[str] = Query(None),
    season: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
) -> dict[str, Any]:
    """Get shared decision history for one player."""
    try:
        return list_team_player_decisions(
            session_token=_session_token(request) or "",
            workspace_id=_workspace_id(request),
            player_id=player_id,
            split=split,
            season=season,
            limit=limit,
        )
    except Exception as exc:  # pragma: no cover - defensive
        _raise_team_http_error(exc)


@router.post("/team/assignments")
async def team_assignments_create(payload: TeamAssignmentRequest, request: Request) -> dict[str, Any]:
    """Create or update a shared player assignment."""
    try:
        return {
            "assignment": create_team_assignment(
                session_token=_session_token(request) or "",
                workspace_id=_workspace_id(request),
                player_id=payload.player_id,
                split=payload.split,
                season=payload.season,
                assignee_user_id=payload.assignee_user_id,
                status=payload.status,
                due_date=payload.due_date,
                note=payload.note,
            )
        }
    except Exception as exc:  # pragma: no cover - defensive
        _raise_team_http_error(exc)


@router.patch("/team/assignments/{assignment_id}")
async def team_assignments_patch(
    assignment_id: str,
    payload: TeamAssignmentPatchRequest,
    request: Request,
) -> dict[str, Any]:
    """Patch one shared player assignment."""
    try:
        return {
            "assignment": update_team_assignment(
                session_token=_session_token(request) or "",
                workspace_id=_workspace_id(request),
                assignment_id=assignment_id,
                assignee_user_id=payload.assignee_user_id,
                status=payload.status,
                due_date=payload.due_date,
                note=payload.note,
            )
        }
    except Exception as exc:  # pragma: no cover - defensive
        _raise_team_http_error(exc)


@router.get("/team/assignments")
async def team_assignments_list(
    request: Request,
    player_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
) -> dict[str, Any]:
    """List shared assignments for the active workspace."""
    try:
        return list_team_assignments(
            session_token=_session_token(request) or "",
            workspace_id=_workspace_id(request),
            player_id=player_id,
            limit=limit,
        )
    except Exception as exc:  # pragma: no cover - defensive
        _raise_team_http_error(exc)


@router.get("/team/player/{player_id}/comments")
async def team_player_comments(
    player_id: str,
    request: Request,
    split: Optional[str] = Query(None),
    season: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
) -> dict[str, Any]:
    """List shared comments for one player."""
    try:
        return list_team_player_comments(
            session_token=_session_token(request) or "",
            workspace_id=_workspace_id(request),
            player_id=player_id,
            split=split,
            season=season,
            limit=limit,
        )
    except Exception as exc:  # pragma: no cover - defensive
        _raise_team_http_error(exc)


@router.post("/team/player/{player_id}/comments")
async def team_player_comments_add(player_id: str, payload: TeamCommentRequest, request: Request) -> dict[str, Any]:
    """Append a shared comment to one player."""
    try:
        return {
            "comment": add_team_player_comment(
                session_token=_session_token(request) or "",
                workspace_id=_workspace_id(request),
                player_id=player_id,
                split=payload.split,
                season=payload.season,
                body=payload.body,
            )
        }
    except Exception as exc:  # pragma: no cover - defensive
        _raise_team_http_error(exc)


@router.get("/team/activity")
async def team_activity(request: Request, limit: int = Query(50, ge=1, le=500)) -> dict[str, Any]:
    """Return the shared workspace activity feed."""
    try:
        return list_team_activity(
            session_token=_session_token(request) or "",
            workspace_id=_workspace_id(request),
            limit=limit,
        )
    except Exception as exc:  # pragma: no cover - defensive
        _raise_team_http_error(exc)


@router.get("/team/compare-lists")
async def team_compare_lists(request: Request) -> dict[str, Any]:
    """List shared compare lists."""
    try:
        return list_team_compare_lists(
            session_token=_session_token(request) or "",
            workspace_id=_workspace_id(request),
        )
    except Exception as exc:  # pragma: no cover - defensive
        _raise_team_http_error(exc)


@router.post("/team/compare-lists")
async def team_compare_lists_create(payload: CompareListCreateRequest, request: Request) -> dict[str, Any]:
    """Create a shared compare list."""
    try:
        return {
            "compare_list": create_team_compare_list(
                session_token=_session_token(request) or "",
                workspace_id=_workspace_id(request),
                name=payload.name,
                notes=payload.notes,
            )
        }
    except Exception as exc:  # pragma: no cover - defensive
        _raise_team_http_error(exc)


@router.patch("/team/compare-lists/{compare_id}")
async def team_compare_lists_patch(compare_id: str, payload: CompareListPatchRequest, request: Request) -> dict[str, Any]:
    """Patch a compare list."""
    try:
        return {
            "compare_list": update_team_compare_list(
                session_token=_session_token(request) or "",
                workspace_id=_workspace_id(request),
                compare_id=compare_id,
                name=payload.name,
                notes=payload.notes,
            )
        }
    except Exception as exc:  # pragma: no cover - defensive
        _raise_team_http_error(exc)


@router.delete("/team/compare-lists/{compare_id}")
async def team_compare_lists_delete(compare_id: str, request: Request) -> dict[str, Any]:
    """Delete a compare list."""
    try:
        return delete_team_compare_list(
            session_token=_session_token(request) or "",
            workspace_id=_workspace_id(request),
            compare_id=compare_id,
        )
    except Exception as exc:  # pragma: no cover - defensive
        _raise_team_http_error(exc)


@router.post("/team/compare-lists/{compare_id}/players")
async def team_compare_lists_add_player(
    compare_id: str,
    payload: CompareListPlayerRequest,
    request: Request,
) -> dict[str, Any]:
    """Add one player to a shared compare list."""
    try:
        return {
            "item": add_player_to_compare_list(
                session_token=_session_token(request) or "",
                workspace_id=_workspace_id(request),
                compare_id=compare_id,
                player_id=payload.player_id,
                split=payload.split,
                season=payload.season,
                pinned=payload.pinned,
                notes=payload.notes,
            )
        }
    except Exception as exc:  # pragma: no cover - defensive
        _raise_team_http_error(exc)


@router.delete("/team/compare-lists/{compare_id}/players/{player_id}")
async def team_compare_lists_delete_player(compare_id: str, player_id: str, request: Request) -> dict[str, Any]:
    """Remove one player from a compare list."""
    try:
        return remove_player_from_compare_list(
            session_token=_session_token(request) or "",
            workspace_id=_workspace_id(request),
            compare_id=compare_id,
            player_id=player_id,
        )
    except Exception as exc:  # pragma: no cover - defensive
        _raise_team_http_error(exc)


@router.get("/team/preferences/me")
async def team_preferences_me(request: Request) -> dict[str, Any]:
    """Return the current user's scout preference profile."""
    try:
        return get_team_preferences_me(
            session_token=_session_token(request) or "",
            workspace_id=_workspace_id(request),
        )
    except Exception as exc:  # pragma: no cover - defensive
        _raise_team_http_error(exc)


@router.put("/team/preferences/me")
async def team_preferences_me_put(payload: PreferenceProfileRequest, request: Request) -> dict[str, Any]:
    """Create or update the current user's scout preference profile."""
    try:
        return put_team_preferences_me(
            session_token=_session_token(request) or "",
            workspace_id=_workspace_id(request),
            payload=payload.model_dump(),
        )
    except Exception as exc:  # pragma: no cover - defensive
        _raise_team_http_error(exc)


@router.get("/team/enabled")
async def team_enabled() -> dict[str, Any]:
    """Return whether shared workspace/team mode is configured."""
    return {"team_mode": bool(team_mode_enabled())}


__all__ = ["router"]
