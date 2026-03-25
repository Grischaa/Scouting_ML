"""Business logic for ScoutML Team Edition shared workspaces."""

from __future__ import annotations

import hashlib
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

import pandas as pd
from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from scouting_ml.core.runtime_config import load_api_runtime_config
from scouting_ml.services.market_value_service import (
    POSITIVE_SCOUT_DECISION_ACTIONS,
    SCOUT_DECISION_ACTIONS,
    SCOUT_DECISION_REASON_TAGS,
    get_player_prediction,
    get_player_profile,
    get_player_report,
    get_player_trajectory_view,
)
from scouting_ml.team.db import create_all_tables, session_scope, team_mode_enabled
from scouting_ml.team.models import (
    ASSIGNMENT_STATUSES,
    AuthSession,
    DECISION_ACTIONS,
    ScoutPreferenceProfile,
    TeamActivityEvent,
    TeamAssignment,
    TeamComment,
    TeamCompareList,
    TeamCompareListPlayer,
    TeamScoutDecision,
    TeamWatchlistItem,
    User,
    WORKSPACE_ROLES,
    Workspace,
    WorkspaceInvite,
    WorkspaceMembership,
)


try:  # pragma: no cover - optional runtime dependency
    from passlib.context import CryptContext  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional runtime dependency
    CryptContext = None

try:  # pragma: no cover - optional runtime dependency
    import bcrypt  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional runtime dependency
    bcrypt = None


SESSION_TTL_DAYS = 30
_PASSLIB_CONTEXT = CryptContext(schemes=["bcrypt"], deprecated="auto") if CryptContext is not None else None


class TeamModeDisabledError(RuntimeError):
    """Raised when a team-only route is used without team mode configuration."""


class TeamAuthError(PermissionError):
    """Raised when authentication or role checks fail."""


@dataclass(frozen=True)
class TeamRequestContext:
    """Resolved authenticated user and workspace scope for one request."""

    enabled: bool
    authenticated: bool
    session_id: str | None = None
    user_id: str | None = None
    user_email: str | None = None
    user_full_name: str | None = None
    workspace_id: str | None = None
    workspace_name: str | None = None
    workspace_slug: str | None = None
    role: str | None = None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _as_utc(value: datetime | None) -> datetime | None:
    """Normalize naive datetimes from SQLite into UTC-aware datetimes."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _normalized_email(email: str) -> str:
    token = _safe_text(email).casefold()
    if not token or "@" not in token:
        raise ValueError("A valid email address is required.")
    return token


def _slugify(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "-", _safe_text(value).casefold()).strip("-")
    return token or "workspace"


def _token_hash(token: str) -> str:
    secret = load_api_runtime_config().session_secret
    return hashlib.sha256(f"{secret}:{token}".encode("utf-8")).hexdigest()


def _password_hash(password: str) -> str:
    plain = _safe_text(password)
    if len(plain) < 8:
        raise ValueError("Password must be at least 8 characters.")
    if _PASSLIB_CONTEXT is not None:
        return str(_PASSLIB_CONTEXT.hash(plain))
    if bcrypt is not None:
        return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    salt = secrets.token_hex(16)
    iterations = 600_000
    digest = hashlib.pbkdf2_hmac("sha256", plain.encode("utf-8"), salt.encode("utf-8"), iterations).hex()
    return f"pbkdf2_sha256${iterations}${salt}${digest}"


def _verify_password(password: str, hashed: str) -> bool:
    plain = _safe_text(password)
    digest = _safe_text(hashed)
    if not plain or not digest:
        return False
    if _PASSLIB_CONTEXT is not None:
        try:
            return bool(_PASSLIB_CONTEXT.verify(plain, digest))
        except Exception:
            return False
    if bcrypt is not None:
        try:
            return bool(bcrypt.checkpw(plain.encode("utf-8"), digest.encode("utf-8")))
        except Exception:
            return False
    if digest.startswith("pbkdf2_sha256$"):
        try:
            _, iterations_raw, salt, expected = digest.split("$", 3)
            candidate = hashlib.pbkdf2_hmac(
                "sha256",
                plain.encode("utf-8"),
                salt.encode("utf-8"),
                int(iterations_raw),
            ).hex()
            return secrets.compare_digest(candidate, expected)
        except Exception:
            return False
    return False


def session_cookie_name() -> str:
    """Return the configured session cookie name."""
    return load_api_runtime_config().session_cookie_name


def session_cookie_settings() -> dict[str, Any]:
    """Return common cookie attributes for auth-session cookies."""
    config = load_api_runtime_config()
    return {
        "httponly": True,
        "samesite": "lax",
        "secure": bool(config.session_secure_cookie),
        "path": "/",
        "max_age": int(timedelta(days=SESSION_TTL_DAYS).total_seconds()),
    }


def _require_team_mode() -> None:
    if not team_mode_enabled():
        raise TeamModeDisabledError("Team mode is disabled. Set SCOUTING_DATABASE_URL to enable shared workspaces.")
    create_all_tables()


def _serialize_user(user: User) -> dict[str, Any]:
    return {
        "user_id": user.id,
        "email": user.email,
        "full_name": user.full_name,
        "is_active": bool(user.is_active),
        "created_at": _iso(user.created_at),
    }


def _workspace_memberships_for_user(db: Session, user_id: str) -> list[tuple[WorkspaceMembership, Workspace]]:
    stmt = (
        select(WorkspaceMembership, Workspace)
        .join(Workspace, Workspace.id == WorkspaceMembership.workspace_id)
        .where(WorkspaceMembership.user_id == user_id)
        .order_by(Workspace.name.asc())
    )
    return list(db.execute(stmt).all())


def _serialize_workspace_membership(membership: WorkspaceMembership, workspace: Workspace) -> dict[str, Any]:
    return {
        "workspace_id": workspace.id,
        "name": workspace.name,
        "slug": workspace.slug,
        "role": membership.role,
        "created_at": _iso(workspace.created_at),
    }


def _serialize_workspace_summary(db: Session, workspace: Workspace) -> dict[str, Any]:
    member_stmt = (
        select(WorkspaceMembership, User)
        .join(User, User.id == WorkspaceMembership.user_id)
        .where(WorkspaceMembership.workspace_id == workspace.id)
        .order_by(User.full_name.asc(), User.email.asc())
    )
    members = [
        {
            "membership_id": membership.id,
            "user_id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "role": membership.role,
            "created_at": _iso(membership.created_at),
        }
        for membership, user in db.execute(member_stmt).all()
    ]
    invite_stmt = (
        select(WorkspaceInvite)
        .where(WorkspaceInvite.workspace_id == workspace.id)
        .order_by(WorkspaceInvite.created_at.desc())
    )
    invites = [
        {
            "invite_id": invite.id,
            "email": invite.email,
            "role": invite.role,
            "accepted_at": _iso(invite.accepted_at),
            "expires_at": _iso(invite.expires_at),
            "created_at": _iso(invite.created_at),
        }
        for invite in db.execute(invite_stmt).scalars().all()
    ]
    return {
        "workspace_id": workspace.id,
        "name": workspace.name,
        "slug": workspace.slug,
        "created_at": _iso(workspace.created_at),
        "updated_at": _iso(workspace.updated_at),
        "members": members,
        "invites": invites,
    }


def _unique_workspace_slug(db: Session, name: str) -> str:
    base = _slugify(name)
    slug = base
    idx = 2
    while db.scalar(select(Workspace.id).where(Workspace.slug == slug).limit(1)) is not None:
        slug = f"{base}-{idx}"
        idx += 1
    return slug


def _log_activity(
    db: Session,
    *,
    workspace_id: str,
    actor_user_id: str | None,
    event_type: str,
    entity_type: str,
    entity_id: str,
    summary: str,
    player_id: str = "",
    split: str = "",
    season: str = "",
    payload: dict[str, Any] | None = None,
) -> None:
    db.add(
        TeamActivityEvent(
            workspace_id=workspace_id,
            actor_user_id=actor_user_id,
            event_type=event_type,
            entity_type=entity_type,
            entity_id=entity_id,
            player_id=player_id,
            split=split,
            season=season,
            summary=summary,
            payload=dict(payload or {}),
        )
    )


def _resolve_context_from_db(
    db: Session,
    *,
    session_token: str | None,
    workspace_id: str | None = None,
    require_auth: bool = False,
    allowed_roles: Sequence[str] | None = None,
) -> TeamRequestContext:
    if not team_mode_enabled():
        if require_auth:
            raise TeamModeDisabledError(
                "Team mode is disabled. Set SCOUTING_DATABASE_URL to enable shared workspaces."
            )
        return TeamRequestContext(enabled=False, authenticated=False)

    token = _safe_text(session_token)
    if not token:
        if require_auth:
            raise TeamAuthError("Authentication required.")
        return TeamRequestContext(enabled=True, authenticated=False)

    now = _utcnow()
    session_record = db.scalar(
        select(AuthSession).where(
            AuthSession.token_hash == _token_hash(token),
            AuthSession.expires_at >= now,
        )
    )
    if session_record is None:
        if require_auth:
            raise TeamAuthError("Session expired or invalid.")
        return TeamRequestContext(enabled=True, authenticated=False)

    user = db.get(User, session_record.user_id)
    if user is None or not user.is_active:
        if require_auth:
            raise TeamAuthError("User account is inactive.")
        return TeamRequestContext(enabled=True, authenticated=False)

    memberships = _workspace_memberships_for_user(db, user.id)
    chosen_membership: WorkspaceMembership | None = None
    chosen_workspace: Workspace | None = None
    wanted_workspace_id = _safe_text(workspace_id)
    if wanted_workspace_id:
        for membership, workspace in memberships:
            if workspace.id == wanted_workspace_id:
                chosen_membership = membership
                chosen_workspace = workspace
                break
        if chosen_membership is None:
            raise TeamAuthError("You do not have access to the requested workspace.")
    elif memberships:
        chosen_membership, chosen_workspace = memberships[0]

    if require_auth and chosen_membership is None:
        raise TeamAuthError("No workspace membership found for this account.")

    if allowed_roles and chosen_membership is not None and chosen_membership.role not in set(allowed_roles):
        raise TeamAuthError("Insufficient role for this workspace action.")

    session_record.last_seen_at = now
    return TeamRequestContext(
        enabled=True,
        authenticated=True,
        session_id=session_record.id,
        user_id=user.id,
        user_email=user.email,
        user_full_name=user.full_name,
        workspace_id=chosen_workspace.id if chosen_workspace is not None else None,
        workspace_name=chosen_workspace.name if chosen_workspace is not None else None,
        workspace_slug=chosen_workspace.slug if chosen_workspace is not None else None,
        role=chosen_membership.role if chosen_membership is not None else None,
    )


def resolve_team_context(
    *,
    session_token: str | None,
    workspace_id: str | None = None,
    require_auth: bool = False,
    allowed_roles: Sequence[str] | None = None,
) -> TeamRequestContext:
    """Resolve the current authenticated user and workspace from cookie/header state."""
    _require_team_mode()
    with session_scope() as db:
        return _resolve_context_from_db(
            db,
            session_token=session_token,
            workspace_id=workspace_id,
            require_auth=require_auth,
            allowed_roles=allowed_roles,
        )


def _build_auth_payload(
    db: Session,
    *,
    session_token: str | None,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    context = _resolve_context_from_db(db, session_token=session_token, workspace_id=workspace_id, require_auth=False)
    if not context.enabled:
        return {
            "team_mode": False,
            "authenticated": False,
            "cookie_name": session_cookie_name(),
            "user": None,
            "workspaces": [],
            "active_workspace": None,
        }
    if not context.authenticated or context.user_id is None:
        return {
            "team_mode": True,
            "authenticated": False,
            "cookie_name": session_cookie_name(),
            "user": None,
            "workspaces": [],
            "active_workspace": None,
        }
    user = db.get(User, context.user_id)
    assert user is not None
    memberships = _workspace_memberships_for_user(db, user.id)
    serialized_workspaces = [_serialize_workspace_membership(membership, workspace) for membership, workspace in memberships]
    active_workspace = db.get(Workspace, context.workspace_id) if context.workspace_id else None
    return {
        "team_mode": True,
        "authenticated": True,
        "cookie_name": session_cookie_name(),
        "user": _serialize_user(user),
        "workspaces": serialized_workspaces,
        "active_workspace": _serialize_workspace_summary(db, active_workspace) if active_workspace is not None else None,
    }


def get_auth_me_payload(*, session_token: str | None, workspace_id: str | None = None) -> dict[str, Any]:
    """Return current auth, user, and workspace state for the frontend."""
    if not team_mode_enabled():
        return {
            "team_mode": False,
            "authenticated": False,
            "cookie_name": session_cookie_name(),
            "user": None,
            "workspaces": [],
            "active_workspace": None,
        }
    _require_team_mode()
    with session_scope() as db:
        return _build_auth_payload(db, session_token=session_token, workspace_id=workspace_id)


def _create_session_record(db: Session, user_id: str) -> tuple[AuthSession, str]:
    token = secrets.token_urlsafe(32)
    session_record = AuthSession(
        user_id=user_id,
        token_hash=_token_hash(token),
        expires_at=_utcnow() + timedelta(days=SESSION_TTL_DAYS),
    )
    db.add(session_record)
    db.flush()
    return session_record, token


def bootstrap_admin(
    *,
    email: str,
    password: str,
    full_name: str | None = None,
    workspace_name: str | None = None,
) -> dict[str, Any]:
    """Create the first admin user and workspace, then issue a login session."""
    _require_team_mode()
    with session_scope() as db:
        existing_admin = db.scalar(
            select(WorkspaceMembership.id).where(WorkspaceMembership.role == "admin").limit(1)
        )
        if existing_admin is not None:
            raise ValueError("Bootstrap admin is disabled because an admin already exists.")
        user = User(
            email=_normalized_email(email),
            password_hash=_password_hash(password),
            full_name=_safe_text(full_name) or "Admin",
        )
        db.add(user)
        db.flush()
        workspace = Workspace(
            name=_safe_text(workspace_name) or "ScoutML Workspace",
            slug=_unique_workspace_slug(db, _safe_text(workspace_name) or "ScoutML Workspace"),
            owner_user_id=user.id,
        )
        db.add(workspace)
        db.flush()
        membership = WorkspaceMembership(workspace_id=workspace.id, user_id=user.id, role="admin")
        db.add(membership)
        session_record, token = _create_session_record(db, user.id)
        _log_activity(
            db,
            workspace_id=workspace.id,
            actor_user_id=user.id,
            event_type="workspace.bootstrap_admin",
            entity_type="workspace",
            entity_id=workspace.id,
            summary=f"{user.full_name or user.email} bootstrapped the workspace.",
        )
        payload = _build_auth_payload(db, session_token=token, workspace_id=workspace.id)
        payload["session_token"] = token
        payload["session_id"] = session_record.id
        return payload


def login_user(*, email: str, password: str, workspace_id: str | None = None) -> dict[str, Any]:
    """Authenticate one user and issue a new opaque session token."""
    _require_team_mode()
    with session_scope() as db:
        user = db.scalar(select(User).where(User.email == _normalized_email(email)))
        if user is None or not _verify_password(password, user.password_hash):
            raise TeamAuthError("Invalid email or password.")
        if not user.is_active:
            raise TeamAuthError("User account is inactive.")
        session_record, token = _create_session_record(db, user.id)
        payload = _build_auth_payload(db, session_token=token, workspace_id=workspace_id)
        payload["session_token"] = token
        payload["session_id"] = session_record.id
        return payload


def logout_user(*, session_token: str | None) -> dict[str, Any]:
    """Invalidate one existing auth session."""
    if not team_mode_enabled():
        return {"logged_out": True}
    if not session_token:
        return {"logged_out": True}
    _require_team_mode()
    with session_scope() as db:
        session_record = db.scalar(select(AuthSession).where(AuthSession.token_hash == _token_hash(session_token)))
        if session_record is not None:
            db.delete(session_record)
    return {"logged_out": True}


def create_workspace_for_current_user(*, session_token: str, name: str, workspace_id: str | None = None) -> dict[str, Any]:
    """Create a new workspace and attach the current user as admin."""
    _require_team_mode()
    with session_scope() as db:
        context = _resolve_context_from_db(db, session_token=session_token, workspace_id=None, require_auth=True)
        assert context.user_id is not None
        workspace_name = _safe_text(name)
        if not workspace_name:
            raise ValueError("Workspace name is required.")
        workspace = Workspace(
            name=workspace_name,
            slug=_unique_workspace_slug(db, workspace_name),
            owner_user_id=context.user_id,
        )
        db.add(workspace)
        db.flush()
        membership = WorkspaceMembership(workspace_id=workspace.id, user_id=context.user_id, role="admin")
        db.add(membership)
        db.flush()
        _log_activity(
            db,
            workspace_id=workspace.id,
            actor_user_id=context.user_id,
            event_type="workspace.created",
            entity_type="workspace",
            entity_id=workspace.id,
            summary=f"{context.user_full_name or context.user_email} created workspace {workspace.name}.",
        )
        return _build_auth_payload(db, session_token=session_token, workspace_id=workspace.id)


def get_workspace_state(*, session_token: str, workspace_id: str | None = None) -> dict[str, Any]:
    """Return current user plus workspace membership and member/invite state."""
    _require_team_mode()
    with session_scope() as db:
        return _build_auth_payload(db, session_token=session_token, workspace_id=workspace_id)


def create_workspace_invite(
    *,
    session_token: str,
    workspace_id: str,
    email: str | None,
    role: str,
) -> dict[str, Any]:
    """Create one manual-share invite token for a workspace."""
    _require_team_mode()
    normalized_role = _safe_text(role) or "scout"
    if normalized_role not in WORKSPACE_ROLES:
        raise ValueError(f"Unsupported workspace role: {role!r}.")
    with session_scope() as db:
        context = _resolve_context_from_db(
            db,
            session_token=session_token,
            workspace_id=workspace_id,
            require_auth=True,
            allowed_roles=("admin",),
        )
        token = secrets.token_urlsafe(24)
        invite = WorkspaceInvite(
            workspace_id=workspace_id,
            invited_by_user_id=context.user_id,
            email=_normalized_email(email) if _safe_text(email) else "",
            role=normalized_role,
            token_hash=_token_hash(token),
            expires_at=_utcnow() + timedelta(hours=load_api_runtime_config().invite_token_ttl_hours),
        )
        db.add(invite)
        db.flush()
        _log_activity(
            db,
            workspace_id=workspace_id,
            actor_user_id=context.user_id,
            event_type="workspace.invite_created",
            entity_type="workspace_invite",
            entity_id=invite.id,
            summary=f"{context.user_full_name or context.user_email} created a {normalized_role} invite.",
            payload={"email": invite.email, "role": normalized_role},
        )
        return {
            "invite_id": invite.id,
            "role": invite.role,
            "email": invite.email,
            "expires_at": _iso(invite.expires_at),
            "token": token,
            "invite_url": f"/app/index.html?invite_token={token}",
        }


def accept_workspace_invite(
    *,
    token: str,
    email: str,
    password: str,
    full_name: str | None = None,
) -> dict[str, Any]:
    """Accept an invite token, create or attach a user, and issue a session."""
    _require_team_mode()
    with session_scope() as db:
        invite = db.scalar(select(WorkspaceInvite).where(WorkspaceInvite.token_hash == _token_hash(token)))
        if invite is None:
            raise ValueError("Invite token is invalid.")
        now = _utcnow()
        if invite.accepted_at is not None:
            raise ValueError("Invite token has already been used.")
        if (_as_utc(invite.expires_at) or now) < now:
            raise ValueError("Invite token has expired.")
        normalized_email = _normalized_email(email)
        if invite.email and invite.email != normalized_email:
            raise ValueError("Invite token is restricted to a different email address.")

        user = db.scalar(select(User).where(User.email == normalized_email))
        if user is None:
            user = User(
                email=normalized_email,
                password_hash=_password_hash(password),
                full_name=_safe_text(full_name) or normalized_email.split("@")[0],
            )
            db.add(user)
            db.flush()
        elif not _verify_password(password, user.password_hash):
            raise TeamAuthError("Existing account password did not match.")

        membership = db.scalar(
            select(WorkspaceMembership).where(
                WorkspaceMembership.workspace_id == invite.workspace_id,
                WorkspaceMembership.user_id == user.id,
            )
        )
        if membership is None:
            membership = WorkspaceMembership(workspace_id=invite.workspace_id, user_id=user.id, role=invite.role)
            db.add(membership)
        invite.accepted_at = now
        session_record, session_token = _create_session_record(db, user.id)
        _log_activity(
            db,
            workspace_id=invite.workspace_id,
            actor_user_id=user.id,
            event_type="workspace.invite_accepted",
            entity_type="workspace_invite",
            entity_id=invite.id,
            summary=f"{user.full_name or user.email} joined the workspace as {membership.role}.",
        )
        payload = _build_auth_payload(db, session_token=session_token, workspace_id=invite.workspace_id)
        payload["session_token"] = session_token
        payload["session_id"] = session_record.id
        return payload


def _build_player_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "player_id": _safe_text(row.get("player_id")),
        "name": _safe_text(row.get("name")),
        "club": _safe_text(row.get("club")),
        "league": _safe_text(row.get("league")),
        "season": _safe_text(row.get("season")),
        "position": _safe_text(row.get("model_position") or row.get("position_group")),
        "age": row.get("age"),
        "market_value_eur": row.get("market_value_eur"),
        "fair_value_eur": row.get("fair_value_eur") if row.get("fair_value_eur") is not None else row.get("expected_value_eur"),
        "value_gap_conservative_eur": row.get("value_gap_conservative_eur"),
        "undervaluation_confidence": row.get("undervaluation_confidence"),
        "league_trust_tier": _safe_text(row.get("league_trust_tier")),
        "league_adjustment_bucket": _safe_text(row.get("league_adjustment_bucket")),
        "current_level_score": row.get("current_level_score"),
        "future_potential_score": row.get("future_potential_score"),
    }


def _currency_text(value: Any) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "-"
    return f"EUR {float(numeric):,.0f}"


def _watchlist_summary_text(row: dict[str, Any]) -> str:
    name = _safe_text(row.get("name")) or _safe_text(row.get("player_id")) or "Player"
    league = _safe_text(row.get("league")) or "Unknown league"
    gap = _currency_text(row.get("value_gap_conservative_eur"))
    conf = pd.to_numeric(pd.Series([row.get("undervaluation_confidence")]), errors="coerce").iloc[0]
    conf_text = "-" if pd.isna(conf) else f"{float(conf):.2f}"
    return f"{name} | {league} | conservative gap {gap} | confidence {conf_text}"


def _serialize_watchlist_item(item: TeamWatchlistItem) -> dict[str, Any]:
    snapshot = dict(item.player_snapshot or {})
    return {
        "watch_id": item.id,
        "player_id": item.player_id,
        "split": item.split,
        "season": item.season or snapshot.get("season") or "",
        "tag": item.tag or "",
        "notes": item.notes or "",
        "source": item.source or "",
        "summary_text": item.summary_text or "",
        "decision_action": item.decision_action or "",
        "decision_reason_tags": list(item.decision_reason_tags or []),
        "decision_note": item.decision_note or "",
        "last_decision_at_utc": _iso(item.last_decision_at_utc),
        "created_at_utc": _iso(item.created_at),
        "updated_at_utc": _iso(item.updated_at),
        **snapshot,
    }


def _serialize_decision(item: TeamScoutDecision) -> dict[str, Any]:
    return {
        "decision_id": item.id,
        "created_at_utc": _iso(item.created_at),
        "player_id": item.player_id,
        "split": item.split,
        "season": item.season,
        "action": item.action,
        "reason_tags": list(item.reason_tags or []),
        "note": item.note or "",
        "actor": item.actor or "",
        "source_surface": item.source_surface or "",
        "player_snapshot": dict(item.player_snapshot or {}),
        "ranking_context": dict(item.ranking_context or {}),
    }


def _serialize_assignment(item: TeamAssignment, *, assignee: User | None = None) -> dict[str, Any]:
    return {
        "assignment_id": item.id,
        "player_id": item.player_id,
        "split": item.split,
        "season": item.season,
        "assignee_user_id": item.assignee_user_id,
        "assignee_name": assignee.full_name if assignee is not None else "",
        "assignee_email": assignee.email if assignee is not None else "",
        "status": item.status,
        "due_date": _iso(item.due_date),
        "note": item.note or "",
        "created_at_utc": _iso(item.created_at),
        "updated_at_utc": _iso(item.updated_at),
    }


def _serialize_comment(item: TeamComment, *, author: User | None = None) -> dict[str, Any]:
    return {
        "comment_id": item.id,
        "player_id": item.player_id,
        "split": item.split,
        "season": item.season,
        "body": item.body,
        "author_user_id": item.author_user_id,
        "author_name": author.full_name if author is not None else "",
        "author_email": author.email if author is not None else "",
        "created_at_utc": _iso(item.created_at),
    }


def _serialize_activity(item: TeamActivityEvent, *, actor: User | None = None) -> dict[str, Any]:
    return {
        "activity_id": item.id,
        "event_type": item.event_type,
        "entity_type": item.entity_type,
        "entity_id": item.entity_id,
        "player_id": item.player_id,
        "split": item.split,
        "season": item.season,
        "summary": item.summary,
        "payload": dict(item.payload or {}),
        "actor_user_id": item.actor_user_id,
        "actor_name": actor.full_name if actor is not None else "",
        "actor_email": actor.email if actor is not None else "",
        "created_at_utc": _iso(item.created_at),
    }


def _find_or_upsert_watchlist_row(
    db: Session,
    *,
    context: TeamRequestContext,
    row: dict[str, Any],
    split: str,
    season: str,
    tag: str,
    notes: str,
    source: str,
    decision_action: str = "",
    decision_reason_tags: Sequence[str] | None = None,
    decision_note: str = "",
    last_decision_at: datetime | None = None,
) -> TeamWatchlistItem:
    normalized_tag = _safe_text(tag)
    existing = db.scalar(
        select(TeamWatchlistItem).where(
            TeamWatchlistItem.workspace_id == context.workspace_id,
            TeamWatchlistItem.player_id == _safe_text(row.get("player_id")),
            TeamWatchlistItem.split == split,
            TeamWatchlistItem.season == season,
            TeamWatchlistItem.tag == normalized_tag,
        )
    )
    if existing is None and not normalized_tag:
        existing = db.scalar(
            select(TeamWatchlistItem).where(
                TeamWatchlistItem.workspace_id == context.workspace_id,
                TeamWatchlistItem.player_id == _safe_text(row.get("player_id")),
                TeamWatchlistItem.split == split,
                TeamWatchlistItem.season == season,
            )
        )
    if existing is None:
        existing = TeamWatchlistItem(
            workspace_id=context.workspace_id or "",
            player_id=_safe_text(row.get("player_id")),
            split=split,
            season=season,
            tag=normalized_tag,
            created_by_user_id=context.user_id,
        )
        db.add(existing)
    snapshot = _build_player_snapshot(row)
    if not existing.tag and normalized_tag:
        existing.tag = normalized_tag
    existing.notes = notes if notes is not None else existing.notes
    existing.source = _safe_text(source) or existing.source or "manual"
    existing.summary_text = _watchlist_summary_text(row)
    existing.player_snapshot = snapshot
    existing.updated_by_user_id = context.user_id
    if decision_action:
        existing.decision_action = decision_action
        existing.decision_reason_tags = list(decision_reason_tags or [])
        existing.decision_note = decision_note or ""
        existing.last_decision_at_utc = last_decision_at
    db.flush()
    return existing


def list_team_watchlist(
    *,
    session_token: str,
    workspace_id: str | None = None,
    split: str | None = None,
    tag: str | None = None,
    player_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    """List workspace-scoped shared watchlist rows."""
    _require_team_mode()
    with session_scope() as db:
        context = _resolve_context_from_db(db, session_token=session_token, workspace_id=workspace_id, require_auth=True)
        stmt: Select[tuple[TeamWatchlistItem]] = select(TeamWatchlistItem).where(
            TeamWatchlistItem.workspace_id == context.workspace_id
        )
        if split:
            stmt = stmt.where(TeamWatchlistItem.split == split)
        if tag is not None:
            stmt = stmt.where(TeamWatchlistItem.tag == _safe_text(tag))
        if player_id:
            stmt = stmt.where(TeamWatchlistItem.player_id == _safe_text(player_id))
        total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
        items = db.execute(
            stmt.order_by(TeamWatchlistItem.last_decision_at_utc.desc(), TeamWatchlistItem.updated_at.desc())
            .offset(int(offset))
            .limit(int(limit))
        ).scalars().all()
        return {
            "workspace_id": context.workspace_id,
            "total": int(total),
            "count": int(len(items)),
            "limit": int(limit),
            "offset": int(offset),
            "items": [_serialize_watchlist_item(item) for item in items],
        }


def add_team_watchlist_item(
    *,
    session_token: str,
    workspace_id: str | None,
    player_id: str,
    split: str = "test",
    season: str | None = None,
    tag: str | None = None,
    notes: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """Create or update a workspace watchlist row from a player snapshot."""
    _require_team_mode()
    with session_scope() as db:
        context = _resolve_context_from_db(
            db,
            session_token=session_token,
            workspace_id=workspace_id,
            require_auth=True,
            allowed_roles=("admin", "scout"),
        )
        row = get_player_prediction(player_id=player_id, split=split, season=season)
        item = _find_or_upsert_watchlist_row(
            db,
            context=context,
            row=row,
            split=split,
            season=_safe_text(season) or _safe_text(row.get("season")),
            tag=_safe_text(tag),
            notes=_safe_text(notes),
            source=_safe_text(source) or "manual",
        )
        _log_activity(
            db,
            workspace_id=context.workspace_id or "",
            actor_user_id=context.user_id,
            event_type="watchlist.upsert",
            entity_type="watchlist_item",
            entity_id=item.id,
            summary=f"{context.user_full_name or context.user_email} updated the shared watchlist for {_safe_text(row.get('name')) or player_id}.",
            player_id=item.player_id,
            split=item.split,
            season=item.season,
            payload={"tag": item.tag, "source": item.source},
        )
        return _serialize_watchlist_item(item)


def update_team_watchlist_item(
    *,
    session_token: str,
    workspace_id: str | None,
    item_id: str,
    tag: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Patch one shared watchlist row."""
    _require_team_mode()
    with session_scope() as db:
        context = _resolve_context_from_db(
            db,
            session_token=session_token,
            workspace_id=workspace_id,
            require_auth=True,
            allowed_roles=("admin", "scout"),
        )
        item = db.scalar(
            select(TeamWatchlistItem).where(
                TeamWatchlistItem.id == item_id,
                TeamWatchlistItem.workspace_id == context.workspace_id,
            )
        )
        if item is None:
            raise ValueError("Shared watchlist item not found.")
        if tag is not None:
            item.tag = _safe_text(tag)
        if notes is not None:
            item.notes = _safe_text(notes)
        item.updated_by_user_id = context.user_id
        db.flush()
        _log_activity(
            db,
            workspace_id=context.workspace_id or "",
            actor_user_id=context.user_id,
            event_type="watchlist.updated",
            entity_type="watchlist_item",
            entity_id=item.id,
            summary=f"{context.user_full_name or context.user_email} edited a shared watchlist row.",
            player_id=item.player_id,
            split=item.split,
            season=item.season,
        )
        return _serialize_watchlist_item(item)


def delete_team_watchlist_item(*, session_token: str, workspace_id: str | None, item_id: str) -> dict[str, Any]:
    """Delete one shared watchlist row."""
    _require_team_mode()
    with session_scope() as db:
        context = _resolve_context_from_db(
            db,
            session_token=session_token,
            workspace_id=workspace_id,
            require_auth=True,
            allowed_roles=("admin", "scout"),
        )
        item = db.scalar(
            select(TeamWatchlistItem).where(
                TeamWatchlistItem.id == item_id,
                TeamWatchlistItem.workspace_id == context.workspace_id,
            )
        )
        if item is None:
            return {"deleted": False, "watch_id": item_id}
        db.delete(item)
        _log_activity(
            db,
            workspace_id=context.workspace_id or "",
            actor_user_id=context.user_id,
            event_type="watchlist.deleted",
            entity_type="watchlist_item",
            entity_id=item_id,
            summary=f"{context.user_full_name or context.user_email} removed a shared watchlist row.",
            player_id=item.player_id,
            split=item.split,
            season=item.season,
        )
        return {"deleted": True, "watch_id": item_id}


def _validate_decision(action: str, reason_tags: Sequence[str]) -> list[str]:
    normalized_action = _safe_text(action)
    if normalized_action not in DECISION_ACTIONS and normalized_action not in SCOUT_DECISION_ACTIONS:
        raise ValueError(f"Unsupported scout decision action: {action!r}.")
    normalized_reasons = [_safe_text(tag) for tag in reason_tags if _safe_text(tag)]
    allowed_tags = set(SCOUT_DECISION_REASON_TAGS["positive"]) | set(SCOUT_DECISION_REASON_TAGS["pass"])
    unknown = [tag for tag in normalized_reasons if tag not in allowed_tags]
    if unknown:
        raise ValueError(f"Unsupported scout decision reason tags: {', '.join(unknown)}.")
    if normalized_action in {"pass", "shortlist"} and not normalized_reasons:
        raise ValueError(f"Scout decision '{normalized_action}' requires at least one reason tag.")
    return normalized_reasons


def _latest_team_decision_record(
    db: Session,
    *,
    workspace_id: str,
    player_id: str,
    split: str | None = None,
    season: str | None = None,
) -> TeamScoutDecision | None:
    stmt = select(TeamScoutDecision).where(
        TeamScoutDecision.workspace_id == workspace_id,
        TeamScoutDecision.player_id == player_id,
    )
    if split:
        stmt = stmt.where(TeamScoutDecision.split == split)
    if season:
        stmt = stmt.where(TeamScoutDecision.season == season)
    stmt = stmt.order_by(TeamScoutDecision.created_at.desc()).limit(1)
    return db.scalar(stmt)


def get_latest_team_decision(
    *,
    session_token: str | None,
    workspace_id: str | None,
    player_id: str,
    split: str | None = None,
    season: str | None = None,
) -> dict[str, Any] | None:
    """Return the latest team decision for one player if a session/workspace is active."""
    if not team_mode_enabled() or not session_token:
        return None
    _require_team_mode()
    try:
        with session_scope() as db:
            context = _resolve_context_from_db(db, session_token=session_token, workspace_id=workspace_id, require_auth=True)
            if not context.workspace_id:
                return None
            record = _latest_team_decision_record(
                db,
                workspace_id=context.workspace_id,
                player_id=player_id,
                split=split,
                season=season,
            )
            return _serialize_decision(record) if record is not None else None
    except Exception:
        return None


def save_team_decision(
    *,
    session_token: str,
    workspace_id: str | None,
    player_id: str,
    split: str = "test",
    season: str | None = None,
    action: str,
    reason_tags: Sequence[str] | None = None,
    note: str | None = None,
    actor: str | None = None,
    source_surface: str | None = None,
    ranking_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist one workspace-scoped scout decision and sync positive actions into the shared watchlist."""
    _require_team_mode()
    normalized_reasons = _validate_decision(action, reason_tags or [])
    with session_scope() as db:
        context = _resolve_context_from_db(
            db,
            session_token=session_token,
            workspace_id=workspace_id,
            require_auth=True,
            allowed_roles=("admin", "scout"),
        )
        row = get_player_prediction(player_id=player_id, split=split, season=season)
        created_at = _utcnow()
        decision = TeamScoutDecision(
            workspace_id=context.workspace_id or "",
            player_id=player_id,
            split=split,
            season=_safe_text(season) or _safe_text(row.get("season")),
            action=_safe_text(action),
            reason_tags=list(normalized_reasons),
            note=_safe_text(note),
            actor=_safe_text(actor) or (context.user_full_name or context.user_email or "local"),
            source_surface=_safe_text(source_surface) or "detail",
            player_snapshot=_build_player_snapshot(row),
            ranking_context=dict(ranking_context or {}),
            created_by_user_id=context.user_id,
            created_at=created_at,
        )
        db.add(decision)
        db.flush()
        watchlist_item = None
        if decision.action in POSITIVE_SCOUT_DECISION_ACTIONS:
            watchlist = _find_or_upsert_watchlist_row(
                db,
                context=context,
                row=row,
                split=split,
                season=decision.season,
                tag="",
                notes=_safe_text(note),
                source=_safe_text(source_surface) or "decision_sync",
                decision_action=decision.action,
                decision_reason_tags=decision.reason_tags,
                decision_note=decision.note,
                last_decision_at=created_at,
            )
            watchlist_item = _serialize_watchlist_item(watchlist)
        _log_activity(
            db,
            workspace_id=context.workspace_id or "",
            actor_user_id=context.user_id,
            event_type="decision.saved",
            entity_type="team_scout_decision",
            entity_id=decision.id,
            summary=f"{context.user_full_name or context.user_email} logged '{decision.action}' for {_safe_text(row.get('name')) or player_id}.",
            player_id=decision.player_id,
            split=decision.split,
            season=decision.season,
            payload={"action": decision.action, "reason_tags": decision.reason_tags},
        )
        return {
            "decision": _serialize_decision(decision),
            "latest_decision": _serialize_decision(decision),
            "watchlist_item": watchlist_item,
        }


def list_team_player_decisions(
    *,
    session_token: str,
    workspace_id: str | None,
    player_id: str,
    split: str | None = None,
    season: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    """List one player's decision history inside the active workspace."""
    _require_team_mode()
    with session_scope() as db:
        context = _resolve_context_from_db(db, session_token=session_token, workspace_id=workspace_id, require_auth=True)
        stmt = select(TeamScoutDecision).where(
            TeamScoutDecision.workspace_id == context.workspace_id,
            TeamScoutDecision.player_id == player_id,
        )
        if split:
            stmt = stmt.where(TeamScoutDecision.split == split)
        if season:
            stmt = stmt.where(TeamScoutDecision.season == season)
        items = db.execute(stmt.order_by(TeamScoutDecision.created_at.desc()).limit(int(limit))).scalars().all()
        return {
            "player_id": player_id,
            "latest_decision": _serialize_decision(items[0]) if items else None,
            "events": [_serialize_decision(item) for item in items],
        }


def create_team_assignment(
    *,
    session_token: str,
    workspace_id: str | None,
    player_id: str,
    split: str = "test",
    season: str | None = None,
    assignee_user_id: str | None = None,
    status: str = "to_watch",
    due_date: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Create or update the current workspace assignment for one player."""
    _require_team_mode()
    normalized_status = _safe_text(status) or "to_watch"
    if normalized_status not in ASSIGNMENT_STATUSES:
        raise ValueError(f"Unsupported assignment status: {status!r}.")
    with session_scope() as db:
        context = _resolve_context_from_db(
            db,
            session_token=session_token,
            workspace_id=workspace_id,
            require_auth=True,
            allowed_roles=("admin", "scout"),
        )
        assignment = db.scalar(
            select(TeamAssignment).where(
                TeamAssignment.workspace_id == context.workspace_id,
                TeamAssignment.player_id == player_id,
                TeamAssignment.split == split,
                TeamAssignment.season == (_safe_text(season) or ""),
            )
        )
        if assignment is None:
            assignment = TeamAssignment(
                workspace_id=context.workspace_id or "",
                player_id=player_id,
                split=split,
                season=_safe_text(season),
                created_by_user_id=context.user_id,
            )
            db.add(assignment)
        if assignee_user_id is not None:
            if assignee_user_id:
                member_exists = db.scalar(
                    select(WorkspaceMembership.id).where(
                        WorkspaceMembership.workspace_id == context.workspace_id,
                        WorkspaceMembership.user_id == assignee_user_id,
                    )
                )
                if member_exists is None:
                    raise ValueError("Assignee is not a member of this workspace.")
            assignment.assignee_user_id = _safe_text(assignee_user_id) or None
        assignment.status = normalized_status
        assignment.note = _safe_text(note)
        assignment.updated_by_user_id = context.user_id
        assignment.due_date = datetime.fromisoformat(due_date) if _safe_text(due_date) else None
        db.flush()
        assignee = db.get(User, assignment.assignee_user_id) if assignment.assignee_user_id else None
        _log_activity(
            db,
            workspace_id=context.workspace_id or "",
            actor_user_id=context.user_id,
            event_type="assignment.saved",
            entity_type="team_assignment",
            entity_id=assignment.id,
            summary=f"{context.user_full_name or context.user_email} updated an assignment for {player_id}.",
            player_id=assignment.player_id,
            split=assignment.split,
            season=assignment.season,
            payload={"status": assignment.status, "assignee_user_id": assignment.assignee_user_id},
        )
        return _serialize_assignment(assignment, assignee=assignee)


def update_team_assignment(
    *,
    session_token: str,
    workspace_id: str | None,
    assignment_id: str,
    assignee_user_id: str | None = None,
    status: str | None = None,
    due_date: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Patch one existing assignment row."""
    _require_team_mode()
    with session_scope() as db:
        context = _resolve_context_from_db(
            db,
            session_token=session_token,
            workspace_id=workspace_id,
            require_auth=True,
            allowed_roles=("admin", "scout"),
        )
        assignment = db.scalar(
            select(TeamAssignment).where(
                TeamAssignment.id == assignment_id,
                TeamAssignment.workspace_id == context.workspace_id,
            )
        )
        if assignment is None:
            raise ValueError("Assignment not found.")
        return create_team_assignment(
            session_token=session_token,
            workspace_id=context.workspace_id,
            player_id=assignment.player_id,
            split=assignment.split,
            season=assignment.season,
            assignee_user_id=assignee_user_id if assignee_user_id is not None else assignment.assignee_user_id,
            status=status or assignment.status,
            due_date=due_date if due_date is not None else _iso(assignment.due_date),
            note=note if note is not None else assignment.note,
        )


def list_team_assignments(
    *,
    session_token: str,
    workspace_id: str | None,
    player_id: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    """List workspace assignments, optionally filtered to one player."""
    _require_team_mode()
    with session_scope() as db:
        context = _resolve_context_from_db(db, session_token=session_token, workspace_id=workspace_id, require_auth=True)
        stmt = select(TeamAssignment).where(TeamAssignment.workspace_id == context.workspace_id)
        if player_id:
            stmt = stmt.where(TeamAssignment.player_id == player_id)
        rows = db.execute(stmt.order_by(TeamAssignment.updated_at.desc()).limit(int(limit))).scalars().all()
        user_ids = {row.assignee_user_id for row in rows if row.assignee_user_id}
        user_map = {user.id: user for user in db.execute(select(User).where(User.id.in_(user_ids))).scalars().all()} if user_ids else {}
        return {
            "workspace_id": context.workspace_id,
            "items": [_serialize_assignment(row, assignee=user_map.get(row.assignee_user_id)) for row in rows],
        }


def list_team_player_comments(
    *,
    session_token: str,
    workspace_id: str | None,
    player_id: str,
    split: str | None = None,
    season: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    """List comments for one player in the active workspace."""
    _require_team_mode()
    with session_scope() as db:
        context = _resolve_context_from_db(db, session_token=session_token, workspace_id=workspace_id, require_auth=True)
        stmt = select(TeamComment).where(
            TeamComment.workspace_id == context.workspace_id,
            TeamComment.player_id == player_id,
        )
        if split:
            stmt = stmt.where(TeamComment.split == split)
        if season:
            stmt = stmt.where(TeamComment.season == season)
        rows = db.execute(stmt.order_by(TeamComment.created_at.desc()).limit(int(limit))).scalars().all()
        user_ids = {row.author_user_id for row in rows if row.author_user_id}
        user_map = {user.id: user for user in db.execute(select(User).where(User.id.in_(user_ids))).scalars().all()} if user_ids else {}
        return {
            "player_id": player_id,
            "items": [_serialize_comment(row, author=user_map.get(row.author_user_id)) for row in rows],
        }


def add_team_player_comment(
    *,
    session_token: str,
    workspace_id: str | None,
    player_id: str,
    split: str = "test",
    season: str | None = None,
    body: str,
) -> dict[str, Any]:
    """Append one workspace-scoped comment to a player."""
    _require_team_mode()
    comment_text = _safe_text(body)
    if not comment_text:
        raise ValueError("Comment body is required.")
    with session_scope() as db:
        context = _resolve_context_from_db(
            db,
            session_token=session_token,
            workspace_id=workspace_id,
            require_auth=True,
            allowed_roles=("admin", "scout"),
        )
        comment = TeamComment(
            workspace_id=context.workspace_id or "",
            player_id=player_id,
            split=split,
            season=_safe_text(season),
            body=comment_text,
            author_user_id=context.user_id,
        )
        db.add(comment)
        db.flush()
        author = db.get(User, context.user_id) if context.user_id else None
        _log_activity(
            db,
            workspace_id=context.workspace_id or "",
            actor_user_id=context.user_id,
            event_type="comment.saved",
            entity_type="team_comment",
            entity_id=comment.id,
            summary=f"{context.user_full_name or context.user_email} commented on {player_id}.",
            player_id=player_id,
            split=split,
            season=_safe_text(season),
        )
        return _serialize_comment(comment, author=author)


def list_team_activity(
    *,
    session_token: str,
    workspace_id: str | None,
    limit: int = 50,
) -> dict[str, Any]:
    """Return the workspace activity feed."""
    _require_team_mode()
    with session_scope() as db:
        context = _resolve_context_from_db(db, session_token=session_token, workspace_id=workspace_id, require_auth=True)
        rows = db.execute(
            select(TeamActivityEvent)
            .where(TeamActivityEvent.workspace_id == context.workspace_id)
            .order_by(TeamActivityEvent.created_at.desc())
            .limit(int(limit))
        ).scalars().all()
        user_ids = {row.actor_user_id for row in rows if row.actor_user_id}
        user_map = {user.id: user for user in db.execute(select(User).where(User.id.in_(user_ids))).scalars().all()} if user_ids else {}
        return {
            "workspace_id": context.workspace_id,
            "items": [_serialize_activity(row, actor=user_map.get(row.actor_user_id)) for row in rows],
        }


def _latest_assignment_for_player(db: Session, *, workspace_id: str, player_id: str, split: str, season: str) -> dict[str, Any] | None:
    row = db.scalar(
        select(TeamAssignment)
        .where(
            TeamAssignment.workspace_id == workspace_id,
            TeamAssignment.player_id == player_id,
            TeamAssignment.split == split,
            TeamAssignment.season == season,
        )
        .order_by(TeamAssignment.updated_at.desc())
        .limit(1)
    )
    if row is None:
        return None
    assignee = db.get(User, row.assignee_user_id) if row.assignee_user_id else None
    return _serialize_assignment(row, assignee=assignee)


def _build_compare_player_payload(
    db: Session,
    *,
    workspace_id: str,
    player_id: str,
    split: str,
    season: str | None,
) -> dict[str, Any]:
    row = get_player_prediction(player_id=player_id, split=split, season=season or None)
    report = get_player_report(player_id=player_id, split=split, season=season or None, top_metrics=4)
    trajectory = None
    try:
        trajectory = get_player_trajectory_view(player_id=player_id, split=split, season=season or None)
    except Exception:
        trajectory = None
    latest_decision = _latest_team_decision_record(
        db,
        workspace_id=workspace_id,
        player_id=player_id,
        split=split,
        season=_safe_text(season),
    )
    return {
        "player_id": player_id,
        "split": split,
        "season": _safe_text(season) or _safe_text(row.get("season")),
        "snapshot": _build_player_snapshot(row),
        "comparison": {
            "market_value_eur": row.get("market_value_eur"),
            "fair_value_eur": row.get("fair_value_eur") if row.get("fair_value_eur") is not None else row.get("expected_value_eur"),
            "current_level_score": row.get("current_level_score"),
            "future_potential_score": row.get("future_potential_score"),
            "undervaluation_confidence": row.get("undervaluation_confidence"),
            "trajectory_label": trajectory.get("trajectory_label") if isinstance(trajectory, dict) else None,
            "trajectory_slope_pct": trajectory.get("slope_pct") if isinstance(trajectory, dict) else None,
            "risk_flags": list(report.get("risk_flags") or []),
            "latest_decision": _serialize_decision(latest_decision) if latest_decision is not None else None,
            "latest_assignment": _latest_assignment_for_player(
                db,
                workspace_id=workspace_id,
                player_id=player_id,
                split=split,
                season=_safe_text(season) or _safe_text(row.get("season")),
            ),
        },
    }


def list_team_compare_lists(*, session_token: str, workspace_id: str | None) -> dict[str, Any]:
    """List shared compare lists with embedded comparison players."""
    _require_team_mode()
    with session_scope() as db:
        context = _resolve_context_from_db(db, session_token=session_token, workspace_id=workspace_id, require_auth=True)
        compare_lists = db.execute(
            select(TeamCompareList)
            .where(TeamCompareList.workspace_id == context.workspace_id)
            .order_by(TeamCompareList.updated_at.desc())
        ).scalars().all()
        out: list[dict[str, Any]] = []
        for compare in compare_lists:
            owner = db.get(User, compare.owner_user_id) if compare.owner_user_id else None
            players = db.execute(
                select(TeamCompareListPlayer)
                .where(TeamCompareListPlayer.compare_list_id == compare.id)
                .order_by(TeamCompareListPlayer.pinned.desc(), TeamCompareListPlayer.created_at.asc())
            ).scalars().all()
            out.append(
                {
                    "compare_id": compare.id,
                    "name": compare.name,
                    "notes": compare.notes or "",
                    "owner_user_id": compare.owner_user_id,
                    "owner_name": owner.full_name if owner is not None else "",
                    "created_at_utc": _iso(compare.created_at),
                    "updated_at_utc": _iso(compare.updated_at),
                    "players": [
                        {
                            "compare_player_id": player.id,
                            "pinned": bool(player.pinned),
                            "notes": player.notes or "",
                            **_build_compare_player_payload(
                                db,
                                workspace_id=context.workspace_id or "",
                                player_id=player.player_id,
                                split=player.split,
                                season=player.season,
                            ),
                        }
                        for player in players
                    ],
                }
            )
        return {"workspace_id": context.workspace_id, "items": out}


def create_team_compare_list(
    *,
    session_token: str,
    workspace_id: str | None,
    name: str,
    notes: str | None = None,
) -> dict[str, Any]:
    """Create a new shared compare list."""
    _require_team_mode()
    compare_name = _safe_text(name)
    if not compare_name:
        raise ValueError("Compare list name is required.")
    with session_scope() as db:
        context = _resolve_context_from_db(
            db,
            session_token=session_token,
            workspace_id=workspace_id,
            require_auth=True,
            allowed_roles=("admin", "scout"),
        )
        compare = TeamCompareList(
            workspace_id=context.workspace_id or "",
            owner_user_id=context.user_id,
            name=compare_name,
            notes=_safe_text(notes),
        )
        db.add(compare)
        db.flush()
        _log_activity(
            db,
            workspace_id=context.workspace_id or "",
            actor_user_id=context.user_id,
            event_type="compare.created",
            entity_type="team_compare_list",
            entity_id=compare.id,
            summary=f"{context.user_full_name or context.user_email} created compare list {compare.name}.",
        )
        return {"compare_id": compare.id, "name": compare.name, "notes": compare.notes or ""}


def update_team_compare_list(
    *,
    session_token: str,
    workspace_id: str | None,
    compare_id: str,
    name: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Patch one compare list."""
    _require_team_mode()
    with session_scope() as db:
        context = _resolve_context_from_db(
            db,
            session_token=session_token,
            workspace_id=workspace_id,
            require_auth=True,
            allowed_roles=("admin", "scout"),
        )
        compare = db.scalar(
            select(TeamCompareList).where(
                TeamCompareList.id == compare_id,
                TeamCompareList.workspace_id == context.workspace_id,
            )
        )
        if compare is None:
            raise ValueError("Compare list not found.")
        if name is not None:
            compare.name = _safe_text(name) or compare.name
        if notes is not None:
            compare.notes = _safe_text(notes)
        db.flush()
        return {"compare_id": compare.id, "name": compare.name, "notes": compare.notes or ""}


def delete_team_compare_list(*, session_token: str, workspace_id: str | None, compare_id: str) -> dict[str, Any]:
    """Delete one compare list and its players."""
    _require_team_mode()
    with session_scope() as db:
        context = _resolve_context_from_db(
            db,
            session_token=session_token,
            workspace_id=workspace_id,
            require_auth=True,
            allowed_roles=("admin", "scout"),
        )
        compare = db.scalar(
            select(TeamCompareList).where(
                TeamCompareList.id == compare_id,
                TeamCompareList.workspace_id == context.workspace_id,
            )
        )
        if compare is None:
            return {"deleted": False, "compare_id": compare_id}
        players = db.execute(select(TeamCompareListPlayer).where(TeamCompareListPlayer.compare_list_id == compare.id)).scalars().all()
        for player in players:
            db.delete(player)
        db.delete(compare)
        _log_activity(
            db,
            workspace_id=context.workspace_id or "",
            actor_user_id=context.user_id,
            event_type="compare.deleted",
            entity_type="team_compare_list",
            entity_id=compare_id,
            summary=f"{context.user_full_name or context.user_email} deleted a compare list.",
        )
        return {"deleted": True, "compare_id": compare_id}


def add_player_to_compare_list(
    *,
    session_token: str,
    workspace_id: str | None,
    compare_id: str,
    player_id: str,
    split: str = "test",
    season: str | None = None,
    pinned: bool = False,
    notes: str | None = None,
) -> dict[str, Any]:
    """Add one player to a shared compare list, capped at four players."""
    _require_team_mode()
    with session_scope() as db:
        context = _resolve_context_from_db(
            db,
            session_token=session_token,
            workspace_id=workspace_id,
            require_auth=True,
            allowed_roles=("admin", "scout"),
        )
        compare = db.scalar(
            select(TeamCompareList).where(
                TeamCompareList.id == compare_id,
                TeamCompareList.workspace_id == context.workspace_id,
            )
        )
        if compare is None:
            raise ValueError("Compare list not found.")
        current_rows = db.execute(
            select(TeamCompareListPlayer).where(TeamCompareListPlayer.compare_list_id == compare_id)
        ).scalars().all()
        if len(current_rows) >= 4 and not any(
            row.player_id == player_id and row.split == split and row.season == (_safe_text(season) or "")
            for row in current_rows
        ):
            raise ValueError("Compare lists support at most 4 players.")
        existing = next(
            (
                row
                for row in current_rows
                if row.player_id == player_id and row.split == split and row.season == (_safe_text(season) or "")
            ),
            None,
        )
        if existing is None:
            existing = TeamCompareListPlayer(
                compare_list_id=compare_id,
                player_id=player_id,
                split=split,
                season=_safe_text(season),
                added_by_user_id=context.user_id,
            )
            db.add(existing)
        existing.pinned = bool(pinned)
        existing.notes = _safe_text(notes)
        db.flush()
        _log_activity(
            db,
            workspace_id=context.workspace_id or "",
            actor_user_id=context.user_id,
            event_type="compare.player_added",
            entity_type="team_compare_list_player",
            entity_id=existing.id,
            summary=f"{context.user_full_name or context.user_email} added {player_id} to compare list {compare.name}.",
            player_id=player_id,
            split=split,
            season=_safe_text(season),
        )
        return {
            "compare_player_id": existing.id,
            "compare_id": compare_id,
            "player_id": player_id,
            "split": split,
            "season": existing.season,
            "pinned": existing.pinned,
            "notes": existing.notes or "",
        }


def remove_player_from_compare_list(
    *,
    session_token: str,
    workspace_id: str | None,
    compare_id: str,
    player_id: str,
) -> dict[str, Any]:
    """Remove one player from a compare list."""
    _require_team_mode()
    with session_scope() as db:
        context = _resolve_context_from_db(
            db,
            session_token=session_token,
            workspace_id=workspace_id,
            require_auth=True,
            allowed_roles=("admin", "scout"),
        )
        rows = db.execute(
            select(TeamCompareListPlayer)
            .join(TeamCompareList, TeamCompareList.id == TeamCompareListPlayer.compare_list_id)
            .where(
                TeamCompareList.workspace_id == context.workspace_id,
                TeamCompareListPlayer.compare_list_id == compare_id,
                TeamCompareListPlayer.player_id == player_id,
            )
        ).scalars().all()
        deleted = False
        for row in rows:
            db.delete(row)
            deleted = True
        return {"deleted": deleted, "compare_id": compare_id, "player_id": player_id}


def _default_preference_payload(*, user_id: str, workspace_id: str) -> dict[str, Any]:
    return {
        "preference_profile_id": None,
        "workspace_id": workspace_id,
        "user_id": user_id,
        "name": "Primary",
        "target_age_min": 18,
        "target_age_max": 24,
        "budget_posture": "balanced",
        "trusted_league_posture": "balanced",
        "role_priorities": {},
        "system_template_default": "",
        "must_have_tags": [],
        "avoid_tags": [],
        "risk_tolerance": "balanced",
        "active_lane_preference": "valuation",
        "apply_by_default": True,
    }


def _serialize_preference_profile(item: ScoutPreferenceProfile) -> dict[str, Any]:
    return {
        "preference_profile_id": item.id,
        "workspace_id": item.workspace_id,
        "user_id": item.user_id,
        "name": item.name,
        "target_age_min": item.target_age_min,
        "target_age_max": item.target_age_max,
        "budget_posture": item.budget_posture,
        "trusted_league_posture": item.trusted_league_posture,
        "role_priorities": dict(item.role_priorities or {}),
        "system_template_default": item.system_template_default or "",
        "must_have_tags": list(item.must_have_tags or []),
        "avoid_tags": list(item.avoid_tags or []),
        "risk_tolerance": item.risk_tolerance,
        "active_lane_preference": item.active_lane_preference,
        "apply_by_default": bool(item.apply_by_default),
        "created_at_utc": _iso(item.created_at),
        "updated_at_utc": _iso(item.updated_at),
    }


def get_team_preferences_me(*, session_token: str, workspace_id: str | None) -> dict[str, Any]:
    """Return the current user's scout preference profile for the active workspace."""
    _require_team_mode()
    with session_scope() as db:
        context = _resolve_context_from_db(db, session_token=session_token, workspace_id=workspace_id, require_auth=True)
        profile = db.scalar(
            select(ScoutPreferenceProfile).where(
                ScoutPreferenceProfile.workspace_id == context.workspace_id,
                ScoutPreferenceProfile.user_id == context.user_id,
            )
        )
        if profile is None:
            assert context.user_id is not None and context.workspace_id is not None
            return _default_preference_payload(user_id=context.user_id, workspace_id=context.workspace_id)
        return _serialize_preference_profile(profile)


def put_team_preferences_me(
    *,
    session_token: str,
    workspace_id: str | None,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Create or update the current user's preference profile."""
    _require_team_mode()
    with session_scope() as db:
        context = _resolve_context_from_db(
            db,
            session_token=session_token,
            workspace_id=workspace_id,
            require_auth=True,
            allowed_roles=("admin", "scout"),
        )
        profile = db.scalar(
            select(ScoutPreferenceProfile).where(
                ScoutPreferenceProfile.workspace_id == context.workspace_id,
                ScoutPreferenceProfile.user_id == context.user_id,
            )
        )
        if profile is None:
            profile = ScoutPreferenceProfile(workspace_id=context.workspace_id or "", user_id=context.user_id or "")
            db.add(profile)
        profile.name = _safe_text(payload.get("name")) or "Primary"
        profile.target_age_min = int(payload["target_age_min"]) if payload.get("target_age_min") is not None else None
        profile.target_age_max = int(payload["target_age_max"]) if payload.get("target_age_max") is not None else None
        profile.budget_posture = _safe_text(payload.get("budget_posture")) or "balanced"
        profile.trusted_league_posture = _safe_text(payload.get("trusted_league_posture")) or "balanced"
        profile.role_priorities = dict(payload.get("role_priorities") or {})
        profile.system_template_default = _safe_text(payload.get("system_template_default"))
        profile.must_have_tags = [tag for tag in payload.get("must_have_tags") or [] if _safe_text(tag)]
        profile.avoid_tags = [tag for tag in payload.get("avoid_tags") or [] if _safe_text(tag)]
        profile.risk_tolerance = _safe_text(payload.get("risk_tolerance")) or "balanced"
        profile.active_lane_preference = _safe_text(payload.get("active_lane_preference")) or "valuation"
        profile.apply_by_default = bool(payload.get("apply_by_default", True))
        db.flush()
        _log_activity(
            db,
            workspace_id=context.workspace_id or "",
            actor_user_id=context.user_id,
            event_type="preferences.saved",
            entity_type="scout_preference_profile",
            entity_id=profile.id,
            summary=f"{context.user_full_name or context.user_email} updated their scout preferences.",
        )
        return _serialize_preference_profile(profile)


def _resolve_preference_profile(
    db: Session,
    *,
    context: TeamRequestContext,
    preference_profile_id: str | None = None,
) -> dict[str, Any] | None:
    if not context.workspace_id or not context.user_id:
        return None
    stmt = select(ScoutPreferenceProfile).where(ScoutPreferenceProfile.workspace_id == context.workspace_id)
    if _safe_text(preference_profile_id):
        stmt = stmt.where(ScoutPreferenceProfile.id == _safe_text(preference_profile_id))
    else:
        stmt = stmt.where(ScoutPreferenceProfile.user_id == context.user_id)
    profile = db.scalar(stmt.limit(1))
    if profile is None:
        return _default_preference_payload(user_id=context.user_id, workspace_id=context.workspace_id)
    return _serialize_preference_profile(profile)


def apply_preference_overlay(
    items: Sequence[dict[str, Any]],
    *,
    profile: dict[str, Any] | None,
    mode: str,
    active_lane: str | None = None,
) -> list[dict[str, Any]]:
    """Apply a lightweight scout-preference reranking overlay to API rows."""
    if not profile:
        return [dict(item) for item in items]
    out: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        row = dict(item)
        score = 1.0
        notes: list[str] = []
        age = pd.to_numeric(pd.Series([row.get("age")]), errors="coerce").iloc[0]
        age_min = profile.get("target_age_min")
        age_max = profile.get("target_age_max")
        if not pd.isna(age):
            if age_min is not None and age < int(age_min):
                score *= 0.92
                notes.append("below target age band")
            elif age_max is not None and age > int(age_max):
                score *= 0.88
                notes.append("above target age band")
            else:
                score *= 1.05
                notes.append("inside target age band")
        trust = _safe_text(row.get("league_trust_tier")) or "unknown"
        posture = _safe_text(profile.get("trusted_league_posture")) or "balanced"
        if posture == "trusted_first":
            trust_boost = {"trusted": 1.08, "watch": 0.98, "unknown": 0.9, "blocked": 0.78}.get(trust, 0.92)
            score *= trust_boost
            notes.append(f"{trust} league posture")
        elif posture == "frontier":
            trust_boost = {"trusted": 1.0, "watch": 1.02, "unknown": 1.0, "blocked": 0.9}.get(trust, 1.0)
            score *= trust_boost
        budget_posture = _safe_text(profile.get("budget_posture")) or "balanced"
        market = pd.to_numeric(pd.Series([row.get("market_value_eur")]), errors="coerce").iloc[0]
        if not pd.isna(market):
            if budget_posture == "disciplined":
                score *= 1.06 if market <= 10_000_000 else 0.92
            elif budget_posture == "stretch":
                score *= 1.03 if market <= 20_000_000 else 0.98
        risk_tolerance = _safe_text(profile.get("risk_tolerance")) or "balanced"
        confidence = pd.to_numeric(
            pd.Series(
                [
                    row.get("system_fit_confidence")
                    if mode == "system_fit"
                    else row.get("undervaluation_confidence")
                ]
            ),
            errors="coerce",
        ).iloc[0]
        if not pd.isna(confidence) and risk_tolerance == "conservative":
            score *= 1.06 if float(confidence) >= 0.8 or float(confidence) >= 80 else 0.9
            notes.append("confidence aligned")
        role_priorities = dict(profile.get("role_priorities") or {})
        position_key = _safe_text(row.get("model_position") or row.get("position_group") or row.get("talent_position_family"))
        if position_key and role_priorities:
            role_weight = float(role_priorities.get(position_key, role_priorities.get(position_key.upper(), 1.0)) or 1.0)
            score *= max(role_weight, 0.5)
            if role_weight != 1.0:
                notes.append(f"role priority {position_key}")
        if active_lane and _safe_text(profile.get("active_lane_preference")) == active_lane:
            score *= 1.02
        row["preference_overlay_score"] = round(float(score), 4)
        row["preference_overlay_notes"] = notes[:3]
        row["_preference_sort_score"] = float(score)
        row["_preference_original_rank"] = index
        out.append(row)
    out.sort(
        key=lambda row: (
            float(row.get("_preference_sort_score") or 1.0),
            float(pd.to_numeric(pd.Series([row.get("system_fit_score") if mode == "system_fit" else row.get("shortlist_score") or row.get("scout_target_score") or row.get("undervaluation_score") or row.get("value_gap_conservative_eur")]), errors="coerce").iloc[0] or 0.0),
            -int(row.get("_preference_original_rank") or 0),
        ),
        reverse=True,
    )
    for row in out:
        row.pop("_preference_sort_score", None)
        row.pop("_preference_original_rank", None)
    return out


def maybe_apply_preferences_to_rows(
    rows: Sequence[dict[str, Any]],
    *,
    session_token: str | None,
    workspace_id: str | None,
    preference_profile_id: str | None,
    apply_preferences: bool,
    mode: str,
    active_lane: str | None = None,
) -> list[dict[str, Any]]:
    """Resolve the active user's preference profile and rerank rows when requested."""
    if not apply_preferences or not team_mode_enabled() or not session_token:
        return [dict(item) for item in rows]
    try:
        with session_scope() as db:
            context = _resolve_context_from_db(db, session_token=session_token, workspace_id=workspace_id, require_auth=True)
            profile = _resolve_preference_profile(db, context=context, preference_profile_id=preference_profile_id)
            return apply_preference_overlay(rows, profile=profile, mode=mode, active_lane=active_lane)
    except Exception:
        return [dict(item) for item in rows]


__all__ = [
    "TeamAuthError",
    "TeamModeDisabledError",
    "TeamRequestContext",
    "accept_workspace_invite",
    "add_player_to_compare_list",
    "add_team_player_comment",
    "add_team_watchlist_item",
    "apply_preference_overlay",
    "bootstrap_admin",
    "create_team_assignment",
    "create_team_compare_list",
    "create_workspace_for_current_user",
    "create_workspace_invite",
    "delete_team_compare_list",
    "delete_team_watchlist_item",
    "get_auth_me_payload",
    "get_latest_team_decision",
    "get_team_preferences_me",
    "get_workspace_state",
    "list_team_activity",
    "list_team_assignments",
    "list_team_compare_lists",
    "list_team_player_comments",
    "list_team_player_decisions",
    "list_team_watchlist",
    "login_user",
    "logout_user",
    "maybe_apply_preferences_to_rows",
    "put_team_preferences_me",
    "remove_player_from_compare_list",
    "resolve_team_context",
    "save_team_decision",
    "session_cookie_name",
    "session_cookie_settings",
    "team_mode_enabled",
    "update_team_assignment",
    "update_team_compare_list",
    "update_team_watchlist_item",
]
