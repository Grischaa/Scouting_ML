"""SQLAlchemy models for ScoutML Team Edition shared workspace state."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


WORKSPACE_ROLES: tuple[str, ...] = ("admin", "scout", "viewer")
DECISION_ACTIONS: tuple[str, ...] = ("shortlist", "watch_live", "request_report", "pass")
ASSIGNMENT_STATUSES: tuple[str, ...] = (
    "to_watch",
    "watched",
    "report_requested",
    "passed",
    "shortlisted",
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _uuid_str() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    """Declarative base for team-edition models."""


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid_str)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    full_name: Mapped[str] = mapped_column(String(200), default="")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)


class Workspace(Base):
    __tablename__ = "workspaces"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid_str)
    name: Mapped[str] = mapped_column(String(200))
    slug: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    owner_user_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)


class WorkspaceMembership(Base):
    __tablename__ = "workspace_memberships"
    __table_args__ = (UniqueConstraint("workspace_id", "user_id", name="uq_workspace_membership"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid_str)
    workspace_id: Mapped[str] = mapped_column(String(32), ForeignKey("workspaces.id"), index=True)
    user_id: Mapped[str] = mapped_column(String(32), ForeignKey("users.id"), index=True)
    role: Mapped[str] = mapped_column(String(16), default="scout")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)


class WorkspaceInvite(Base):
    __tablename__ = "workspace_invites"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid_str)
    workspace_id: Mapped[str] = mapped_column(String(32), ForeignKey("workspaces.id"), index=True)
    invited_by_user_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("users.id"), nullable=True)
    email: Mapped[str] = mapped_column(String(320), default="")
    role: Mapped[str] = mapped_column(String(16), default="scout")
    token_hash: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    accepted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class AuthSession(Base):
    __tablename__ = "auth_sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid_str)
    user_id: Mapped[str] = mapped_column(String(32), ForeignKey("users.id"), index=True)
    token_hash: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class TeamWatchlistItem(Base):
    __tablename__ = "team_watchlist_items"
    __table_args__ = (
        UniqueConstraint(
            "workspace_id",
            "player_id",
            "split",
            "season",
            "tag",
            name="uq_team_watchlist_player_scope",
        ),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid_str)
    workspace_id: Mapped[str] = mapped_column(String(32), ForeignKey("workspaces.id"), index=True)
    player_id: Mapped[str] = mapped_column(String(128), index=True)
    split: Mapped[str] = mapped_column(String(8), default="test")
    season: Mapped[str] = mapped_column(String(32), default="")
    tag: Mapped[str] = mapped_column(String(120), default="")
    notes: Mapped[str] = mapped_column(Text, default="")
    source: Mapped[str] = mapped_column(String(64), default="manual")
    summary_text: Mapped[str] = mapped_column(Text, default="")
    player_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    decision_action: Mapped[str] = mapped_column(String(32), default="")
    decision_reason_tags: Mapped[list] = mapped_column(JSON, default=list)
    decision_note: Mapped[str] = mapped_column(Text, default="")
    last_decision_at_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_by_user_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("users.id"), nullable=True)
    updated_by_user_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)


class TeamScoutDecision(Base):
    __tablename__ = "team_scout_decisions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid_str)
    workspace_id: Mapped[str] = mapped_column(String(32), ForeignKey("workspaces.id"), index=True)
    player_id: Mapped[str] = mapped_column(String(128), index=True)
    split: Mapped[str] = mapped_column(String(8), default="test")
    season: Mapped[str] = mapped_column(String(32), default="")
    action: Mapped[str] = mapped_column(String(32))
    reason_tags: Mapped[list] = mapped_column(JSON, default=list)
    note: Mapped[str] = mapped_column(Text, default="")
    actor: Mapped[str] = mapped_column(String(80), default="local")
    source_surface: Mapped[str] = mapped_column(String(80), default="detail")
    player_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    ranking_context: Mapped[dict] = mapped_column(JSON, default=dict)
    created_by_user_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)


class TeamAssignment(Base):
    __tablename__ = "team_assignments"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid_str)
    workspace_id: Mapped[str] = mapped_column(String(32), ForeignKey("workspaces.id"), index=True)
    player_id: Mapped[str] = mapped_column(String(128), index=True)
    split: Mapped[str] = mapped_column(String(8), default="test")
    season: Mapped[str] = mapped_column(String(32), default="")
    assignee_user_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("users.id"), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(32), default="to_watch")
    due_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    note: Mapped[str] = mapped_column(Text, default="")
    created_by_user_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("users.id"), nullable=True)
    updated_by_user_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)


class TeamComment(Base):
    __tablename__ = "team_comments"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid_str)
    workspace_id: Mapped[str] = mapped_column(String(32), ForeignKey("workspaces.id"), index=True)
    player_id: Mapped[str] = mapped_column(String(128), index=True)
    split: Mapped[str] = mapped_column(String(8), default="test")
    season: Mapped[str] = mapped_column(String(32), default="")
    body: Mapped[str] = mapped_column(Text)
    author_user_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)


class TeamActivityEvent(Base):
    __tablename__ = "team_activity_events"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid_str)
    workspace_id: Mapped[str] = mapped_column(String(32), ForeignKey("workspaces.id"), index=True)
    actor_user_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("users.id"), nullable=True)
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    entity_type: Mapped[str] = mapped_column(String(64), default="")
    entity_id: Mapped[str] = mapped_column(String(64), default="")
    player_id: Mapped[str] = mapped_column(String(128), default="", index=True)
    split: Mapped[str] = mapped_column(String(8), default="")
    season: Mapped[str] = mapped_column(String(32), default="")
    summary: Mapped[str] = mapped_column(Text, default="")
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)


class TeamCompareList(Base):
    __tablename__ = "team_compare_lists"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid_str)
    workspace_id: Mapped[str] = mapped_column(String(32), ForeignKey("workspaces.id"), index=True)
    owner_user_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("users.id"), nullable=True)
    name: Mapped[str] = mapped_column(String(200))
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)


class TeamCompareListPlayer(Base):
    __tablename__ = "team_compare_list_players"
    __table_args__ = (
        UniqueConstraint(
            "compare_list_id",
            "player_id",
            "split",
            "season",
            name="uq_team_compare_list_player_scope",
        ),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid_str)
    compare_list_id: Mapped[str] = mapped_column(String(32), ForeignKey("team_compare_lists.id"), index=True)
    player_id: Mapped[str] = mapped_column(String(128), index=True)
    split: Mapped[str] = mapped_column(String(8), default="test")
    season: Mapped[str] = mapped_column(String(32), default="")
    pinned: Mapped[bool] = mapped_column(Boolean, default=False)
    notes: Mapped[str] = mapped_column(Text, default="")
    added_by_user_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class ScoutPreferenceProfile(Base):
    __tablename__ = "scout_preference_profiles"
    __table_args__ = (UniqueConstraint("workspace_id", "user_id", name="uq_scout_pref_workspace_user"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid_str)
    workspace_id: Mapped[str] = mapped_column(String(32), ForeignKey("workspaces.id"), index=True)
    user_id: Mapped[str] = mapped_column(String(32), ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(200), default="Primary")
    target_age_min: Mapped[int | None] = mapped_column(Integer, nullable=True)
    target_age_max: Mapped[int | None] = mapped_column(Integer, nullable=True)
    budget_posture: Mapped[str] = mapped_column(String(32), default="balanced")
    trusted_league_posture: Mapped[str] = mapped_column(String(32), default="balanced")
    role_priorities: Mapped[dict] = mapped_column(JSON, default=dict)
    system_template_default: Mapped[str] = mapped_column(String(64), default="")
    must_have_tags: Mapped[list] = mapped_column(JSON, default=list)
    avoid_tags: Mapped[list] = mapped_column(JSON, default=list)
    risk_tolerance: Mapped[str] = mapped_column(String(32), default="balanced")
    active_lane_preference: Mapped[str] = mapped_column(String(32), default="valuation")
    apply_by_default: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)


__all__ = [
    "ASSIGNMENT_STATUSES",
    "AuthSession",
    "Base",
    "DECISION_ACTIONS",
    "ScoutPreferenceProfile",
    "TeamActivityEvent",
    "TeamAssignment",
    "TeamComment",
    "TeamCompareList",
    "TeamCompareListPlayer",
    "TeamScoutDecision",
    "TeamWatchlistItem",
    "User",
    "WORKSPACE_ROLES",
    "Workspace",
    "WorkspaceInvite",
    "WorkspaceMembership",
]
