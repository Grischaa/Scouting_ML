"""Create initial ScoutML Team Edition schema."""

from __future__ import annotations

from alembic import op

from scouting_ml.team.models import Base


revision = "20260325_0001"
down_revision = None
branch_labels = None
depends_on = None


def _team_tables():
    return list(Base.metadata.sorted_tables)


def upgrade() -> None:
    """Create all team-edition tables."""
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind, tables=_team_tables(), checkfirst=True)


def downgrade() -> None:
    """Drop all team-edition tables."""
    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind, tables=list(reversed(_team_tables())), checkfirst=True)
