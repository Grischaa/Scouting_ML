"""Service layer entrypoints with lazy imports."""


def get_scouting_report(*args, **kwargs):
    """Proxy to the scouting report service with lazy import semantics."""
    from .scouting_report_service import get_scouting_report as _get_scouting_report

    return _get_scouting_report(*args, **kwargs)


def find_similar_players(*args, **kwargs):
    """Proxy to the position-aware similarity service with lazy import semantics."""
    from .similarity_service import find_similar_players as _find_similar_players

    return _find_similar_players(*args, **kwargs)


def get_player_trajectory(*args, **kwargs):
    """Proxy to the trajectory service with lazy import semantics."""
    from .trajectory_service import get_player_trajectory as _get_player_trajectory

    return _get_player_trajectory(*args, **kwargs)

__all__ = ["find_similar_players", "get_player_trajectory", "get_scouting_report"]
