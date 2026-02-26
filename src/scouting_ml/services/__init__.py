"""Service layer entrypoints with lazy imports."""


def get_scouting_report(*args, **kwargs):
    from .scouting_report_service import get_scouting_report as _get_scouting_report

    return _get_scouting_report(*args, **kwargs)


def find_similar_players(*args, **kwargs):
    from .player_similarity_service import find_similar_players as _find_similar_players

    return _find_similar_players(*args, **kwargs)


__all__ = ["find_similar_players", "get_scouting_report"]
