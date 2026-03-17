"""Helpers for interacting with the Sofascore API and related utilities."""

def pull_league_stats() -> None:
    # `league_pull` depends on ScraperFC, which is intentionally optional in the
    # lightweight runtime used for tests and provider pipelines.
    from .league_pull import main as _pull_league_stats

    _pull_league_stats()


def probe_tvchannels(*args, **kwargs):
    from .sofa_rapid import probe_tvchannels as _probe_tvchannels

    return _probe_tvchannels(*args, **kwargs)

from .sofa_parser import parse_all_players
from .sofa_scraper import (
    ENDPOINT_SHORTCUTS,
    SofaClient,
    SofaConfig,
    app as sofa_app,
    create_client,
    fetch as fetch_endpoint,
    player_harvest,
    run_config,
    search_player,
    team_harvest,
)

__all__ = [
    "ENDPOINT_SHORTCUTS",
    "SofaClient",
    "SofaConfig",
    "sofa_app",
    "create_client",
    "search_player",
    "fetch_endpoint",
    "player_harvest",
    "team_harvest",
    "run_config",
    "parse_all_players",
    "probe_tvchannels",
    "pull_league_stats",
]
