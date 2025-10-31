# src/scouting_ml/sofa_scraper.py
from __future__ import annotations

import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from scouting_ml.utils.import_guard import *  # noqa: F403
import httpx
import typer
import yaml

from scouting_ml.paths import ensure_dirs, sofa_json, RAW_DIR
from scouting_ml.logging import get_logger

app = typer.Typer(add_completion=False, help="Sofascore scraper (robust, configurable)")

logger = get_logger()

# ------------------------------
# Constants / Defaults
# ------------------------------

DEFAULT_BASE = "https://api.sofascore.com/api/v1"
DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
# Some endpoints you’ll likely use often (can be extended freely)
ENDPOINT_SHORTCUTS = {
    # Search
    "search_player": "/search/all?q={query}&sports=football",  # returns players, teams, etc.
    # Player
    "player_info": "/player/{playerId}",
    "player_seasons": "/player/{playerId}/unique-tournament/seasons",  # available seasons by competition
    "player_season_stats": "/player/{playerId}/season/{seasonId}",
    "player_transfers": "/player/{playerId}/transfers",
    # Team
    "team_info": "/team/{teamId}",
    "team_season_players": "/team/{teamId}/season/{seasonId}",
    "team_season_stats": "/team/{teamId}/unique-tournament/{tournamentId}/season/{seasonId}",
    # Competition
    "tournament_info": "/unique-tournament/{tournamentId}",
    "tournament_seasons": "/unique-tournament/{tournamentId}/seasons",
    "season_standings": "/unique-tournament/{tournamentId}/season/{seasonId}/standings",
    # Match
    "match_info": "/event/{matchId}",
    "match_lineups": "/event/{matchId}/lineups",
    "match_statistics": "/event/{matchId}/statistics",
    "match_shotmap": "/event/{matchId}/shotmap",
}

# Throttle defaults
DEFAULT_RATE_LIMIT_RPS = 2.0  # ~2 requests/second
DEFAULT_MAX_RETRIES = 4
DEFAULT_BACKOFF = 0.7  # seconds base, + jitter

# Where we put Sofascore raw JSONs
SOFA_ROOT = RAW_DIR / "sofascore"  # …/data/raw/sofascore   (from paths.py)
# Structured subfolders
SUBDIRS = {
    "player": SOFA_ROOT / "players",
    "team": SOFA_ROOT / "teams",
    "match": SOFA_ROOT / "matches",
    "tournament": SOFA_ROOT / "tournaments",
    "search": SOFA_ROOT / "search",
    "misc": SOFA_ROOT / "misc",
}

for p in SUBDIRS.values():
    p.mkdir(parents=True, exist_ok=True)


# ------------------------------
# Helpers
# ------------------------------

def _safe_slug(s: str, maxlen: int = 140) -> str:
    s = re.sub(r"\s+", "_", str(s)).strip("_")
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "-", s)
    return s[:maxlen] or "file"

def _now_ms() -> int:
    return int(time.time() * 1000)

def _sleep_for_ratelimit(last_call_ms: int, rps: float) -> int:
    """Return milliseconds slept."""
    if rps <= 0:
        return 0
    min_interval = 1.0 / rps
    elapsed = (time.time() - (last_call_ms / 1000.0))
    if elapsed < min_interval:
        to_sleep = min_interval - elapsed
        time.sleep(to_sleep)
        return int(to_sleep * 1000)
    return 0

def _expand_template(path: str, **kw: Any) -> str:
    # Accept both {name} and {{name}} styles
    path = re.sub(r"\{\{(\w+)\}\}", r"{\1}", path)
    return path.format(**kw)

def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


# ------------------------------
# HTTP Client
# ------------------------------

@dataclass
class SofaConfig:
    base_url: str = DEFAULT_BASE
    rps: float = DEFAULT_RATE_LIMIT_RPS
    timeout: float = 30.0
    retries: int = DEFAULT_MAX_RETRIES
    backoff: float = DEFAULT_BACKOFF
    http2: bool = False
    follow_redirects: bool = True
    headers: Optional[Dict[str, str]] = None

class SofaClient:
    """Reusable Sofascore API client with retries, throttle, and convenience helpers."""
    def __init__(self, cfg: SofaConfig):
        self.cfg = cfg
        self._client = httpx.Client(
            base_url=cfg.base_url,
            headers={
                "User-Agent": DEFAULT_UA,
                "Accept": "application/json, */*;q=0.1",
                "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
                **(cfg.headers or {}),
            },
            http2=cfg.http2,
            follow_redirects=cfg.follow_redirects,
            timeout=httpx.Timeout(cfg.timeout, connect=12.0, read=cfg.timeout),
        )
        self._last_call_ms = 0

    def _request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # rate limit
        slept = _sleep_for_ratelimit(self._last_call_ms, self.cfg.rps)
        if slept:
            logger.debug(f"[sofa] rate-limit sleep {slept} ms")

        # retry loop
        last_exc: Optional[BaseException] = None
        for attempt in range(1, self.cfg.retries + 1):
            try:
                self._last_call_ms = _now_ms()
                r = self._client.request(method, path, params=params)
                if r.status_code in (429, 503):
                    # backoff + jitter
                    delay = self.cfg.backoff * attempt + random.uniform(0.05, 0.35)
                    logger.warning(f"[sofa] {r.status_code} on {path}; retrying in {delay:.2f}s")
                    time.sleep(delay)
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_exc = e
                delay = self.cfg.backoff * attempt + random.uniform(0.05, 0.35)
                logger.warning(f"[sofa] attempt {attempt}/{self.cfg.retries} failed: {e} -> sleep {delay:.2f}s")
                time.sleep(delay)
        # exhausted
        if last_exc:
            raise last_exc
        raise RuntimeError("Unknown sofa client failure")

    # ---- High-level convenience ----

    def get(self, api_path: str, **tpl_vars: Any) -> Dict[str, Any]:
        path = _expand_template(api_path, **tpl_vars)
        return self._request("GET", path)

    # Shortcut wrappers for common tasks
    def search_player(self, query: str) -> Dict[str, Any]:
        return self.get(ENDPOINT_SHORTCUTS["search_player"], query=query)

    def player_info(self, player_id: int) -> Dict[str, Any]:
        return self.get(ENDPOINT_SHORTCUTS["player_info"], playerId=player_id)

    def player_seasons(self, player_id: int) -> Dict[str, Any]:
        return self.get(ENDPOINT_SHORTCUTS["player_seasons"], playerId=player_id)

    def player_season_stats(self, player_id: int, season_id: int) -> Dict[str, Any]:
        return self.get(ENDPOINT_SHORTCUTS["player_season_stats"], playerId=player_id, seasonId=season_id)

    def team_info(self, team_id: int) -> Dict[str, Any]:
        return self.get(ENDPOINT_SHORTCUTS["team_info"], teamId=team_id)

    def team_season_players(self, team_id: int, season_id: int) -> Dict[str, Any]:
        return self.get(ENDPOINT_SHORTCUTS["team_season_players"], teamId=team_id, seasonId=season_id)


# ------------------------------
# Disk Layout Helpers
# ------------------------------

def _out_path_search(q: str) -> Path:
    return SUBDIRS["search"] / f"players_{_safe_slug(q)}.json"

def _out_path_player(player_id: int, suffix: str) -> Path:
    return SUBDIRS["player"] / f"{int(player_id)}" / f"{suffix}.json"

def _out_path_team(team_id: int, suffix: str) -> Path:
    return SUBDIRS["team"] / f"{int(team_id)}" / f"{suffix}.json"

def _out_path_match(match_id: int, suffix: str) -> Path:
    return SUBDIRS["match"] / f"{int(match_id)}" / f"{suffix}.json"

def _out_path_misc(name: str) -> Path:
    return SUBDIRS["misc"] / f"{_safe_slug(name)}.json"


# ------------------------------
# CLI Commands
# ------------------------------

def _mk_client(
    base: str = DEFAULT_BASE,
    rps: float = DEFAULT_RATE_LIMIT_RPS,
    timeout: float = 30.0,
    retries: int = DEFAULT_MAX_RETRIES,
    backoff: float = DEFAULT_BACKOFF,
    http2: bool = False,
) -> SofaClient:
    cfg = SofaConfig(base_url=base, rps=rps, timeout=timeout, retries=retries, backoff=backoff, http2=http2)
    return SofaClient(cfg)

@app.command()
def search_player(
    query: str = typer.Argument(..., help="Player name search string"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Re-download even if file exists"),
    base: str = typer.Option(DEFAULT_BASE, "--base", help="Base API URL"),
    rps: float = typer.Option(DEFAULT_RATE_LIMIT_RPS, "--rps", help="Requests per second"),
    http2: bool = typer.Option(False, "--http2/--no-http2", help="Use HTTP/2 if available"),
):
    """Search Sofascore for players."""
    ensure_dirs()
    client = _mk_client(base=base, rps=rps, http2=http2)
    out = _out_path_search(query)
    if out.exists() and not overwrite:
        typer.echo(f"[sofa] Exists -> {out}")
        raise typer.Exit(0)
    data = client.search_player(query)
    _write_json(out, data)
    typer.echo(f"[sofa] Wrote {out}")

@app.command()
def fetch(
    path: str = typer.Argument(..., help="API path (e.g. /player/{playerId}/season/{seasonId})"),
    params: List[str] = typer.Option([], "--param", help="Key=Value query params"),
    vars: List[str] = typer.Option([], "--var", help="Template var key=val for {placeholders}"),
    out: Optional[str] = typer.Option(None, "--out", help="Output filename under data/raw/sofascore"),
    base: str = typer.Option(DEFAULT_BASE, "--base", help="Base API URL"),
    rps: float = typer.Option(DEFAULT_RATE_LIMIT_RPS, "--rps"),
    http2: bool = typer.Option(False, "--http2/--no-http2"),
    overwrite: bool = typer.Option(False, "--overwrite"),
):
    """
    Fetch any Sofascore endpoint. Templating supported via --var key=val.

    Example:
      python -m scouting_ml.sofa_scraper fetch "/player/{playerId}/season/{seasonId}" --var playerId=123 --var seasonId=54186 --out players/123/season_54186.json
    """
    ensure_dirs()
    client = _mk_client(base=base, rps=rps, http2=http2)

    # parse params & vars
    q: Dict[str, Any] = {}
    for kv in params:
        k, v = kv.split("=", 1)
        q[k] = v
    tpl: Dict[str, Any] = {}
    for kv in vars:
        k, v = kv.split("=", 1)
        tpl[k] = v

    data = client.get(path, **tpl)
    # destination
    if out:
        dest = SOFA_ROOT / out
    else:
        # infer a sensible name
        stem = _safe_slug(re.sub(r"^/+", "", _expand_template(path, **tpl)).replace("/", "__"))
        dest = _out_path_misc(stem)
    if dest.exists() and not overwrite:
        typer.echo(f"[sofa] Exists -> {dest}")
        raise typer.Exit(0)
    _write_json(dest, data)
    typer.echo(f"[sofa] Wrote {dest}")

@app.command()
def player_harvest(
    player_id: int = typer.Argument(..., help="Sofascore playerId"),
    include_transfers: bool = typer.Option(True, "--transfers/--no-transfers"),
    base: str = typer.Option(DEFAULT_BASE, "--base"),
    rps: float = typer.Option(DEFAULT_RATE_LIMIT_RPS, "--rps"),
    http2: bool = typer.Option(False, "--http2/--no-http2"),
    overwrite: bool = typer.Option(False, "--overwrite"),
):
    """
    Harvest a player's canonical bundle:
      - player_info.json
      - seasons.json (available seasons per competition)
      - season_<id>.json for each listed season
      - transfers.json (optional)
    """
    ensure_dirs()
    client = _mk_client(base=base, rps=rps, http2=http2)

    # info
    info_path = _out_path_player(player_id, "player_info")
    if not info_path.exists() or overwrite:
        info = client.player_info(player_id)
        _write_json(info_path, info)
        typer.echo(f"[sofa] Wrote {info_path}")
    else:
        typer.echo(f"[sofa] Exists -> {info_path}")

    # seasons
    seasons_path = _out_path_player(player_id, "seasons")
    if not seasons_path.exists() or overwrite:
        seasons = client.player_seasons(player_id)
        _write_json(seasons_path, seasons)
        typer.echo(f"[sofa] Wrote {seasons_path}")
    else:
        with open(seasons_path, "r", encoding="utf-8") as fh:
            seasons = json.load(fh)
        typer.echo(f"[sofa] Loaded cached seasons")

    # iterate seasons
    # Sofascore often nests as {"seasons":[{"season":{"id":54186, "name":"2023/24"}, ...}]}
    season_ids: List[int] = []
    try:
        for row in seasons.get("seasons", []):
            sid = row.get("season", {}).get("id")
            if isinstance(sid, int):
                season_ids.append(sid)
    except Exception:
        pass

    for sid in sorted(set(season_ids)):
        outp = _out_path_player(player_id, f"season_{sid}")
        if outp.exists() and not overwrite:
            typer.echo(f"[sofa] Exists -> {outp}")
            continue
        data = client.player_season_stats(player_id, sid)
        _write_json(outp, data)
        typer.echo(f"[sofa] Wrote {outp}")

    if include_transfers:
        transfers_path = _out_path_player(player_id, "transfers")
        if not transfers_path.exists() or overwrite:
            tr = client.get(ENDPOINT_SHORTCUTS["player_transfers"], playerId=player_id)
            _write_json(transfers_path, tr)
            typer.echo(f"[sofa] Wrote {transfers_path}")
        else:
            typer.echo(f"[sofa] Exists -> {transfers_path}")

@app.command()
def team_harvest(
    team_id: int = typer.Argument(..., help="Sofascore teamId"),
    season_id: int = typer.Argument(..., help="Sofascore seasonId (use tournament seasons endpoint to discover)"),
    roster_only: bool = typer.Option(False, "--roster-only", help="Only save team roster"),
    base: str = typer.Option(DEFAULT_BASE, "--base"),
    rps: float = typer.Option(DEFAULT_RATE_LIMIT_RPS, "--rps"),
    http2: bool = typer.Option(False, "--http2/--no-http2"),
    overwrite: bool = typer.Option(False, "--overwrite"),
):
    """
    Harvest a team season bundle:
      - team_info.json
      - players_season_<seasonId>.json (squad with playerIds)
      - (optional) stats for each player season
    """
    ensure_dirs()
    client = _mk_client(base=base, rps=rps, http2=http2)

    # team info
    tinfo = _out_path_team(team_id, "team_info")
    if not tinfo.exists() or overwrite:
        info = client.team_info(team_id)
        _write_json(tinfo, info)
        typer.echo(f"[sofa] Wrote {tinfo}")
    else:
        typer.echo(f"[sofa] Exists -> {tinfo}")

    # roster
    roster_path = _out_path_team(team_id, f"players_season_{season_id}")
    if not roster_path.exists() or overwrite:
        roster = client.team_season_players(team_id, season_id)
        _write_json(roster_path, roster)
        typer.echo(f"[sofa] Wrote {roster_path}")
    else:
        with open(roster_path, "r", encoding="utf-8") as fh:
            roster = json.load(fh)
        typer.echo(f"[sofa] Loaded cached roster")

    if roster_only:
        return

    # Extract playerIds and fetch their season stats
    ids: List[int] = []
    try:
        # typical structure: {"players": [{"player": {"id": 123, "name": ...}}, ...]}
        for row in roster.get("players", []):
            pid = row.get("player", {}).get("id")
            if isinstance(pid, int):
                ids.append(pid)
    except Exception:
        pass

    for pid in sorted(set(ids)):
        outp = _out_path_player(pid, f"season_{season_id}")
        if outp.exists() and not overwrite:
            typer.echo(f"[sofa] Exists -> {outp}")
            continue
        data = client.player_season_stats(pid, season_id)
        _write_json(outp, data)
        typer.echo(f"[sofa] Wrote {outp}")

@app.command()
def run_config(
    config: Path = typer.Argument(..., help="YAML file defining a batch of endpoints to fetch"),
    base: str = typer.Option(DEFAULT_BASE, "--base"),
    rps: float = typer.Option(DEFAULT_RATE_LIMIT_RPS, "--rps"),
    http2: bool = typer.Option(False, "--http2/--no-http2"),
    overwrite: bool = typer.Option(False, "--overwrite"),
):
    """
    Run a YAML-defined batch job. Each item supports:
      - name: label
      - path: endpoint template (e.g. /player/{playerId}/season/{seasonId})
      - vars: dict of template vars
      - out: relative path under data/raw/sofascore (e.g. players/123/season_54186.json)
      - params: optional query params map
    """
    ensure_dirs()
    client = _mk_client(base=base, rps=rps, http2=http2)

    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    jobs: List[Dict[str, Any]] = cfg if isinstance(cfg, list) else cfg.get("jobs", [])

    for job in jobs:
        name = job.get("name") or job.get("out") or job.get("path")
        path = job["path"]
        vars_ = job.get("vars", {})
        params = job.get("params", {})
        out_rel = job.get("out")

        try:
            data = client.get(path, **vars_)
            if out_rel:
                dest = SOFA_ROOT / out_rel
            else:
                stem = _safe_slug(re.sub(r"^/+", "", _expand_template(path, **vars_)).replace("/", "__"))
                dest = _out_path_misc(stem)
            if dest.exists() and not overwrite:
                typer.echo(f"[sofa] Exists -> {dest}")
                continue
            _write_json(dest, data)
            typer.echo(f"[sofa] [{name}] -> {dest}")
        except Exception as e:
            typer.echo(f"[sofa] [{name}] FAILED: {e}")

if __name__ == "__main__":
    app()
