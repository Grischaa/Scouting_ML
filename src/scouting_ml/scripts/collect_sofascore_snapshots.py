from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scouting_ml.sofa.sofa_scraper import create_client


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _slugify(value: str) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value).strip())
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_") or "snapshot"


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def _dict_value(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _competition_list(section_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    competitions = section_cfg.get("competitions")
    if isinstance(competitions, list):
        return [dict(item) for item in competitions if isinstance(item, dict)]
    single = _dict_value(section_cfg.get("competition"))
    return [single] if single else []


def _competition_tokens(competition_cfg: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for value in (
        competition_cfg.get("name"),
        competition_cfg.get("league"),
        competition_cfg.get("season"),
    ):
        text = str(value or "").strip()
        if not text:
            continue
        lowered = text.lower()
        tokens.add(lowered)
        tokens.add(_slugify(lowered))
    return tokens


def _filter_competitions(
    competitions: list[dict[str, Any]],
    selected: list[str] | None,
) -> list[dict[str, Any]]:
    if not selected:
        return competitions
    wanted = {_slugify(str(item).strip().lower()) for item in selected if str(item).strip()}
    out: list[dict[str, Any]] = []
    for competition in competitions:
        tokens = _competition_tokens(competition)
        if wanted.intersection(tokens):
            out.append(competition)
    return out


def _override_competition_paging(
    competitions: list[dict[str, Any]],
    *,
    max_pages: int | None = None,
    team_schedule_max_pages: int | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in competitions:
        updated = dict(item)
        if max_pages is not None:
            updated["max_pages"] = int(max_pages)
        if team_schedule_max_pages is not None:
            updated["team_schedule_max_pages"] = int(team_schedule_max_pages)
        out.append(updated)
    return out


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_id_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return ""
    try:
        number = float(text)
    except (TypeError, ValueError):
        return text
    if number.is_integer():
        return str(int(number))
    return text


def _extract_events(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    events = payload.get("events")
    if isinstance(events, list):
        return [item for item in events if isinstance(item, dict)]
    return []


def _base_url(client: Any) -> str:
    cfg = getattr(client, "cfg", None)
    return str(getattr(cfg, "base_url", "") or "")


def _uses_rapidapi(client: Any) -> bool:
    return ".rapidapi." in _base_url(client)


def _extract_seasons(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    seasons = payload.get("seasons")
    if isinstance(seasons, list):
        return [item for item in seasons if isinstance(item, dict)]
    data = payload.get("data")
    if isinstance(data, dict):
        inner = data.get("seasons")
        if isinstance(inner, list):
            return [item for item in inner if isinstance(item, dict)]
    return []


def _season_values(season_payload: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key in ("name", "year", "slug"):
        value = season_payload.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            out.append(text)
    return out


def _season_aliases(value: Any) -> set[str]:
    text = str(value or "").strip().lower()
    if not text:
        return set()
    compact = re.sub(r"\s+", "", text)
    aliases = {text, compact}

    match = re.search(r"\b((?:19|20)\d{2})\s*[/_-]\s*((?:19|20)\d{2}|\d{2})\b", text)
    if match:
        start_text, end_text = match.groups()
        start_year = int(start_text)
        if len(end_text) == 2:
            end_year = (start_year // 100) * 100 + int(end_text)
        else:
            end_year = int(end_text)
        aliases.update(
            {
                start_text,
                str(end_year),
                f"{start_year}/{str(end_year)[-2:]}",
                f"{start_year}-{str(end_year)[-2:]}",
                f"{start_year}/{end_year}",
                f"{start_year}-{end_year}",
                f"{str(start_year)[-2:]}/{str(end_year)[-2:]}",
                f"{start_year}{end_year}",
            }
        )
    return {item for item in aliases if item}


def _render_params(params: dict[str, Any], **tpl_vars: Any) -> dict[str, Any]:
    rendered: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, str):
            text = value
            for var_name, var_value in tpl_vars.items():
                text = text.replace(f"{{{var_name}}}", str(var_value))
            rendered[key] = text
        else:
            rendered[key] = value
    return rendered


def _season_matches_label(candidate: Any, desired: Any) -> bool:
    desired_aliases = _season_aliases(desired)
    candidate_aliases = _season_aliases(candidate)
    return bool(desired_aliases and candidate_aliases and desired_aliases.intersection(candidate_aliases))


def _default_seasons_endpoint(client: Any) -> str:
    if _uses_rapidapi(client):
        return "/tournaments/get-seasons"
    return "/unique-tournament/{tournamentId}/seasons"


def _default_seasons_params(client: Any) -> dict[str, Any]:
    if _uses_rapidapi(client):
        return {"tournamentId": "{tournamentId}"}
    return {}


def _default_events_endpoint(client: Any) -> str:
    if _uses_rapidapi(client):
        return "/tournaments/get-last-matches"
    return "/unique-tournament/{tournamentId}/season/{seasonId}/events/{segment}/{page}"


def _default_events_params(client: Any) -> dict[str, Any]:
    if _uses_rapidapi(client):
        return {
            "tournamentId": "{tournamentId}",
            "seasonId": "{seasonId}",
            "page": "{page}",
        }
    return {}


def _default_lineups_endpoint(client: Any) -> str:
    if _uses_rapidapi(client):
        return "/matches/get-lineups"
    return "/event/{matchId}/lineups"


def _default_lineups_params(client: Any) -> dict[str, Any]:
    if _uses_rapidapi(client):
        return {"matchId": "{matchId}"}
    return {}


def _players_source_team_ids(*, players_source: str, competition_cfg: dict[str, Any]) -> list[str]:
    explicit_ids = [_normalize_id_text(item) for item in _string_list(competition_cfg.get("team_ids"))]
    explicit_ids = [item for item in explicit_ids if item]
    if explicit_ids:
        return sorted(set(explicit_ids))

    source = Path(players_source)
    files = [source] if source.is_file() else sorted(source.rglob("*_with_sofa.csv"))
    desired_season = str(competition_cfg.get("season") or "").strip()
    desired_league = str(competition_cfg.get("league") or "").strip().lower()
    team_ids: set[str] = set()
    for file in files:
        if not file.exists():
            continue
        with file.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "sofa_team_id" not in reader.fieldnames:
                continue
            for row in reader:
                team_id = _normalize_id_text(row.get("sofa_team_id"))
                if not team_id:
                    continue
                row_season = str(row.get("season") or "").strip()
                row_league = str(row.get("league") or "").strip().lower()
                if desired_season and row_season and not _season_matches_label(row_season, desired_season):
                    continue
                if desired_league and row_league and row_league != desired_league:
                    continue
                team_ids.add(team_id)
    return sorted(team_ids)


def _event_matches_competition(event: dict[str, Any], competition_cfg: dict[str, Any]) -> bool:
    tournament = event.get("tournament") or {}
    unique = tournament.get("uniqueTournament") if isinstance(tournament, dict) else {}
    event_tournament_id = unique.get("id") if isinstance(unique, dict) else None
    event_season = event.get("season") or {}
    event_season_id = event_season.get("id") if isinstance(event_season, dict) else None
    desired_tournament_id = competition_cfg.get("tournament_id")
    desired_season_id = competition_cfg.get("season_id")
    desired_season = competition_cfg.get("season")
    if event_tournament_id not in (None, "") and desired_tournament_id not in (None, ""):
        if int(event_tournament_id) != int(desired_tournament_id):
            return False
    if event_season_id not in (None, "") and desired_season_id not in (None, ""):
        if int(event_season_id) != int(desired_season_id):
            return False
    if desired_season:
        event_season_values = _season_values(event_season if isinstance(event_season, dict) else {})
        if event_season_values and not any(_season_matches_label(value, desired_season) for value in event_season_values):
            return False
    return True


def _fetch_team_schedule_events(*, client: Any, competition_cfg: dict[str, Any], players_source: str) -> dict[str, Any]:
    team_ids = _players_source_team_ids(players_source=players_source, competition_cfg=competition_cfg)
    if not team_ids:
        raise ValueError(
            f"No Sofa team ids found in players_source={players_source!r} for competition "
            f"{competition_cfg.get('name') or competition_cfg.get('league') or competition_cfg.get('tournament_id')}"
        )

    endpoint = str(
        competition_cfg.get("team_events_endpoint_template")
        or "/teams/get-matches"
    )
    params = _dict_value(competition_cfg.get("team_params")) or {
        "teamId": "{teamId}",
        "pageIndex": "{page}",
    }
    max_pages = _as_int(competition_cfg.get("team_schedule_max_pages"), 8)

    seen_ids: set[str] = set()
    events: list[dict[str, Any]] = []
    pages_fetched: list[dict[str, Any]] = []
    for team_id in team_ids:
        for page in range(max_pages):
            payload = _sofa_fetch(
                client=client,
                endpoint=endpoint,
                params=_render_params(params, teamId=team_id, page=page) or None,
                teamId=team_id,
                page=page,
            )
            raw_events = _extract_events(payload)
            matched_events = [event for event in raw_events if _event_matches_competition(event, competition_cfg)]
            pages_fetched.append(
                {
                    "team_id": team_id,
                    "page": page,
                    "events": int(len(raw_events)),
                    "matched_events": int(len(matched_events)),
                }
            )
            if not raw_events:
                break
            for event in matched_events:
                event_id = str(event.get("id") or "").strip()
                if event_id and event_id in seen_ids:
                    continue
                if event_id:
                    seen_ids.add(event_id)
                events.append(event)
            if not bool(payload.get("hasNextPage")):
                break

    return {
        "provider": "sofascore",
        "competition": {
            "name": competition_cfg.get("name"),
            "league": competition_cfg.get("league"),
            "season": competition_cfg.get("season"),
            "tournament_id": int(competition_cfg["tournament_id"]),
            "team_schedule_fallback": True,
            **(
                {"season_id": int(competition_cfg["season_id"])}
                if competition_cfg.get("season_id") not in (None, "")
                else {}
            ),
        },
        "events": events,
        "pages_fetched": pages_fetched,
    }


def _resolve_season_id(*, client: Any, competition_cfg: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
    explicit = competition_cfg.get("season_id")
    if explicit not in (None, ""):
        return int(explicit), []

    tournament_id = int(competition_cfg["tournament_id"])
    desired_label = str(
        competition_cfg.get("season_lookup")
        or competition_cfg.get("season")
        or competition_cfg.get("season_name")
        or ""
    ).strip()
    if not desired_label:
        raise ValueError(
            f"Competition {competition_cfg.get('name') or competition_cfg.get('league') or tournament_id} "
            "requires season_id or a season label for auto-discovery."
        )

    seasons_template = str(
        competition_cfg.get("seasons_endpoint_template")
        or _default_seasons_endpoint(client)
    )
    params = _dict_value(competition_cfg.get("season_lookup_params")) or _default_seasons_params(client)
    payload = _sofa_fetch(
        client=client,
        endpoint=seasons_template,
        params=_render_params(params, tournamentId=tournament_id) or None,
        tournamentId=tournament_id,
    )
    seasons = _extract_seasons(payload)
    if not seasons:
        raise ValueError(
            f"No seasons returned for tournament_id={tournament_id} from endpoint {seasons_template!r}."
        )

    desired_lower = desired_label.lower()
    desired_aliases = _season_aliases(desired_label)
    exact_matches = [
        season
        for season in seasons
        if any(str(value).strip().lower() == desired_lower for value in _season_values(season))
    ]
    if len(exact_matches) == 1:
        return int(exact_matches[0]["id"]), seasons
    if len(exact_matches) > 1:
        names = ", ".join(f"{item.get('id')}:{'/'.join(_season_values(item))}" for item in exact_matches[:5])
        raise ValueError(f"Ambiguous exact season matches for {desired_label!r}: {names}")

    alias_matches = [
        season
        for season in seasons
        if desired_aliases.intersection(
            {
                alias
                for value in _season_values(season)
                for alias in _season_aliases(value)
            }
        )
    ]
    if len(alias_matches) == 1:
        return int(alias_matches[0]["id"]), seasons
    if len(alias_matches) > 1:
        names = ", ".join(f"{item.get('id')}:{'/'.join(_season_values(item))}" for item in alias_matches[:5])
        raise ValueError(
            f"Ambiguous season matches for {desired_label!r} in tournament_id={tournament_id}: {names}"
        )

    available = ", ".join(f"{item.get('id')}:{'/'.join(_season_values(item))}" for item in seasons[:10])
    raise ValueError(
        f"Could not resolve season_id for {desired_label!r} in tournament_id={tournament_id}. "
        f"Available seasons: {available}"
    )


def _prepare_competition(*, client: Any, competition_cfg: dict[str, Any], dry_run: bool) -> dict[str, Any]:
    prepared = dict(competition_cfg)
    if dry_run or prepared.get("season_id") not in (None, ""):
        return prepared
    if str(prepared.get("events_mode") or "").strip().lower() == "team_schedule":
        # Team schedule mode can filter by league and season label directly from
        # returned events, so resolving season_id upfront just burns provider quota.
        return prepared
    season_id, seasons = _resolve_season_id(client=client, competition_cfg=prepared)
    prepared["season_id"] = season_id
    if seasons:
        prepared["_resolved_seasons_count"] = len(seasons)
    return prepared


def _sofa_fetch(*, client: Any, endpoint: str, params: dict[str, Any] | None = None, **tpl_vars: Any) -> Any:
    return client.get(endpoint, params=params, **tpl_vars)


def _fetch_competition_events(*, client: Any, competition_cfg: dict[str, Any], players_source: str | None = None) -> dict[str, Any]:
    events_mode = str(competition_cfg.get("events_mode") or "").strip().lower()
    if events_mode == "team_schedule":
        if not players_source:
            raise ValueError("team_schedule events_mode requires players_source.")
        return _fetch_team_schedule_events(
            client=client,
            competition_cfg=competition_cfg,
            players_source=players_source,
        )

    event_template = str(
        competition_cfg.get("events_endpoint_template")
        or _default_events_endpoint(client)
    )
    tournament_id = int(competition_cfg["tournament_id"])
    season_id = int(competition_cfg["season_id"])
    max_pages = _as_int(competition_cfg.get("max_pages"), 25)
    segments = _string_list(competition_cfg.get("segments")) or ["last"]
    params = _dict_value(competition_cfg.get("params")) or _default_events_params(client)

    seen_ids: set[str] = set()
    events: list[dict[str, Any]] = []
    pages_fetched: list[dict[str, Any]] = []

    for segment in segments:
        for page in range(max_pages):
            payload = _sofa_fetch(
                client=client,
                endpoint=event_template,
                params=_render_params(
                    params,
                    tournamentId=tournament_id,
                    seasonId=season_id,
                    segment=segment,
                    page=page,
                )
                or None,
                tournamentId=tournament_id,
                seasonId=season_id,
                segment=segment,
                page=page,
            )
            page_events = _extract_events(payload)
            pages_fetched.append(
                {
                    "segment": segment,
                    "page": page,
                    "events": int(len(page_events)),
                }
            )
            if not page_events:
                break
            new_count = 0
            for event in page_events:
                event_id = str(event.get("id") or "").strip()
                if event_id and event_id in seen_ids:
                    continue
                if event_id:
                    seen_ids.add(event_id)
                events.append(event)
                new_count += 1
            if new_count == 0:
                break

    return {
        "provider": "sofascore",
        "competition": {
            "name": competition_cfg.get("name"),
            "league": competition_cfg.get("league"),
            "season": competition_cfg.get("season"),
            "tournament_id": tournament_id,
            "season_id": season_id,
            "segments": segments,
        },
        "events": events,
        "pages_fetched": pages_fetched,
    }


def _load_fixture_payload_map(paths: list[str]) -> dict[tuple[str, str], dict[str, Any]]:
    payloads: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        candidate = Path(path)
        if not candidate.exists():
            continue
        try:
            payload = _read_json(candidate)
        except Exception:
            continue
        competition = payload.get("competition") if isinstance(payload, dict) else {}
        if not isinstance(competition, dict):
            continue
        key = (str(competition.get("league") or ""), str(competition.get("season") or ""))
        if key != ("", ""):
            payloads.setdefault(key, payload)
    return payloads


def _fetch_lineups_bundle(
    *,
    client: Any,
    competition_cfg: dict[str, Any],
    fixture_payload: dict[str, Any],
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    lineups_template = str(
        competition_cfg.get("lineups_endpoint_template")
        or _default_lineups_endpoint(client)
    )
    params = _dict_value(competition_cfg.get("lineup_params")) or _default_lineups_params(client)
    checkpoint_every = _as_int(competition_cfg.get("lineup_checkpoint_every"), 10)
    matches: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    completed_ids: set[str] = set()
    if checkpoint_path and checkpoint_path.exists():
        try:
            existing_payload = _read_json(checkpoint_path)
            existing_matches = existing_payload.get("matches") if isinstance(existing_payload, dict) else []
            if isinstance(existing_matches, list):
                matches = [item for item in existing_matches if isinstance(item, dict)]
                for item in matches:
                    event = item.get("event") if isinstance(item.get("event"), dict) else {}
                    event_id = str(event.get("id") or "").strip()
                    if event_id:
                        completed_ids.add(event_id)
            existing_errors = existing_payload.get("errors") if isinstance(existing_payload, dict) else []
            if isinstance(existing_errors, list):
                errors = [item for item in existing_errors if isinstance(item, dict)]
        except Exception:
            matches = []
            errors = []
            completed_ids = set()

    processed_since_checkpoint = 0
    for event in fixture_payload.get("events", []):
        if not isinstance(event, dict):
            continue
        event_id = event.get("id")
        if event_id in (None, ""):
            continue
        event_key = str(event_id).strip()
        if event_key in completed_ids:
            continue
        try:
            lineups = _sofa_fetch(
                client=client,
                endpoint=lineups_template,
                params=_render_params(params, matchId=event_id) or None,
                matchId=event_id,
            )
            matches.append({"event": event, "lineups": lineups})
            completed_ids.add(event_key)
        except Exception as exc:
            errors.append(
                {
                    "match_id": event_key,
                    "error": str(exc),
                }
            )
        processed_since_checkpoint += 1
        if checkpoint_path and processed_since_checkpoint >= checkpoint_every:
            payload = {
                "provider": "sofascore",
                "competition": dict(fixture_payload.get("competition") or {}),
                "matches": matches,
            }
            if errors:
                payload["errors"] = errors
            _write_json(checkpoint_path, payload)
            processed_since_checkpoint = 0
    payload = {
        "provider": "sofascore",
        "competition": dict(fixture_payload.get("competition") or {}),
        "matches": matches,
    }
    if errors:
        payload["errors"] = errors
    if checkpoint_path:
        _write_json(checkpoint_path, payload)
    return payload


def _resolve_output_path(
    *,
    prefix: str,
    competition_cfg: dict[str, Any],
    raw_output_dir: Path,
    snapshot_date: str,
) -> Path:
    explicit = str(competition_cfg.get("output_json") or "").strip()
    if explicit:
        return Path(explicit)
    name = str(competition_cfg.get("name") or competition_cfg.get("league") or "competition")
    filename = f"{prefix}_{_slugify(name)}_{snapshot_date}.json"
    return raw_output_dir / filename


def _section_pipeline_config(
    *,
    section_cfg: dict[str, Any],
    collected_paths: list[str],
) -> dict[str, Any] | None:
    existing = _string_list(section_cfg.get("input_json"))
    seen: set[str] = set()
    all_paths: list[str] = []
    for path in existing + collected_paths:
        text = str(path).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        all_paths.append(text)
    if not all_paths:
        return None
    out: dict[str, Any] = {
        "provider": "sofascore",
        "input_json": all_paths,
    }
    players_source = str(section_cfg.get("players_source") or "").strip()
    if players_source:
        out["players_source"] = players_source
    return out


def _existing_output_paths(
    *,
    competitions: list[dict[str, Any]],
    prefix: str,
    raw_output_dir: Path,
    snapshot_date: str,
) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for competition_cfg in competitions:
        output_path = _resolve_output_path(
            prefix=prefix,
            competition_cfg=competition_cfg,
            raw_output_dir=raw_output_dir,
            snapshot_date=snapshot_date,
        )
        output_text = str(output_path)
        if output_path.exists() and output_text not in seen:
            seen.add(output_text)
            paths.append(output_text)
    return paths


def collect_sofascore_snapshots(
    *,
    config_path: str,
    raw_output_dir: str | None = None,
    provider_config_out: str | None = None,
    summary_out: str | None = None,
    dry_run: bool = False,
    competitions: list[str] | None = None,
    rps: float | None = None,
    retries: int | None = None,
    backoff: float | None = None,
    max_pages: int | None = None,
    team_schedule_max_pages: int | None = None,
) -> dict[str, Any]:
    cfg_path = Path(config_path)
    payload = _read_json(cfg_path)
    snapshot_date = str(payload.get("snapshot_date") or datetime.now(timezone.utc).date().isoformat())
    raw_dir = Path(raw_output_dir or payload.get("raw_output_dir") or "data/raw/providers")
    provider_cfg_out = Path(
        provider_config_out
        or payload.get("provider_config_out")
        or (raw_dir / f"sofascore_provider_pipeline_{snapshot_date}.generated.json")
    )
    summary_path = Path(
        summary_out
        or payload.get("summary_out")
        or provider_cfg_out.with_suffix(".summary.json")
    )

    players_source = str(payload.get("players_source") or "").strip()
    if not players_source:
        raise ValueError("SofaScore snapshot collection requires top-level players_source.")

    effective_rps = float(rps if rps is not None else (payload.get("rps") or 2.0))
    effective_retries = _as_int(retries if retries is not None else payload.get("retries"), 4)
    effective_backoff = float(backoff if backoff is not None else (payload.get("backoff") or 0.7))

    client = None
    if not dry_run:
        client = create_client(
            base=str(payload.get("base_url") or "https://api.sofascore.com/api/v1"),
            rps=effective_rps,
            timeout=float(payload.get("timeout") or 30.0),
            retries=effective_retries,
            backoff=effective_backoff,
            http2=bool(payload.get("http2") or False),
        )

    fixture_cfg = _dict_value(payload.get("fixture_context"))
    availability_cfg = _dict_value(payload.get("player_availability"))
    fixture_competitions_all = _competition_list(fixture_cfg)
    availability_competitions_all = _competition_list(availability_cfg)
    fixture_competitions = _filter_competitions(fixture_competitions_all, competitions)
    availability_competitions = _filter_competitions(availability_competitions_all, competitions)
    fixture_competitions = _override_competition_paging(
        fixture_competitions,
        max_pages=max_pages,
        team_schedule_max_pages=team_schedule_max_pages,
    )
    availability_competitions = _override_competition_paging(
        availability_competitions,
        max_pages=max_pages,
        team_schedule_max_pages=team_schedule_max_pages,
    )
    if availability_cfg and not availability_competitions:
        availability_competitions = fixture_competitions
    if competitions and not fixture_competitions and not availability_competitions:
        wanted = ", ".join(competitions)
        raise ValueError(f"No competitions matched selection: {wanted}")

    summary: dict[str, Any] = {
        "config_path": str(cfg_path),
        "snapshot_date": snapshot_date,
        "raw_output_dir": str(raw_dir),
        "provider_config_out": str(provider_cfg_out),
        "summary_out": str(summary_path),
        "players_source": players_source,
        "dry_run": bool(dry_run),
        "runtime_overrides": {
            "competitions": list(competitions or []),
            "rps": effective_rps,
            "retries": effective_retries,
            "backoff": effective_backoff,
            "max_pages": max_pages,
            "team_schedule_max_pages": team_schedule_max_pages,
        },
        "sections": {},
        "warnings": [],
    }

    generated_provider_cfg: dict[str, Any] = {
        "snapshot_date": snapshot_date,
        "players_source": players_source,
        "player_links": payload.get("player_links", "data/external/player_provider_links.csv"),
        "club_links": payload.get("club_links", "data/external/club_provider_links.csv"),
    }
    if isinstance(payload.get("statsbomb"), dict):
        generated_provider_cfg["statsbomb"] = dict(payload["statsbomb"])

    fixture_paths: list[str] = []
    fixture_results: list[dict[str, Any]] = []
    fixture_payload_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for competition_cfg in fixture_competitions:
        prepared_cfg = _prepare_competition(client=client, competition_cfg=competition_cfg, dry_run=dry_run)
        name = str(competition_cfg.get("name") or competition_cfg.get("league") or "competition")
        output_path = _resolve_output_path(
            prefix="sofascore_fixtures",
            competition_cfg=prepared_cfg,
            raw_output_dir=raw_dir,
            snapshot_date=snapshot_date,
        )
        result = {
            "name": name,
            "output_json": str(output_path),
            "status": "planned" if dry_run else "pending",
            "league": prepared_cfg.get("league"),
            "season": prepared_cfg.get("season"),
            "tournament_id": prepared_cfg.get("tournament_id"),
            "season_id": prepared_cfg.get("season_id"),
        }
        if prepared_cfg.get("_resolved_seasons_count") not in (None, ""):
            result["resolved_from_seasons"] = int(prepared_cfg["_resolved_seasons_count"])
        if not dry_run:
            if output_path.exists():
                fixture_payload = _read_json(output_path)
                fixture_paths.append(str(output_path))
                result["status"] = "reused"
                result["events"] = int(len(fixture_payload.get("events", [])))
                result["pages"] = int(len(fixture_payload.get("pages_fetched", [])))
            else:
                fixture_payload = _fetch_competition_events(
                    client=client,
                    competition_cfg=prepared_cfg,
                    players_source=players_source,
                )
                _write_json(output_path, fixture_payload)
                fixture_paths.append(str(output_path))
                result["status"] = "fetched"
                result["events"] = int(len(fixture_payload.get("events", [])))
                result["pages"] = int(len(fixture_payload.get("pages_fetched", [])))
            key = (str(prepared_cfg.get("league") or ""), str(prepared_cfg.get("season") or ""))
            fixture_payload_by_key[key] = fixture_payload
        fixture_results.append(result)

    fixture_pipeline_cfg = _section_pipeline_config(
        section_cfg=fixture_cfg,
        collected_paths=fixture_paths
        + _existing_output_paths(
            competitions=fixture_competitions_all,
            prefix="sofascore_fixtures",
            raw_output_dir=raw_dir,
            snapshot_date=snapshot_date,
        ),
    )
    if fixture_pipeline_cfg:
        fixture_pipeline_cfg.setdefault("players_source", players_source)
        generated_provider_cfg["fixture_context"] = fixture_pipeline_cfg
    summary["sections"]["fixture_context"] = {
        "provider": "sofascore",
        "status": "planned" if dry_run and fixture_results else ("ready" if fixture_pipeline_cfg else "no_competitions"),
        "competitions": fixture_results,
    }

    availability_paths: list[str] = []
    availability_results: list[dict[str, Any]] = []
    if not dry_run:
        fixture_payload_by_key.update(
            _load_fixture_payload_map(_string_list(fixture_cfg.get("input_json")))
        )
    for competition_cfg in availability_competitions:
        prepared_cfg = _prepare_competition(client=client, competition_cfg=competition_cfg, dry_run=dry_run)
        name = str(competition_cfg.get("name") or competition_cfg.get("league") or "competition")
        output_path = _resolve_output_path(
            prefix="sofascore_lineups",
            competition_cfg=prepared_cfg,
            raw_output_dir=raw_dir,
            snapshot_date=snapshot_date,
        )
        result = {
            "name": name,
            "output_json": str(output_path),
            "status": "planned" if dry_run else "pending",
            "league": prepared_cfg.get("league"),
            "season": prepared_cfg.get("season"),
            "tournament_id": prepared_cfg.get("tournament_id"),
            "season_id": prepared_cfg.get("season_id"),
        }
        if prepared_cfg.get("_resolved_seasons_count") not in (None, ""):
            result["resolved_from_seasons"] = int(prepared_cfg["_resolved_seasons_count"])
        if not dry_run:
            key = (str(prepared_cfg.get("league") or ""), str(prepared_cfg.get("season") or ""))
            fixture_payload = fixture_payload_by_key.get(key)
            if fixture_payload is None:
                fixture_payload = _fetch_competition_events(
                    client=client,
                    competition_cfg=prepared_cfg,
                    players_source=players_source,
                )
            lineups_payload = _fetch_lineups_bundle(
                client=client,
                competition_cfg=prepared_cfg,
                fixture_payload=fixture_payload,
                checkpoint_path=output_path,
            )
            availability_paths.append(str(output_path))
            result["status"] = "fetched"
            result["matches"] = int(len(lineups_payload.get("matches", [])))
            if lineups_payload.get("errors"):
                result["errors"] = int(len(lineups_payload.get("errors", [])))
                summary["warnings"].append(
                    f"Availability collection recorded {result['errors']} lineup fetch errors for {name}."
                )
        availability_results.append(result)

    availability_pipeline_cfg = _section_pipeline_config(
        section_cfg=availability_cfg,
        collected_paths=availability_paths
        + _existing_output_paths(
            competitions=availability_competitions_all,
            prefix="sofascore_lineups",
            raw_output_dir=raw_dir,
            snapshot_date=snapshot_date,
        ),
    )
    if availability_pipeline_cfg:
        availability_pipeline_cfg.setdefault("players_source", players_source)
        generated_provider_cfg["player_availability"] = availability_pipeline_cfg
    summary["sections"]["player_availability"] = {
        "provider": "sofascore",
        "status": "planned" if dry_run and availability_results else ("ready" if availability_pipeline_cfg else "no_competitions"),
        "competitions": availability_results,
    }

    if not dry_run:
        _write_json(provider_cfg_out, generated_provider_cfg)
    summary["generated_provider_config"] = generated_provider_cfg
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch SofaScore season fixtures and lineups, then emit a generated provider pipeline config."
    )
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--raw-output-dir", default="")
    parser.add_argument("--provider-config-out", default="")
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--competitions",
        default="",
        help="Optional comma-separated competition names/leagues to collect, e.g. scottish_premiership_2024_25",
    )
    parser.add_argument("--rps", type=float, default=None)
    parser.add_argument("--retries", type=int, default=None)
    parser.add_argument("--backoff", type=float, default=None)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--team-schedule-max-pages", type=int, default=None)
    args = parser.parse_args()

    payload = collect_sofascore_snapshots(
        config_path=args.config_json,
        raw_output_dir=args.raw_output_dir or None,
        provider_config_out=args.provider_config_out or None,
        summary_out=args.summary_out or None,
        dry_run=args.dry_run,
        competitions=[item.strip() for item in str(args.competitions).split(",") if item.strip()] or None,
        rps=args.rps,
        retries=args.retries,
        backoff=args.backoff,
        max_pages=args.max_pages,
        team_schedule_max_pages=args.team_schedule_max_pages,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
