from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scouting_ml.providers.football_api.client import ApiFootballClient, SportmonksClient
from scouting_ml.providers.odds.client import OddsApiClient

SECTION_TO_ENV_VAR = {
    "sportmonks": "SPORTMONKS_API_TOKEN",
    "api-football": "APIFOOTBALL_API_KEY",
    "odds": "ODDS_API_KEY",
}


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


def _provider_name_for_section(section_name: str, section_cfg: dict[str, Any]) -> str:
    if section_name == "market_context":
        return "odds"
    provider = str(section_cfg.get("provider") or "").strip().lower()
    if provider not in {"sportmonks", "api-football"}:
        raise ValueError(
            f"Section {section_name} requires provider 'sportmonks' or 'api-football'."
        )
    return provider


def _required_env_var(provider: str) -> str:
    env_name = SECTION_TO_ENV_VAR.get(provider)
    if not env_name:
        raise ValueError(f"Unsupported provider: {provider}")
    return env_name


def _client_fetch(
    *,
    provider: str,
    endpoint: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> Any:
    if provider == "sportmonks":
        return SportmonksClient().get_json(endpoint, params=params, headers=headers)
    if provider == "api-football":
        return ApiFootballClient().get_json(endpoint, params=params, headers=headers)
    if provider == "odds":
        return OddsApiClient().get_json(endpoint, params=params)
    raise ValueError(f"Unsupported provider: {provider}")


def _resolve_output_path(
    *,
    section_name: str,
    request_cfg: dict[str, Any],
    request_name: str,
    raw_output_dir: Path,
    snapshot_date: str,
) -> Path:
    explicit = str(request_cfg.get("output_json") or "").strip()
    if explicit:
        return Path(explicit)
    filename = f"{section_name}_{_slugify(request_name)}_{snapshot_date}.json"
    return raw_output_dir / filename


def _request_list(section_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    requests = section_cfg.get("requests")
    if isinstance(requests, list):
        return [dict(item) for item in requests if isinstance(item, dict)]

    # Backward-compatible single-request shape.
    single = {
        "name": section_cfg.get("name") or section_cfg.get("provider") or "snapshot",
        "api_url": section_cfg.get("api_url"),
        "endpoint": section_cfg.get("endpoint"),
        "params": section_cfg.get("params"),
        "headers": section_cfg.get("headers"),
        "output_json": section_cfg.get("output_json"),
    }
    if any(single.get(key) for key in ("api_url", "endpoint", "params", "headers", "output_json")):
        return [single]
    return []


def _fetch_request(
    *,
    section_name: str,
    section_cfg: dict[str, Any],
    request_cfg: dict[str, Any],
    raw_output_dir: Path,
    snapshot_date: str,
    dry_run: bool,
    allow_missing_secrets: bool,
) -> dict[str, Any]:
    provider = _provider_name_for_section(section_name, section_cfg)
    request_name = str(request_cfg.get("name") or request_cfg.get("endpoint") or request_cfg.get("api_url") or "snapshot")
    endpoint = str(request_cfg.get("api_url") or request_cfg.get("endpoint") or "").strip()
    if not endpoint:
        raise ValueError(f"Section {section_name} request '{request_name}' is missing api_url/endpoint.")

    params = _dict_value(request_cfg.get("params"))
    headers = {str(k): str(v) for k, v in _dict_value(request_cfg.get("headers")).items()}
    output_path = _resolve_output_path(
        section_name=section_name,
        request_cfg=request_cfg,
        request_name=request_name,
        raw_output_dir=raw_output_dir,
        snapshot_date=snapshot_date,
    )
    env_var = _required_env_var(provider)
    has_secret = bool(os.getenv(env_var, "").strip())

    result = {
        "name": request_name,
        "provider": provider,
        "endpoint": endpoint,
        "params": params,
        "headers": headers,
        "output_json": str(output_path),
        "required_env_var": env_var,
        "status": "planned" if dry_run else "pending",
    }
    if dry_run:
        result["has_secret"] = has_secret
        return result
    if not has_secret:
        if allow_missing_secrets:
            result["status"] = "skipped_missing_secret"
            result["warning"] = f"Missing required environment variable {env_var}"
            return result
        raise RuntimeError(
            f"Section {section_name} request '{request_name}' requires {env_var} to be set."
        )

    payload = _client_fetch(provider=provider, endpoint=endpoint, params=params, headers=headers)
    _write_json(output_path, payload)
    result["status"] = "fetched"
    result["bytes_written"] = output_path.stat().st_size
    return result


def _section_pipeline_config(
    *,
    section_name: str,
    section_cfg: dict[str, Any],
    fetched: list[dict[str, Any]],
) -> dict[str, Any] | None:
    existing_json = _string_list(section_cfg.get("input_json"))
    fetched_json = [
        str(item["output_json"])
        for item in fetched
        if item.get("status") == "fetched"
    ]
    if not existing_json and not fetched_json:
        return None

    out: dict[str, Any] = {}
    if section_name in {"fixture_context", "player_availability"}:
        out["provider"] = section_cfg.get("provider")
    if section_name == "market_context":
        out["league"] = section_cfg.get("league")
        out["season"] = section_cfg.get("season")
    out["input_json"] = existing_json + fetched_json
    out["api_url"] = []
    return out


def collect_provider_snapshots(
    *,
    config_path: str,
    raw_output_dir: str | None = None,
    provider_config_out: str | None = None,
    summary_out: str | None = None,
    dry_run: bool = False,
    allow_missing_secrets: bool = False,
) -> dict[str, Any]:
    cfg_path = Path(config_path)
    payload = _read_json(cfg_path)
    snapshot_date = str(
        payload.get("snapshot_date")
        or datetime.now(timezone.utc).date().isoformat()
    )
    raw_dir = Path(raw_output_dir or payload.get("raw_output_dir") or "data/raw/providers")
    provider_cfg_out = Path(
        provider_config_out
        or payload.get("provider_config_out")
        or (raw_dir / f"provider_pipeline_{snapshot_date}.generated.json")
    )
    summary_path = Path(
        summary_out
        or payload.get("summary_out")
        or provider_cfg_out.with_suffix(".summary.json")
    )

    summary: dict[str, Any] = {
        "config_path": str(cfg_path),
        "snapshot_date": snapshot_date,
        "raw_output_dir": str(raw_dir),
        "provider_config_out": str(provider_cfg_out),
        "summary_out": str(summary_path),
        "dry_run": bool(dry_run),
        "allow_missing_secrets": bool(allow_missing_secrets),
        "sections": {},
        "warnings": [],
    }

    generated_provider_cfg: dict[str, Any] = {
        "snapshot_date": snapshot_date,
        "player_links": payload.get("player_links", "data/external/player_provider_links.csv"),
        "club_links": payload.get("club_links", "data/external/club_provider_links.csv"),
    }
    if isinstance(payload.get("statsbomb"), dict):
        generated_provider_cfg["statsbomb"] = dict(payload["statsbomb"])

    for section_name in ("fixture_context", "player_availability", "market_context"):
        section_cfg = _dict_value(payload.get(section_name))
        if not section_cfg:
            continue

        requests = _request_list(section_cfg)
        section_results: list[dict[str, Any]] = []
        for request_cfg in requests:
            section_results.append(
                _fetch_request(
                    section_name=section_name,
                    section_cfg=section_cfg,
                    request_cfg=request_cfg,
                    raw_output_dir=raw_dir,
                    snapshot_date=snapshot_date,
                    dry_run=dry_run,
                    allow_missing_secrets=allow_missing_secrets,
                )
            )

        section_summary = {
            "provider": _provider_name_for_section(section_name, section_cfg),
            "requests": section_results,
        }
        section_pipeline_cfg = _section_pipeline_config(
            section_name=section_name,
            section_cfg=section_cfg,
            fetched=section_results,
        )
        if section_pipeline_cfg:
            generated_provider_cfg[section_name] = section_pipeline_cfg
            section_summary["status"] = "ready"
        elif requests:
            statuses = {str(item.get("status")) for item in section_results}
            if statuses == {"planned"}:
                section_summary["status"] = "planned"
            elif statuses == {"skipped_missing_secret"}:
                section_summary["status"] = "skipped_missing_secret"
                summary["warnings"].append(
                    f"Section {section_name} skipped because required secrets were missing."
                )
            else:
                section_summary["status"] = "no_snapshots"
        else:
            section_summary["status"] = "no_requests"
            summary["warnings"].append(
                f"Section {section_name} has no requests configured."
            )
        summary["sections"][section_name] = section_summary

    if not dry_run:
        _write_json(provider_cfg_out, generated_provider_cfg)
    summary["generated_provider_config"] = generated_provider_cfg
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch raw provider snapshots and emit a generated provider pipeline config."
    )
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--raw-output-dir", default="")
    parser.add_argument("--provider-config-out", default="")
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-missing-secrets", action="store_true")
    args = parser.parse_args()

    summary = collect_provider_snapshots(
        config_path=args.config_json,
        raw_output_dir=args.raw_output_dir or None,
        provider_config_out=args.provider_config_out or None,
        summary_out=args.summary_out or None,
        dry_run=args.dry_run,
        allow_missing_secrets=args.allow_missing_secrets,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
