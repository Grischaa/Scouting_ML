from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from scouting_ml.league_registry import LeagueConfig, get_league, season_slug_label
from scouting_ml.paths import PROCESSED_DIR

DEFAULT_BACKFILL_LEAGUES = (
    "dutch_eredivisie",
    "portuguese_primeira_liga",
    "belgian_pro_league",
    "turkish_super_lig",
    "greek_super_league",
    "scottish_premiership",
)


def _parse_csv_tokens(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def _artifact_meta(path: str | Path | None) -> dict[str, object] | None:
    if path in (None, ""):
        return None
    resolved = Path(path)
    exists = resolved.exists()
    return {
        "path": str(resolved.resolve()),
        "exists": exists,
        "size_bytes": int(resolved.stat().st_size) if exists else None,
    }


def _require_artifact(path: str | Path | None, label: str) -> dict[str, object] | None:
    meta = _artifact_meta(path)
    if meta is None:
        return None
    if not meta["exists"]:
        raise FileNotFoundError(f"Expected {label} artifact was not created: {path}")
    return meta


def _infer_tm_season_id(season: str) -> int:
    season_text = str(season).strip()
    if not season_text:
        raise ValueError("season is required")
    start_year = season_text.split("/", 1)[0].strip()
    return int(start_year)


def _infer_sofa_season_label(season: str) -> str:
    season_text = str(season).strip()
    if "/" not in season_text:
        return season_text
    start_text, end_text = season_text.split("/", 1)
    start_year = int(start_text)
    end_text = end_text.strip()
    if len(end_text) == 2:
        end_year = end_text
    else:
        end_year = str(int(end_text))[-2:]
    return f"{str(start_year)[-2:]}/{end_year}"


def _extend_league_for_season(config: LeagueConfig, season: str) -> LeagueConfig:
    if season in config.tm_season_ids and season in config.sofa_season_map:
        if season in config.seasons:
            return config
        return replace(config, seasons=[season, *config.seasons])

    seasons = [season, *[item for item in config.seasons if item != season]]
    tm_season_ids = dict(config.tm_season_ids)
    tm_season_ids.setdefault(season, _infer_tm_season_id(season))
    sofa_season_map = dict(config.sofa_season_map)
    sofa_season_map.setdefault(season, _infer_sofa_season_label(season))
    return replace(
        config,
        seasons=seasons,
        tm_season_ids=tm_season_ids,
        sofa_season_map=sofa_season_map,
    )


def _run_transfermarkt_pipeline(
    config: LeagueConfig,
    season: str,
    *,
    force: bool,
    python_executable: str | None,
):
    from scouting_ml.pipeline.tm import run_transfermarkt

    return run_transfermarkt(
        config,
        season,
        force=force,
        python_executable=python_executable,
    )


def _run_sofascore_pipeline(
    config: LeagueConfig,
    season: str,
    *,
    force: bool,
    python_executable: str | None,
) -> Path:
    from scouting_ml.pipeline.sofa import run_sofascore

    return run_sofascore(
        config,
        season,
        force=force,
        python_executable=python_executable,
    )


def _merge_tm_sofa_pipeline(
    config: LeagueConfig,
    season: str,
    *,
    tm_clean_path: Path,
    sofa_path: Path,
    force: bool,
) -> Path:
    from scouting_ml.pipeline.merge import merge_tm_sofa

    return merge_tm_sofa(
        config,
        season,
        tm_clean_path=tm_clean_path,
        sofa_path=sofa_path,
        force=force,
    )


def _merged_output_summary(path: Path) -> dict[str, object]:
    row_count = 0
    matched_count = 0
    if not path.exists():
        return {
            "rows": 0,
            "sofa_matched_rows": 0,
            "sofa_matched_share": None,
        }
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_count += 1
            matched_value = str(row.get("sofa_matched", "")).strip().lower()
            if matched_value in {"1", "true", "yes"}:
                matched_count += 1
    return {
        "rows": row_count,
        "sofa_matched_rows": matched_count,
        "sofa_matched_share": (matched_count / row_count) if row_count else None,
    }


def run_future_league_backfill(
    *,
    leagues: Sequence[str],
    season: str,
    import_dir: str,
    python_executable: str | None = None,
    force: bool = False,
    skip_transfermarkt: bool = False,
    skip_sofascore: bool = False,
    skip_merge: bool = False,
    summary_json: str | None = None,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    import_root = Path(import_dir)
    import_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    success_count = 0
    error_count = 0

    for league_slug in leagues:
        config = _extend_league_for_season(get_league(league_slug), season)
        season_slug = season_slug_label(season)
        tm_combined_path = PROCESSED_DIR / f"{config.slug}_{season_slug}_players.csv"
        tm_clean_path = PROCESSED_DIR / f"{config.slug}_{season_slug}_clean.csv"
        sofa_path = PROCESSED_DIR / f"sofa_{config.slug}_{season_slug}.csv"
        merged_path = config.guess_processed_dataset(season)
        staged_path = import_root / merged_path.name

        row: dict[str, Any] = {
            "league_slug": config.slug,
            "league_name": config.name,
            "season": season,
            "status": "pending",
            "inputs": {
                "tm_season_id": int(config.tm_season_ids[season]),
                "sofa_season_label": str(config.sofa_season_map[season]),
            },
            "artifacts": {},
            "merge_summary": None,
            "error": None,
        }

        try:
            if not skip_transfermarkt:
                _run_transfermarkt_pipeline(
                    config,
                    season,
                    force=force,
                    python_executable=python_executable,
                )
            if not skip_sofascore:
                sofa_path = _run_sofascore_pipeline(
                    config,
                    season,
                    force=force,
                    python_executable=python_executable,
                )
            if not skip_merge:
                merged_path = _merge_tm_sofa_pipeline(
                    config,
                    season,
                    tm_clean_path=tm_clean_path,
                    sofa_path=sofa_path,
                    force=force,
                )

            if not skip_merge:
                shutil.copy2(merged_path, staged_path)
                row["merge_summary"] = _merged_output_summary(staged_path)

            row["artifacts"] = {
                "tm_combined": _artifact_meta(tm_combined_path),
                "tm_clean": _artifact_meta(tm_clean_path),
                "sofa_csv": _artifact_meta(sofa_path),
                "merged_csv": _artifact_meta(merged_path),
                "staged_import_csv": _artifact_meta(staged_path) if not skip_merge else _artifact_meta(None),
            }
            row["status"] = "ok"
            success_count += 1
        except Exception as exc:  # noqa: BLE001
            row["status"] = "error"
            row["error"] = str(exc)
            row["artifacts"] = {
                "tm_combined": _artifact_meta(tm_combined_path),
                "tm_clean": _artifact_meta(tm_clean_path),
                "sofa_csv": _artifact_meta(sofa_path),
                "merged_csv": _artifact_meta(merged_path),
                "staged_import_csv": _artifact_meta(staged_path),
            }
            error_count += 1
        rows.append(row)

    finished_at = datetime.now(timezone.utc)
    summary: dict[str, Any] = {
        "generated_at_utc": finished_at.isoformat(),
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "status": "ok" if error_count == 0 else "partial_error",
        "inputs": {
            "season": season,
            "leagues": list(leagues),
            "import_dir": str(import_root.resolve()),
            "python_executable": python_executable or sys.executable,
        },
        "flags": {
            "force": bool(force),
            "skip_transfermarkt": bool(skip_transfermarkt),
            "skip_sofascore": bool(skip_sofascore),
            "skip_merge": bool(skip_merge),
        },
        "artifacts": {
            "import_dir": _require_artifact(import_root, "import dir"),
            "summary_json": _artifact_meta(summary_json),
        },
        "snapshots": {
            "results": rows,
        },
        "counts": {
            "total": len(rows),
            "ok": success_count,
            "error": error_count,
        },
        "results": rows,
    }

    if summary_json:
        out_path = Path(summary_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary["artifacts"]["summary_json"] = _require_artifact(out_path, "summary json")
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[future-backfill] wrote summary -> {out_path}")

    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill real future-season *_with_sofa.csv league files by running the existing "
            "Transfermarkt scrape, Sofascore league pull, and TM+Sofa merge pipeline for one or more leagues."
        )
    )
    parser.add_argument("--season", default="2025/26")
    parser.add_argument(
        "--leagues",
        default=",".join(DEFAULT_BACKFILL_LEAGUES),
        help="Comma-separated league slugs from scouting_ml.league_registry.",
    )
    parser.add_argument("--import-dir", default="data/incoming_future_2025_26")
    parser.add_argument("--python-executable", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-transfermarkt", action="store_true")
    parser.add_argument("--skip-sofascore", action="store_true")
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument(
        "--summary-json",
        default="data/model/reports/future_scored/future_backfill_summary.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    run_future_league_backfill(
        leagues=_parse_csv_tokens(args.leagues),
        season=args.season,
        import_dir=args.import_dir,
        python_executable=args.python_executable or None,
        force=args.force,
        skip_transfermarkt=args.skip_transfermarkt,
        skip_sofascore=args.skip_sofascore,
        skip_merge=args.skip_merge,
        summary_json=args.summary_json or None,
    )


if __name__ == "__main__":
    main()
