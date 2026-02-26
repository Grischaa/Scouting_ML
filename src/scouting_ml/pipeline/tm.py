from __future__ import annotations

import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from scouting_ml.paths import PROCESSED_DIR
from scouting_ml.league_registry import LeagueConfig, season_slug_label


@dataclass
class TransfermarktResult:
    season: str
    combined_path: Path
    clean_path: Path


def run_transfermarkt(
    config: LeagueConfig,
    season: str,
    *,
    force: bool = False,
    python_executable: str | None = None,
) -> TransfermarktResult:
    """
    Orchestrate the Transfermarkt side of the pipeline for a league season.

    1. Run the club-by-club scrape via scouting_ml.tm.process_league.
    2. Combine club CSVs for the league/season into a single players file.
    3. Run the cleaning step to produce a tidy dataset.
    """

    python_cmd = python_executable or sys.executable
    season_slug = season_slug_label(season)

    combined_path = PROCESSED_DIR / f"{config.slug}_{season_slug}_players.csv"
    clean_path = PROCESSED_DIR / f"{config.slug}_{season_slug}_clean.csv"

    if clean_path.exists() and not force:
        print(f"[tm] Using cached clean dataset {clean_path}")
        return TransfermarktResult(season=season, combined_path=combined_path, clean_path=clean_path)

    season_id = config.tm_season_ids[season]
    league_url = _season_specific_league_url(config.tm_league_url, season_id)

    cmd = [
        python_cmd,
        "-m",
        "scouting_ml.tm.process_league",
        "--league-url",
        league_url,
        "--league-name",
        config.name,
        "--season",
        season,
        "--season-id",
        str(season_id),
        "--sleep",
        "3",
    ]
    _run_command(cmd, "[tm] process_league")

    print(f"[tm] Combining club CSVs for {config.name} {season}")
    combined_rows = _collect_league_rows(config, season)
    if not combined_rows.rows:
        raise RuntimeError(
            f"No team-level rows found for {config.name} {season}. "
            "Ensure scouting_ml.tm.process_league produced *_team_players.csv files."
        )
    _write_combined_csv(combined_path, combined_rows)

    cmd = [
        python_cmd,
        "-m",
        "scouting_ml.tm.clean_tm",
        "--infile",
        str(combined_path),
        "--outfile",
        str(clean_path),
        "--default-league",
        config.name,
        "--default-season",
        season,
    ]
    _run_command(cmd, "[tm] clean_tm")

    return TransfermarktResult(season=season, combined_path=combined_path, clean_path=clean_path)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

@dataclass
class _CollectedRows:
    header: Tuple[str, ...]
    rows: Dict[Tuple[str | None, str | None], Dict[str, str | None]]


def _collect_league_rows(config: LeagueConfig, season: str) -> _CollectedRows:
    header: Tuple[str, ...] | None = None
    rows: Dict[Tuple[str | None, str | None], Dict[str, str | None]] = {}

    for path in PROCESSED_DIR.rglob("*_team_players.csv"):
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                continue
            if header is None:
                header = tuple(reader.fieldnames)
            for row in reader:
                if row.get("league") != config.name:
                    continue
                if row.get("season") != season:
                    continue
                player_key = (row.get("player_id") or row.get("name"), row.get("season"))
                rows[player_key] = row

    if header is None:
        header = ()
    return _CollectedRows(header=header, rows=rows)


def _write_combined_csv(dest: Path, collected: _CollectedRows) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8") as fh:
        if not collected.header:
            raise RuntimeError("No header detected while combining league CSVs.")
        writer = csv.DictWriter(fh, fieldnames=list(collected.header))
        writer.writeheader()
        for row in collected.rows.values():
            writer.writerow(row)
    print(f"[tm] Wrote {len(collected.rows)} player rows -> {dest}")


def _run_command(cmd: list[str], label: str) -> None:
    print(f"{label} :: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")


def _season_specific_league_url(base_url: str, season_id: int) -> str:
    if "{season_id}" in base_url:
        return base_url.format(season_id=season_id)

    pattern = re.compile(r"saison_id/\d+")
    if "saison_id" in base_url:
        return pattern.sub(f"saison_id/{season_id}", base_url)

    if "startseite" in base_url and "?" not in base_url:
        base = base_url.rstrip("/")
        if not base.endswith("plus"):
            base = base + "/plus"
        return f"{base}/?saison_id={season_id}"

    if base_url.rstrip("/").endswith("?"):
        return f"{base_url}season_id={season_id}"

    if "?" in base_url:
        sep = "&" if not base_url.endswith("&") else ""
        return f"{base_url}{sep}saison_id={season_id}"

    return base_url.rstrip("/") + f"/saison_id/{season_id}"
