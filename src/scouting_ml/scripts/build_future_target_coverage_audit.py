from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from scouting_ml.scripts.build_future_value_targets import build_future_value_targets_frame


def _find_files(pattern: str) -> list[Path]:
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        matches = glob.glob(str(Path.cwd() / pattern), recursive=True)
    return sorted({Path(match).resolve() for match in matches if Path(match).is_file()})


def _season_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if "season" not in frame.columns:
        return out
    work = frame.copy()
    work["season"] = work["season"].astype(str).str.strip()
    work["_has_target"] = pd.to_numeric(work.get("has_next_season_target"), errors="coerce") == 1
    for season, group in work.groupby("season", dropna=False):
        out.append(
            {
                "slice_type": "season",
                "season": str(season),
                "league": "",
                "rows": int(len(group)),
                "labeled_rows": int(group["_has_target"].sum()),
                "labeled_share": float(group["_has_target"].mean()),
            }
        )
    out.sort(key=lambda row: row["season"])
    return out


def _league_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if "season" not in frame.columns or "league" not in frame.columns:
        return out
    work = frame.copy()
    work["season"] = work["season"].astype(str).str.strip()
    work["league"] = work["league"].astype(str).str.strip()
    work["_has_target"] = pd.to_numeric(work.get("has_next_season_target"), errors="coerce") == 1
    for (season, league), group in work.groupby(["season", "league"], dropna=False):
        out.append(
            {
                "slice_type": "season_league",
                "season": str(season),
                "league": str(league),
                "rows": int(len(group)),
                "labeled_rows": int(group["_has_target"].sum()),
                "labeled_share": float(group["_has_target"].mean()),
            }
        )
    out.sort(key=lambda row: (row["season"], -row["labeled_rows"], row["league"]))
    return out


def _source_inventory(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.append(
            {
                "path": str(path),
                "name": path.name,
                "stem": path.stem,
                "parent": str(path.parent),
            }
        )
    return rows


def build_future_target_coverage_audit(
    *,
    dataset_path: str,
    future_targets_output: str,
    out_json: str,
    out_csv: str,
    min_next_minutes: float = 450.0,
    future_source_glob: str = "data/processed/**/*2025-26*_with_sofa.csv",
) -> dict[str, Any]:
    dataset = Path(dataset_path)
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")

    frame = pd.read_parquet(dataset)
    targets = build_future_value_targets_frame(
        frame,
        min_next_minutes=min_next_minutes,
        drop_na_target=False,
    )

    future_targets_path = Path(future_targets_output)
    future_targets_path.parent.mkdir(parents=True, exist_ok=True)
    targets.to_parquet(future_targets_path, index=False)

    season_rows = _season_rows(targets)
    league_rows = _league_rows(targets)
    source_paths = _find_files(future_source_glob)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset.resolve()),
        "future_targets_output": str(future_targets_path.resolve()),
        "min_next_minutes": float(min_next_minutes),
        "total_rows": int(len(targets)),
        "labeled_rows": int((pd.to_numeric(targets.get("has_next_season_target"), errors="coerce") == 1).sum()),
        "future_source_glob": future_source_glob,
        "future_source_files": _source_inventory(source_paths),
        "season_rows": season_rows,
        "season_league_rows": league_rows,
    }

    out_json_path = Path(out_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    out_csv_path = Path(out_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([*season_rows, *league_rows]).to_csv(out_csv_path, index=False)

    print(f"[future-audit] wrote future targets -> {future_targets_path}")
    print(f"[future-audit] wrote json -> {out_json_path}")
    print(f"[future-audit] wrote csv -> {out_csv_path}")
    print(f"[future-audit] discovered future-season source files -> {len(source_paths)}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a future-target parquet from the current clean dataset and audit how much next-season coverage "
            "is actually available by season and league."
        )
    )
    parser.add_argument("--dataset", default="data/model/tm_context_candidate_clean.parquet")
    parser.add_argument("--future-targets-output", default="data/model/big5_players_future_targets.parquet")
    parser.add_argument("--out-json", default="data/model/reports/future_target_coverage_audit.json")
    parser.add_argument("--out-csv", default="data/model/reports/future_target_coverage_audit.csv")
    parser.add_argument("--min-next-minutes", type=float, default=450.0)
    parser.add_argument(
        "--future-source-glob",
        default="data/processed/**/*2025-26*_with_sofa.csv",
        help="Glob used to inventory locally available future-season raw sources.",
    )
    args = parser.parse_args()

    build_future_target_coverage_audit(
        dataset_path=args.dataset,
        future_targets_output=args.future_targets_output,
        out_json=args.out_json,
        out_csv=args.out_csv,
        min_next_minutes=args.min_next_minutes,
        future_source_glob=args.future_source_glob,
    )


if __name__ == "__main__":
    main()
