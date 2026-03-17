from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ArchiveCandidate:
    path: Path
    reason: str


def _repo_root() -> Path:
    return Path.cwd()


def _load_manifest_index(manifest_path: Path) -> dict[str, dict[str, str]]:
    if not manifest_path.exists():
        return {}
    rows: dict[str, dict[str, str]] = {}
    with manifest_path.open(encoding="utf-8") as handle:
        header = None
        for raw in handle:
            line = raw.rstrip("\n")
            if not line:
                continue
            parts = line.split(",")
            if header is None:
                header = parts
                continue
            row = {header[idx]: parts[idx] if idx < len(parts) else "" for idx in range(len(header))}
            basename = row.get("basename")
            if basename:
                rows[basename] = row
    return rows


def _norm_path_str(value: str | None) -> str:
    return str(value or "").replace("\\", "/").strip()


def _iter_incoming_future_candidates(base_dir: Path, manifest_index: dict[str, dict[str, str]]) -> Iterable[ArchiveCandidate]:
    incoming_dir = base_dir / "processed" / "_incoming_future"
    if not incoming_dir.exists():
        return
    for path in sorted(incoming_dir.glob("*_with_sofa.csv")):
        row = manifest_index.get(path.name)
        if not row:
            continue
        source_path = _norm_path_str(row.get("source_path"))
        if "/_incoming_future/" in f"/{source_path}/":
            continue
        yield ArchiveCandidate(path=path, reason="staging_import_superseded")


def _iter_with_sofa_candidates(base_dir: Path, manifest_index: dict[str, dict[str, str]]) -> Iterable[ArchiveCandidate]:
    legacy_dir = base_dir / "processed" / "_with_sofa"
    if not legacy_dir.exists():
        return
    for path in sorted(legacy_dir.glob("*_with_sofa.csv")):
        row = manifest_index.get(path.name)
        if not row:
            continue
        source_path = _norm_path_str(row.get("source_path"))
        if not source_path:
            continue
        if "/_with_sofa/" in f"/{source_path}/":
            continue
        yield ArchiveCandidate(path=path, reason="legacy_with_sofa_superseded")


def _iter_tmp_model_candidates(base_dir: Path) -> Iterable[ArchiveCandidate]:
    model_dir = base_dir / "model"
    if not model_dir.exists():
        return
    for path in sorted(model_dir.glob("_tmp_*")):
        if path.is_file():
            yield ArchiveCandidate(path=path, reason="temporary_model_artifact")


def _iter_old_scout_workflow_candidates(base_dir: Path) -> Iterable[ArchiveCandidate]:
    workflow_dir = base_dir / "model" / "scout_workflow"
    if not workflow_dir.exists():
        return
    keep_names = {
        "future_scored_review",
        "future_scored_review_no_estonia",
    }
    for path in sorted(workflow_dir.iterdir()):
        if path.name in keep_names:
            continue
        if path.name.startswith("memos_") or path.name.startswith("scout_workflow_") or path.name.startswith("weekly_ops_summary_"):
            yield ArchiveCandidate(path=path, reason="stale_scout_workflow_output")


def _iter_stale_candidate_candidates(base_dir: Path) -> Iterable[ArchiveCandidate]:
    candidates_dir = base_dir / "model" / "candidates"
    if not candidates_dir.exists():
        return
    keep_names = {
        "cheap_aggressive_prod60.csv",
        "cheap_aggressive_prod60.error_priors.csv",
        "cheap_aggressive_prod60.metrics.json",
        "cheap_aggressive_prod60.quality.json",
        "cheap_aggressive_prod60_future_benchmark_report.json",
        "cheap_aggressive_prod60_future_benchmark_report.md",
        "cheap_aggressive_prod60_future_diagnostics.json",
        "cheap_aggressive_prod60_future_diagnostics.md",
        "cheap_aggressive_prod60_future_score_diagnostics.json",
        "cheap_aggressive_prod60_future_scored.csv",
        "cheap_aggressive_prod60_val.csv",
        "cheap_aggressive_prod60_val_future_scored.csv",
    }
    for path in sorted(candidates_dir.iterdir()):
        if path.name in keep_names:
            continue
        if path.name.startswith("cheap_aggressive_2024_25_"):
            yield ArchiveCandidate(path=path, reason="stale_candidate_experiment")


def _iter_stale_report_candidates(base_dir: Path) -> Iterable[ArchiveCandidate]:
    reports_dir = base_dir / "model" / "reports"
    if not reports_dir.exists():
        return

    benchmark_holdouts = reports_dir / "benchmark_holdouts"
    if benchmark_holdouts.exists():
        yield ArchiveCandidate(path=benchmark_holdouts, reason="stale_benchmark_holdouts")

    low_value_root = reports_dir / "low_value_contract_holdout"
    drop_dir = low_value_root / "cheap_aggressive_drop_contract_security"
    if drop_dir.exists():
        yield ArchiveCandidate(path=drop_dir, reason="stale_contract_variant_holdout")
    for name in (
        "cheap_aggressive_vs_drop_contract_security.csv",
        "cheap_aggressive_vs_drop_contract_security.md",
    ):
        path = low_value_root / name
        if path.exists():
            yield ArchiveCandidate(path=path, reason="stale_contract_variant_report")

    holdout_compare = reports_dir / "holdout_compare"
    for name in (
        "no_contract",
        "no_contract_fast",
        "no_contract_eb_fast",
        "compare_full_vs_no_contract_tp.csv",
        "compare_full_vs_no_contract_eb.csv",
    ):
        path = holdout_compare / name
        if path.exists():
            yield ArchiveCandidate(path=path, reason="stale_holdout_compare_output")

    candidate_promotion = reports_dir / "candidate_promotion"
    for name in (
        "candidate_vs_champion.json",
        "candidate_vs_champion.md",
        "cheap_aggressive_vs_champion.json",
        "cheap_aggressive_vs_champion.md",
    ):
        path = candidate_promotion / name
        if path.exists():
            yield ArchiveCandidate(path=path, reason="stale_candidate_promotion_report")


def _unique_candidates(candidates: Iterable[ArchiveCandidate]) -> list[ArchiveCandidate]:
    seen: set[Path] = set()
    out: list[ArchiveCandidate] = []
    for item in candidates:
        resolved = item.path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(item)
    return out


def archive_stale_workspace_artifacts(
    *,
    data_root: str = "data",
    manifest_path: str = "data/processed/organization_manifest.csv",
    archive_root: str | None = None,
    execute: bool = False,
    summary_json: str | None = None,
) -> dict[str, object]:
    repo_root = _repo_root()
    base_dir = repo_root / data_root
    manifest = repo_root / manifest_path
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = repo_root / (archive_root or f"data/archive/stale_cleanup_{stamp}")
    manifest_index = _load_manifest_index(manifest)

    candidates = _unique_candidates(
        [
            *list(_iter_incoming_future_candidates(base_dir, manifest_index)),
            *list(_iter_with_sofa_candidates(base_dir, manifest_index)),
            *list(_iter_tmp_model_candidates(base_dir)),
            *list(_iter_old_scout_workflow_candidates(base_dir)),
            *list(_iter_stale_candidate_candidates(base_dir)),
            *list(_iter_stale_report_candidates(base_dir)),
        ]
    )

    moved: list[dict[str, object]] = []
    for item in candidates:
        rel = item.path.resolve().relative_to(repo_root.resolve())
        dest = archive_dir / rel
        row = {
            "source_path": str(item.path.resolve()),
            "archive_path": str(dest.resolve()),
            "reason": item.reason,
            "size_bytes": int(item.path.stat().st_size) if item.path.exists() and item.path.is_file() else None,
            "path_type": "dir" if item.path.exists() and item.path.is_dir() else "file",
        }
        if execute:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(item.path), str(dest))
        moved.append(row)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(base_dir.resolve()),
        "manifest_path": str(manifest.resolve()) if manifest.exists() else None,
        "archive_root": str(archive_dir.resolve()),
        "execute": bool(execute),
        "candidate_count": len(candidates),
        "moved_count": len(moved) if execute else 0,
        "items": moved,
    }

    if execute:
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_manifest = archive_dir / "archive_manifest.json"
        archive_manifest.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary["archive_manifest"] = str(archive_manifest.resolve())

    if summary_json:
        out = repo_root / summary_json
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Archive stale workspace artifacts conservatively: superseded future-import staging files, "
            "superseded legacy _with_sofa files, and obvious temporary model artifacts."
        )
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--manifest-path", default="data/processed/organization_manifest.csv")
    parser.add_argument("--archive-root", default=None)
    parser.add_argument("--execute", action="store_true", help="Move files instead of reporting only.")
    parser.add_argument(
        "--summary-json",
        default="data/model/reports/archive_stale_workspace_summary.json",
        help="Optional JSON summary output.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = archive_stale_workspace_artifacts(
        data_root=args.data_root,
        manifest_path=args.manifest_path,
        archive_root=args.archive_root,
        execute=bool(args.execute),
        summary_json=args.summary_json,
    )
    action = "archived" if args.execute else "identified"
    print(f"[archive] {action} {summary['candidate_count']} files")
    print(f"[archive] archive root -> {summary['archive_root']}")
    if summary.get("archive_manifest"):
        print(f"[archive] manifest -> {summary['archive_manifest']}")
    if args.summary_json:
        print(f"[archive] summary -> {Path(args.summary_json)}")


if __name__ == "__main__":
    main()
