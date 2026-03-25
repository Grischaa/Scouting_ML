from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from scouting_ml.models.build_dataset import main as build_dataset_main
from scouting_ml.models.clean_dataset import clean_dataset
from scouting_ml.reporting.future_value_benchmarks import (
    build_future_value_benchmark_payload,
    write_future_value_benchmark_report,
)
from scouting_ml.reporting.future_value_diagnostics import (
    build_future_value_diagnostics_payload,
    write_future_value_diagnostics_report,
)
from scouting_ml.scripts.build_future_scout_score import build_future_scout_score
from scouting_ml.scripts.build_future_target_coverage_audit import build_future_target_coverage_audit
from scouting_ml.scripts.organize_processed_csvs import organize_processed_files
from scouting_ml.scripts.run_market_value_candidate_promotion import main as promotion_main


def _parse_csv_tokens(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    values = {token.strip().casefold() for token in str(raw).split(",") if token.strip()}
    return values or None


def _find_import_candidates(import_dir: Path, season_filter: str) -> list[Path]:
    pattern = f"*{season_filter}*_with_sofa.csv"
    return sorted(path for path in import_dir.rglob(pattern) if path.is_file())


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


def _load_json_snapshot(path: str | Path | None) -> dict | list | None:
    if path in (None, ""):
        return None
    target = Path(path)
    if not target.exists():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return None


def _append_future_promotion_args(
    promotion_args: list[str],
    *,
    candidate_future_benchmark_json: str,
    champion_future_benchmark_json: str | None,
) -> None:
    promotion_args.extend(
        [
            "--candidate-future-benchmark-json",
            candidate_future_benchmark_json,
            "--require-future-benchmark",
        ]
    )
    if champion_future_benchmark_json and Path(champion_future_benchmark_json).exists():
        promotion_args.extend(
            [
                "--champion-future-benchmark-json",
                champion_future_benchmark_json,
                "--require-future-precision-vs-champion",
            ]
        )


def _manifest_lane_summary(manifest_payload: dict | None) -> dict[str, Any]:
    if not isinstance(manifest_payload, dict):
        return {}
    lanes: dict[str, Any] = {}
    for key in ("valuation_champion", "future_shortlist_champion"):
        section = manifest_payload.get(key)
        if not isinstance(section, dict):
            continue
        artifacts = section.get("artifacts") if isinstance(section.get("artifacts"), dict) else {}
        config = section.get("config") if isinstance(section.get("config"), dict) else {}
        role = str(section.get("role") or key.replace("_champion", ""))
        lanes[role] = {
            "role": role,
            "label": section.get("label"),
            "lane_state": section.get("lane_state"),
            "promotion_state": section.get("promotion_state"),
            "promotion_reasons": section.get("promotion_reasons"),
            "generated_at_utc": section.get("generated_at_utc"),
            "test_season": config.get("test_season"),
            "artifact_paths": {
                artifact_key: artifact_meta.get("path")
                for artifact_key, artifact_meta in artifacts.items()
                if isinstance(artifact_meta, dict)
            },
        }
    return lanes


def _confidence_distribution(series: pd.Series) -> dict[str, int]:
    numeric = pd.to_numeric(series, errors="coerce")
    return {
        "high": int((numeric >= 70.0).sum()),
        "medium": int(((numeric >= 45.0) & (numeric < 70.0)).sum()),
        "low": int((numeric < 45.0).sum()),
    }


def _summarize_talent_outputs(path: str | Path | None) -> dict[str, Any]:
    if path in (None, ""):
        return {}
    target = Path(path)
    if not target.exists():
        return {}
    frame = pd.read_csv(target, low_memory=False)
    summary: dict[str, Any] = {
        "rows": int(len(frame)),
        "position_family_counts": {},
        "future_label_coverage": {
            "labeled_rows": 0,
            "total_rows": int(len(frame)),
        },
        "confidence_distribution": {"high": 0, "medium": 0, "low": 0},
    }
    if "talent_position_family" in frame.columns:
        summary["position_family_counts"] = {
            key: int(value)
            for key, value in frame["talent_position_family"].astype(str).value_counts().sort_index().items()
        }
    if "has_next_season_target" in frame.columns:
        labeled = pd.to_numeric(frame["has_next_season_target"], errors="coerce").fillna(0.0)
        summary["future_label_coverage"] = {
            "labeled_rows": int(labeled.sum()),
            "total_rows": int(len(frame)),
        }
    if "future_potential_confidence" in frame.columns:
        summary["confidence_distribution"] = _confidence_distribution(frame["future_potential_confidence"])
    return summary


def _copy_import_sources(
    *,
    import_dir: str | None,
    staging_dir: str,
    season_filter: str,
) -> dict[str, Any]:
    if not import_dir:
        return {
            "import_dir": None,
            "staging_dir": str(Path(staging_dir).resolve()),
            "season_filter": season_filter,
            "copied_files": [],
            "copied_count": 0,
            "skipped": True,
            "skip_reason": "no_import_dir_provided",
        }

    source_root = Path(import_dir)
    if not source_root.exists():
        print(f"[future-refresh] import dir missing, skipping import -> {source_root}")
        return {
            "import_dir": str(source_root.resolve()),
            "staging_dir": str(Path(staging_dir).resolve()),
            "season_filter": season_filter,
            "copied_files": [],
            "copied_count": 0,
            "skipped": True,
            "skip_reason": "import_dir_missing",
        }
    candidates = _find_import_candidates(source_root, season_filter=season_filter)
    if not candidates:
        print(f"[future-refresh] no matching import files found, skipping import -> {source_root}")
        return {
            "import_dir": str(source_root.resolve()),
            "staging_dir": str(Path(staging_dir).resolve()),
            "season_filter": season_filter,
            "copied_files": [],
            "copied_count": 0,
            "skipped": True,
            "skip_reason": "no_matching_import_files",
        }

    by_name: dict[str, list[Path]] = {}
    for path in candidates:
        by_name.setdefault(path.name, []).append(path)
    duplicates = {name: paths for name, paths in by_name.items() if len(paths) > 1}
    if duplicates:
        details = "; ".join(f"{name}: {len(paths)} files" for name, paths in sorted(duplicates.items()))
        raise ValueError(
            "Import directory must contain at most one file per league-season basename. "
            f"Duplicate basenames found: {details}"
        )

    dest_root = Path(staging_dir)
    dest_root.mkdir(parents=True, exist_ok=True)

    copied_files: list[dict[str, str]] = []
    for src in candidates:
        dst = dest_root / src.name
        shutil.copy2(src, dst)
        copied_files.append(
            {
                "source": str(src.resolve()),
                "destination": str(dst.resolve()),
                "basename": src.name,
            }
        )
    return {
        "import_dir": str(source_root.resolve()),
        "staging_dir": str(dest_root.resolve()),
        "season_filter": season_filter,
        "copied_files": copied_files,
        "copied_count": len(copied_files),
        "skipped": False,
        "skip_reason": None,
    }


def run_future_data_refresh(
    *,
    import_dir: str | None,
    staging_dir: str,
    season_filter: str,
    source_dir: str,
    combined_dir: str,
    country_root: str,
    season_root: str,
    organization_manifest: str,
    clean_targets: bool,
    dataset_output: str,
    clean_output: str,
    external_dir: str,
    clean_min_minutes: float,
    future_targets_output: str,
    future_audit_json: str,
    future_audit_csv: str,
    future_source_glob: str,
    min_next_minutes: float,
    base_val_predictions: str,
    base_test_predictions: str | None,
    scored_val_output: str,
    scored_test_output: str | None,
    diagnostics_output: str,
    label_mode: str,
    k_eval: int,
    scoring_min_minutes: float,
    scoring_max_age: float | None,
    scoring_positions: set[str] | None,
    scoring_include_leagues: set[str] | None,
    scoring_exclude_leagues: set[str] | None,
    future_benchmark_json: str,
    future_benchmark_md: str,
    future_diagnostics_json: str,
    future_diagnostics_md: str,
    future_u23_nonbig5_json: str | None,
    future_u23_nonbig5_md: str | None,
    future_score_col: str,
    future_cohort_min_labeled: int,
    future_k_values: Sequence[int],
    promotion_enabled: bool,
    promotion_args: Sequence[str],
    summary_json: str,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    import_summary = _copy_import_sources(
        import_dir=import_dir,
        staging_dir=staging_dir,
        season_filter=season_filter,
    )

    organize_processed_files(
        source_dir=source_dir,
        combined_dir=combined_dir,
        country_root=country_root,
        season_root=season_root,
        manifest_path=organization_manifest,
        clean_targets=clean_targets,
    )

    build_dataset_main(
        data_dir=combined_dir,
        output=dataset_output,
        external_dir=external_dir,
    )
    clean_dataset(
        input_path=dataset_output,
        output_path=clean_output,
        min_minutes=clean_min_minutes,
    )

    audit_payload = build_future_target_coverage_audit(
        dataset_path=clean_output,
        future_targets_output=future_targets_output,
        out_json=future_audit_json,
        out_csv=future_audit_csv,
        min_next_minutes=min_next_minutes,
        future_source_glob=future_source_glob,
    )

    diagnostics = build_future_scout_score(
        val_predictions_path=base_val_predictions,
        test_predictions_path=base_test_predictions,
        out_val_path=scored_val_output,
        out_test_path=scored_test_output,
        diagnostics_out=diagnostics_output,
        future_targets_path=future_targets_output,
        dataset_path=clean_output,
        min_next_minutes=min_next_minutes,
        min_minutes=scoring_min_minutes,
        max_age=scoring_max_age,
        positions=scoring_positions,
        include_leagues=scoring_include_leagues,
        exclude_leagues=scoring_exclude_leagues,
        label_mode=label_mode,
        k_eval=k_eval,
    )

    future_payload = build_future_value_benchmark_payload(
        test_predictions_path=scored_test_output,
        val_predictions_path=scored_val_output,
        future_targets_path=future_targets_output,
        dataset_path=clean_output,
        score_col=future_score_col,
        k_values=future_k_values,
        cohort_min_labeled=future_cohort_min_labeled,
        min_next_minutes=min_next_minutes,
        min_minutes=scoring_min_minutes,
        max_age=scoring_max_age,
        non_big5_only=False,
        top_realized_limit=max(future_k_values),
    )
    future_report_paths = write_future_value_benchmark_report(
        future_payload,
        out_json=future_benchmark_json,
        out_md=future_benchmark_md,
    )
    future_diagnostics_payload = build_future_value_diagnostics_payload(
        future_payload,
        source_benchmark_json=future_report_paths["json"],
        k=max(future_k_values),
        top_n=5,
    )
    future_diagnostics_paths = write_future_value_diagnostics_report(
        future_diagnostics_payload,
        out_json=future_diagnostics_json,
        out_md=future_diagnostics_md,
    )

    u23_report_paths: dict[str, str] | None = None
    if future_u23_nonbig5_json and future_u23_nonbig5_md:
        u23_payload = build_future_value_benchmark_payload(
            test_predictions_path=scored_test_output,
            val_predictions_path=scored_val_output,
            future_targets_path=future_targets_output,
            dataset_path=clean_output,
            score_col=future_score_col,
            k_values=future_k_values,
            cohort_min_labeled=future_cohort_min_labeled,
            min_next_minutes=min_next_minutes,
            min_minutes=scoring_min_minutes,
            max_age=23.0,
            non_big5_only=True,
            top_realized_limit=max(future_k_values),
        )
        u23_report_paths = write_future_value_benchmark_report(
            u23_payload,
            out_json=future_u23_nonbig5_json,
            out_md=future_u23_nonbig5_md,
        )

    promotion_rc: int | None = None
    if promotion_enabled:
        promotion_rc = promotion_main(list(promotion_args))

    artifacts = {
        "organization_manifest": _require_artifact(organization_manifest, "organization manifest"),
        "dataset_output": _require_artifact(dataset_output, "dataset"),
        "clean_output": _require_artifact(clean_output, "clean dataset"),
        "future_targets_output": _require_artifact(future_targets_output, "future targets"),
        "future_audit_json": _require_artifact(future_audit_json, "future audit json"),
        "future_audit_csv": _require_artifact(future_audit_csv, "future audit csv"),
        "scored_val_output": _require_artifact(scored_val_output, "future scored val predictions"),
        "scored_test_output": _require_artifact(scored_test_output, "future scored test predictions")
        if scored_test_output
        else _artifact_meta(scored_test_output),
        "diagnostics_output": _require_artifact(diagnostics_output, "future scoring diagnostics"),
        "future_benchmark_json": _require_artifact(future_report_paths["json"], "future benchmark json"),
        "future_benchmark_md": _require_artifact(future_report_paths["markdown"], "future benchmark markdown"),
        "future_diagnostics_json": _require_artifact(
            future_diagnostics_paths["json"], "future diagnostics json"
        ),
        "future_diagnostics_md": _require_artifact(
            future_diagnostics_paths["markdown"], "future diagnostics markdown"
        ),
        "future_u23_nonbig5_json": _require_artifact(u23_report_paths["json"], "future U23 non-Big5 benchmark json")
        if u23_report_paths
        else _artifact_meta(None),
        "future_u23_nonbig5_md": _require_artifact(u23_report_paths["markdown"], "future U23 non-Big5 benchmark markdown")
        if u23_report_paths
        else _artifact_meta(None),
    }

    finished_at = datetime.now(timezone.utc)
    summary = {
        "generated_at_utc": finished_at.isoformat(),
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "status": "ok",
        "inputs": {
            "import_dir": None if not import_dir else str(Path(import_dir).resolve()),
            "staging_dir": str(Path(staging_dir).resolve()),
            "season_filter": season_filter,
            "source_dir": str(Path(source_dir).resolve()),
            "combined_dir": str(Path(combined_dir).resolve()),
            "external_dir": str(Path(external_dir).resolve()),
            "base_val_predictions": str(Path(base_val_predictions).resolve()),
            "base_test_predictions": None if not base_test_predictions else str(Path(base_test_predictions).resolve()),
        },
        "flags": {
            "clean_targets": bool(clean_targets),
            "promotion_enabled": bool(promotion_enabled),
            "u23_nonbig5_report_enabled": bool(future_u23_nonbig5_json and future_u23_nonbig5_md),
        },
        "import": import_summary,
        "artifacts": artifacts,
        "snapshots": {
            "future_audit": _load_json_snapshot(future_audit_json),
            "future_score_diagnostics": _load_json_snapshot(diagnostics_output),
            "future_benchmark": _load_json_snapshot(future_report_paths["json"]),
            "future_benchmark_diagnostics": _load_json_snapshot(future_diagnostics_paths["json"]),
            "future_u23_nonbig5_benchmark": _load_json_snapshot(u23_report_paths["json"]) if u23_report_paths else None,
            "lock_manifest": _load_json_snapshot(next((promotion_args[idx + 1] for idx, value in enumerate(promotion_args) if value == "--manifest-out"), None))
            if promotion_enabled
            else None,
        },
        "future_audit": {
            "total_rows": audit_payload.get("total_rows"),
            "labeled_rows": audit_payload.get("labeled_rows"),
            "future_source_file_count": len(audit_payload.get("future_source_files") or []),
            "season_rows": audit_payload.get("season_rows") or [],
        },
        "future_score": {
            "training_rows": diagnostics.get("training_rows"),
            "training_positive_rate": diagnostics.get("training_positive_rate"),
            "val_metrics": diagnostics.get("val_metrics") or {},
        },
        "promotion": {
            "enabled": bool(promotion_enabled),
            "rc": promotion_rc,
        },
    }
    if promotion_enabled:
        summary["artifact_lanes"] = _manifest_lane_summary(summary["snapshots"].get("lock_manifest"))
    summary["talent_summary"] = {
        "val": _summarize_talent_outputs(scored_val_output),
        "test": _summarize_talent_outputs(scored_test_output),
        "lane_posture": {
            "valuation": (summary.get("artifact_lanes") or {}).get("valuation")
            if isinstance(summary.get("artifact_lanes"), dict)
            else None,
            "future_shortlist": (summary.get("artifact_lanes") or {}).get("future_shortlist")
            if isinstance(summary.get("artifact_lanes"), dict)
            else {
                "role": "future_shortlist",
                "lane_state": "live",
                "promotion_state": "advisory_only",
            },
        },
    }

    out_path = Path(summary_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[future-refresh] wrote summary -> {out_path}")
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import new future-season *_with_sofa.csv files, canonicalize data/processed, rebuild the clean dataset, "
            "refresh future targets and future-scored predictions, and optionally rerun future-gated promotion."
        )
    )
    parser.add_argument("--import-dir", default=None, help="Folder containing new 2025-26 *_with_sofa.csv files.")
    parser.add_argument("--staging-dir", default="data/processed/_incoming_future")
    parser.add_argument("--season-filter", default="2025-26")
    parser.add_argument("--source-dir", default="data/processed")
    parser.add_argument("--combined-dir", default="data/processed/Clubs combined")
    parser.add_argument("--country-root", default="data/processed/by_country")
    parser.add_argument("--season-root", default="data/processed/by_season")
    parser.add_argument("--organization-manifest", default="data/processed/organization_manifest.csv")
    parser.add_argument("--no-clean-targets", action="store_true")
    parser.add_argument("--dataset-output", default="data/model/tm_context_candidate.parquet")
    parser.add_argument("--clean-output", default="data/model/tm_context_candidate_clean.parquet")
    parser.add_argument("--external-dir", default="data/external")
    parser.add_argument("--clean-min-minutes", type=float, default=450.0)
    parser.add_argument("--future-targets-output", default="data/model/big5_players_future_targets.parquet")
    parser.add_argument("--future-audit-json", default="data/model/reports/future_target_coverage_audit.json")
    parser.add_argument("--future-audit-csv", default="data/model/reports/future_target_coverage_audit.csv")
    parser.add_argument(
        "--future-source-glob",
        default="data/processed/**/*2025-26*_with_sofa.csv",
        help="Glob used in the future coverage audit to inventory available future-season sources.",
    )
    parser.add_argument("--min-next-minutes", type=float, default=450.0)
    parser.add_argument(
        "--base-val-predictions",
        default="data/model/champion_predictions_2024-25_val.csv",
    )
    parser.add_argument(
        "--base-test-predictions",
        default="data/model/champion_predictions_2024-25.csv",
    )
    parser.add_argument(
        "--scored-val-output",
        default="data/model/reports/future_scored/cheap_aggressive_2024-25_val_future_scored.csv",
    )
    parser.add_argument(
        "--scored-test-output",
        default="data/model/reports/future_scored/cheap_aggressive_2024-25_future_scored.csv",
    )
    parser.add_argument(
        "--diagnostics-output",
        default="data/model/reports/future_scored/cheap_aggressive_future_score_diagnostics.json",
    )
    parser.add_argument("--label-mode", default="positive_growth", choices=["positive_growth", "growth_gt25pct"])
    parser.add_argument("--k-eval", type=int, default=25)
    parser.add_argument("--scoring-min-minutes", type=float, default=900.0)
    parser.add_argument("--scoring-max-age", type=float, default=-1.0, help="Set negative to disable.")
    parser.add_argument("--scoring-positions", default="")
    parser.add_argument("--scoring-include-leagues", default="")
    parser.add_argument("--scoring-exclude-leagues", default="")
    parser.add_argument(
        "--future-benchmark-json",
        default="data/model/reports/future_scored/cheap_aggressive_future_benchmark_report.json",
    )
    parser.add_argument(
        "--future-benchmark-md",
        default="data/model/reports/future_scored/cheap_aggressive_future_benchmark_report.md",
    )
    parser.add_argument(
        "--future-diagnostics-json",
        default="data/model/reports/future_scored/cheap_aggressive_future_diagnostics.json",
    )
    parser.add_argument(
        "--future-diagnostics-md",
        default="data/model/reports/future_scored/cheap_aggressive_future_diagnostics.md",
    )
    parser.add_argument(
        "--future-u23-nonbig5-json",
        default="data/model/reports/future_scored/cheap_aggressive_future_benchmark_u23_nonbig5_report.json",
    )
    parser.add_argument(
        "--future-u23-nonbig5-md",
        default="data/model/reports/future_scored/cheap_aggressive_future_benchmark_u23_nonbig5_report.md",
    )
    parser.add_argument("--future-score-col", default="future_scout_blend_score")
    parser.add_argument("--future-cohort-min-labeled", type=int, default=25)
    parser.add_argument("--future-k-values", default="10,25,50")
    parser.add_argument("--run-promotion", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--champion-metrics", default="data/model/champion_predictions_2024-25.metrics.json")
    parser.add_argument("--champion-label", default="champion")
    parser.add_argument(
        "--candidate-metrics",
        default="data/model/champion_predictions_2024-25.metrics.json",
    )
    parser.add_argument(
        "--candidate-holdout-glob",
        default="data/model/reports/low_value_contract_holdout/cheap_aggressive/*.holdout_*.metrics.json",
    )
    parser.add_argument(
        "--reference-holdout-glob",
        default="data/model/reports/holdout_compare/full_*/*.holdout_*.metrics.json",
    )
    parser.add_argument(
        "--champion-future-benchmark-json",
        default="data/model/reports/future_value_benchmark_report.json",
    )
    parser.add_argument("--candidate-label", default="cheap_aggressive_future_scored")
    parser.add_argument("--reference-label", default="full_reference")
    parser.add_argument(
        "--comparison-out-json",
        default="data/model/reports/candidate_promotion/future_scored_candidate_vs_champion.json",
    )
    parser.add_argument(
        "--comparison-out-md",
        default="data/model/reports/candidate_promotion/future_scored_candidate_vs_champion.md",
    )
    parser.add_argument("--benchmark-out-json", default="data/model/reports/market_value_benchmark_report.json")
    parser.add_argument("--benchmark-out-md", default="data/model/reports/market_value_benchmark_report.md")
    parser.add_argument("--manifest-out", default="data/model/model_manifest.json")
    parser.add_argument("--env-out", default="data/model/model_artifacts.env")
    parser.add_argument("--bundle-label", default="future_scored_market_value_bundle")
    parser.add_argument("--onboarding-json", default="data/model/onboarding/non_big5_onboarding_report.json")
    parser.add_argument("--ablation-bundle", default="data/model/ablation/ablation_bundle_2024-25.json")
    parser.add_argument("--promote-on-pass", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--summary-json", default="data/model/reports/future_scored/future_refresh_summary.json")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    future_k_values = [int(token.strip()) for token in str(args.future_k_values).split(",") if token.strip()]
    promotion_args = [
        "--champion-metrics",
        args.champion_metrics,
        "--champion-label",
        args.champion_label,
        "--candidate-predictions",
        args.scored_test_output,
        "--candidate-val-predictions",
        args.scored_val_output,
        "--candidate-metrics",
        args.candidate_metrics,
        "--candidate-holdout-glob",
        args.candidate_holdout_glob,
        "--candidate-label",
        args.candidate_label,
        "--reference-holdout-glob",
        args.reference_holdout_glob,
        "--reference-label",
        args.reference_label,
        "--comparison-out-json",
        args.comparison_out_json,
        "--comparison-out-md",
        args.comparison_out_md,
        "--onboarding-json",
        args.onboarding_json,
        "--ablation-bundle",
        args.ablation_bundle,
        "--benchmark-out-json",
        args.benchmark_out_json,
        "--benchmark-out-md",
        args.benchmark_out_md,
        "--manifest-out",
        args.manifest_out,
        "--env-out",
        args.env_out,
        "--label",
        args.bundle_label,
        "--primary-role",
        "future_shortlist",
    ]
    _append_future_promotion_args(
        promotion_args,
        candidate_future_benchmark_json=args.future_benchmark_json,
        champion_future_benchmark_json=args.champion_future_benchmark_json,
    )
    if args.promote_on_pass:
        promotion_args.append("--promote-on-pass")

    run_future_data_refresh(
        import_dir=args.import_dir,
        staging_dir=args.staging_dir,
        season_filter=args.season_filter,
        source_dir=args.source_dir,
        combined_dir=args.combined_dir,
        country_root=args.country_root,
        season_root=args.season_root,
        organization_manifest=args.organization_manifest,
        clean_targets=not args.no_clean_targets,
        dataset_output=args.dataset_output,
        clean_output=args.clean_output,
        external_dir=args.external_dir,
        clean_min_minutes=float(args.clean_min_minutes),
        future_targets_output=args.future_targets_output,
        future_audit_json=args.future_audit_json,
        future_audit_csv=args.future_audit_csv,
        future_source_glob=args.future_source_glob,
        min_next_minutes=float(args.min_next_minutes),
        base_val_predictions=args.base_val_predictions,
        base_test_predictions=args.base_test_predictions,
        scored_val_output=args.scored_val_output,
        scored_test_output=args.scored_test_output,
        diagnostics_output=args.diagnostics_output,
        label_mode=args.label_mode,
        k_eval=int(args.k_eval),
        scoring_min_minutes=float(args.scoring_min_minutes),
        scoring_max_age=None if float(args.scoring_max_age) < 0 else float(args.scoring_max_age),
        scoring_positions={token.strip().upper() for token in str(args.scoring_positions).split(",") if token.strip()} or None,
        scoring_include_leagues=_parse_csv_tokens(args.scoring_include_leagues),
        scoring_exclude_leagues=_parse_csv_tokens(args.scoring_exclude_leagues),
        future_benchmark_json=args.future_benchmark_json,
        future_benchmark_md=args.future_benchmark_md,
        future_diagnostics_json=args.future_diagnostics_json,
        future_diagnostics_md=args.future_diagnostics_md,
        future_u23_nonbig5_json=args.future_u23_nonbig5_json,
        future_u23_nonbig5_md=args.future_u23_nonbig5_md,
        future_score_col=args.future_score_col,
        future_cohort_min_labeled=int(args.future_cohort_min_labeled),
        future_k_values=future_k_values,
        promotion_enabled=bool(args.run_promotion),
        promotion_args=promotion_args,
        summary_json=args.summary_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
