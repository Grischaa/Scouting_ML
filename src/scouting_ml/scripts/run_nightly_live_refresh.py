from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from scouting_ml.reporting.operator_health import regenerate_ingestion_health_report
from scouting_ml.scripts.run_future_data_refresh import (
    _append_future_promotion_args,
    _parse_args as _parse_future_defaults,
    run_future_data_refresh,
)


def _lane_line(role: str, section: dict[str, object] | None) -> str:
    payload = section if isinstance(section, dict) else {}
    lane_state = str(payload.get("lane_state") or "unknown")
    promotion_state = str(payload.get("promotion_state") or "advisory_only")
    label = str(payload.get("label") or role)
    return f"[nightly] {role}: {label} | {lane_state} | {promotion_state}"


def _talent_line(summary: dict[str, object] | None) -> str:
    payload = summary if isinstance(summary, dict) else {}
    label_cov = payload.get("future_label_coverage") if isinstance(payload.get("future_label_coverage"), dict) else {}
    confidence = payload.get("confidence_distribution") if isinstance(payload.get("confidence_distribution"), dict) else {}
    families = payload.get("position_family_counts") if isinstance(payload.get("position_family_counts"), dict) else {}
    family_text = ", ".join(f"{key}:{value}" for key, value in sorted(families.items())) or "none"
    return (
        "[nightly] future talent -> "
        f"labels {label_cov.get('labeled_rows', 0)}/{label_cov.get('total_rows', 0)} | "
        f"confidence high/med/low {confidence.get('high', 0)}/{confidence.get('medium', 0)}/{confidence.get('low', 0)} | "
        f"positions {family_text}"
    )


def _prefer_existing_path(preferred: str, fallback: str) -> str:
    preferred_path = Path(preferred)
    if preferred_path.exists():
        return preferred
    if fallback:
        return fallback
    return preferred


def _build_promotion_args(args: argparse.Namespace, defaults: argparse.Namespace) -> list[str]:
    candidate_metrics = _prefer_existing_path(
        defaults.candidate_metrics,
        defaults.champion_metrics,
    )
    promotion_args = [
        "--champion-metrics",
        defaults.champion_metrics,
        "--champion-label",
        defaults.champion_label,
        "--candidate-predictions",
        args.scored_test_output,
        "--candidate-val-predictions",
        args.scored_val_output,
        "--candidate-metrics",
        candidate_metrics,
        "--candidate-holdout-glob",
        defaults.candidate_holdout_glob,
        "--reference-holdout-glob",
        defaults.reference_holdout_glob,
        "--candidate-label",
        defaults.candidate_label,
        "--reference-label",
        defaults.reference_label,
        "--comparison-out-json",
        defaults.comparison_out_json,
        "--comparison-out-md",
        defaults.comparison_out_md,
        "--benchmark-out-json",
        defaults.benchmark_out_json,
        "--benchmark-out-md",
        defaults.benchmark_out_md,
        "--manifest-out",
        args.manifest_out,
        "--env-out",
        args.env_out,
        "--label",
        args.bundle_label,
        "--primary-role",
        "future_shortlist",
        "--onboarding-json",
        defaults.onboarding_json,
        "--ablation-bundle",
        defaults.ablation_bundle,
        "--promote-on-pass" if args.promote_on_pass else "--no-promote-on-pass",
    ]
    _append_future_promotion_args(
        promotion_args,
        candidate_future_benchmark_json=defaults.future_benchmark_json,
        champion_future_benchmark_json=defaults.champion_future_benchmark_json,
    )
    return promotion_args


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    defaults = _parse_future_defaults([])
    parser = argparse.ArgumentParser(
        description=(
            "Nightly/live operator wrapper: refresh current-season imports, future-scored outputs, "
            "and ingestion health with one command."
        )
    )
    parser.add_argument("--import-dir", default=defaults.import_dir)
    parser.add_argument("--season-filter", default=defaults.season_filter)
    parser.add_argument("--summary-json", default=defaults.summary_json)
    parser.add_argument("--run-promotion", action=argparse.BooleanOptionalAction, default=defaults.run_promotion)
    parser.add_argument("--manifest-out", default=defaults.manifest_out)
    parser.add_argument("--env-out", default=defaults.env_out)
    parser.add_argument("--bundle-label", default=defaults.bundle_label)
    parser.add_argument(
        "--promote-on-pass",
        action=argparse.BooleanOptionalAction,
        default=defaults.promote_on_pass,
    )
    parser.add_argument("--scored-val-output", default=defaults.scored_val_output)
    parser.add_argument("--scored-test-output", default=defaults.scored_test_output)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    defaults = _parse_future_defaults([])
    future_k_values = [int(token.strip()) for token in str(defaults.future_k_values).split(",") if token.strip()]
    promotion_args = _build_promotion_args(args, defaults)
    base_val_predictions = _prefer_existing_path(
        defaults.base_val_predictions,
        "data/model/champion_predictions_2024-25_val.csv",
    )
    base_test_predictions = _prefer_existing_path(
        defaults.base_test_predictions,
        "data/model/champion_predictions_2024-25.csv",
    )

    summary = run_future_data_refresh(
        import_dir=args.import_dir,
        staging_dir=defaults.staging_dir,
        season_filter=args.season_filter,
        source_dir=defaults.source_dir,
        combined_dir=defaults.combined_dir,
        country_root=defaults.country_root,
        season_root=defaults.season_root,
        organization_manifest=defaults.organization_manifest,
        clean_targets=not defaults.no_clean_targets,
        dataset_output=defaults.dataset_output,
        clean_output=defaults.clean_output,
        external_dir=defaults.external_dir,
        clean_min_minutes=defaults.clean_min_minutes,
        future_targets_output=defaults.future_targets_output,
        future_audit_json=defaults.future_audit_json,
        future_audit_csv=defaults.future_audit_csv,
        future_source_glob=defaults.future_source_glob,
        min_next_minutes=defaults.min_next_minutes,
        base_val_predictions=base_val_predictions,
        base_test_predictions=base_test_predictions,
        scored_val_output=args.scored_val_output,
        scored_test_output=args.scored_test_output,
        diagnostics_output=defaults.diagnostics_output,
        label_mode=defaults.label_mode,
        k_eval=defaults.k_eval,
        scoring_min_minutes=defaults.scoring_min_minutes,
        scoring_max_age=None if float(defaults.scoring_max_age) < 0 else defaults.scoring_max_age,
        scoring_positions={token.strip().upper() for token in str(defaults.scoring_positions).split(",") if token.strip()}
        or None,
        scoring_include_leagues={token.strip().casefold() for token in str(defaults.scoring_include_leagues).split(",") if token.strip()}
        or None,
        scoring_exclude_leagues={token.strip().casefold() for token in str(defaults.scoring_exclude_leagues).split(",") if token.strip()}
        or None,
        future_benchmark_json=defaults.future_benchmark_json,
        future_benchmark_md=defaults.future_benchmark_md,
        future_diagnostics_json=defaults.future_diagnostics_json,
        future_diagnostics_md=defaults.future_diagnostics_md,
        future_u23_nonbig5_json=defaults.future_u23_nonbig5_json,
        future_u23_nonbig5_md=defaults.future_u23_nonbig5_md,
        future_score_col=defaults.future_score_col,
        future_cohort_min_labeled=defaults.future_cohort_min_labeled,
        future_k_values=future_k_values,
        promotion_enabled=args.run_promotion,
        promotion_args=promotion_args,
        summary_json=args.summary_json,
    )

    clean_meta = summary.get("artifacts", {}).get("clean_output") if isinstance(summary.get("artifacts"), dict) else {}
    clean_path = Path(str(clean_meta.get("path"))) if isinstance(clean_meta, dict) and clean_meta.get("path") else None
    ingestion_payload = regenerate_ingestion_health_report(clean_dataset_path=clean_path if clean_path and clean_path.exists() else None)

    print(f"[nightly] summary -> {Path(args.summary_json).resolve()}")
    print(f"[nightly] operator health -> {ingestion_payload['_meta']['json_path']}")
    if args.run_promotion:
        print(f"[nightly] manifest -> {Path(args.manifest_out).resolve()}")
    lanes = summary.get("artifact_lanes") or {}
    if isinstance(lanes, dict):
        for role in ("valuation", "future_shortlist"):
            if role in lanes:
                print(_lane_line(role, lanes.get(role)))
    else:
        print("[nightly] promotion skipped; no artifact lane update emitted.")
    talent_summary = summary.get("talent_summary") if isinstance(summary, dict) else None
    if isinstance(talent_summary, dict):
        print(_talent_line(talent_summary.get("test")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
