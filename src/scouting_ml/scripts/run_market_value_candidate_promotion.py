from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from scouting_ml.reporting.market_value_benchmarks import (
    build_market_value_benchmark_payload,
    write_market_value_benchmark_report,
)
from scouting_ml.reporting.market_value_candidate_promotion import (
    build_candidate_promotion_payload,
    candidate_passes_promotion_gates,
    write_candidate_promotion_report,
)
from scouting_ml.scripts.lock_market_value_artifacts import build_lock_bundle


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a market-value candidate against the active champion and optionally promote it."
    )
    parser.add_argument("--champion-metrics", default="data/model/champion_predictions_2024-25.metrics.json")
    parser.add_argument("--champion-label", default="champion")
    parser.add_argument("--candidate-predictions", required=True)
    parser.add_argument("--candidate-val-predictions", required=True)
    parser.add_argument("--candidate-metrics", required=True)
    parser.add_argument("--candidate-holdout-glob", required=True)
    parser.add_argument("--candidate-future-benchmark-json", default=None)
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument(
        "--reference-holdout-glob",
        default="data/model/reports/holdout_compare/full_*/*.holdout_*.metrics.json",
        help="Reference holdout glob the candidate must beat on weighted WMAPE.",
    )
    parser.add_argument("--reference-label", default="full_reference")
    parser.add_argument("--champion-future-benchmark-json", default=None)
    parser.add_argument(
        "--comparison-out-json",
        default="data/model/reports/candidate_promotion/candidate_vs_champion.json",
    )
    parser.add_argument(
        "--comparison-out-md",
        default="data/model/reports/candidate_promotion/candidate_vs_champion.md",
    )
    parser.add_argument(
        "--require-test-wmape-improvement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require candidate test WMAPE to be <= champion test WMAPE.",
    )
    parser.add_argument(
        "--require-under-20m-wmape-improvement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require candidate under-20m test WMAPE to be <= champion under-20m test WMAPE.",
    )
    parser.add_argument(
        "--require-holdout-wmape-improvement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require candidate weighted holdout WMAPE to be <= reference holdout weighted WMAPE.",
    )
    parser.add_argument(
        "--promote-on-pass",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When gates pass, refresh the benchmark report and lock bundle to the candidate artifacts.",
    )
    parser.add_argument(
        "--require-future-benchmark",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require candidate future-benchmark coverage + precision-vs-base gates on the labeled validation split.",
    )
    parser.add_argument(
        "--require-future-precision-vs-champion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require candidate future-benchmark precision@k to be >= the champion future benchmark precision@k.",
    )
    parser.add_argument("--future-split", default="val", choices=["val", "test"])
    parser.add_argument("--future-label-key", default="positive_growth", choices=["positive_growth", "growth_gt25pct"])
    parser.add_argument("--future-k", type=int, default=25)
    parser.add_argument("--future-min-label-coverage", type=float, default=0.25)
    parser.add_argument("--onboarding-json", default="data/model/onboarding/non_big5_onboarding_report.json")
    parser.add_argument("--ablation-bundle", default="data/model/ablation/ablation_bundle_2024-25.json")
    parser.add_argument("--benchmark-out-json", default="data/model/reports/market_value_benchmark_report.json")
    parser.add_argument("--benchmark-out-md", default="data/model/reports/market_value_benchmark_report.md")
    parser.add_argument("--manifest-out", default="data/model/model_manifest.json")
    parser.add_argument("--env-out", default="data/model/model_artifacts.env")
    parser.add_argument("--label", default="market_value_candidate_bundle")
    parser.add_argument("--primary-role", choices=["valuation", "future_shortlist"], default="valuation")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    payload = build_candidate_promotion_payload(
        champion_metrics_path=args.champion_metrics,
        candidate_metrics_path=args.candidate_metrics,
        candidate_holdout_glob=args.candidate_holdout_glob,
        reference_holdout_glob=args.reference_holdout_glob,
        candidate_future_benchmark_path=args.candidate_future_benchmark_json,
        champion_future_benchmark_path=args.champion_future_benchmark_json,
        champion_label=args.champion_label,
        candidate_label=args.candidate_label,
        reference_label=args.reference_label,
        require_test_wmape_improvement=bool(args.require_test_wmape_improvement),
        require_under_20m_wmape_improvement=bool(args.require_under_20m_wmape_improvement),
        require_holdout_wmape_improvement=bool(args.require_holdout_wmape_improvement),
        require_future_benchmark=bool(args.require_future_benchmark),
        require_future_precision_vs_champion=bool(args.require_future_precision_vs_champion),
        future_split=args.future_split,
        future_label_key=args.future_label_key,
        future_k=int(args.future_k),
        future_min_label_coverage=float(args.future_min_label_coverage),
    )
    report_paths = write_candidate_promotion_report(
        payload,
        out_json=args.comparison_out_json,
        out_md=args.comparison_out_md,
    )
    print(f"[promotion] wrote comparison json -> {Path(report_paths['json'])}")
    print(f"[promotion] wrote comparison md   -> {Path(report_paths['markdown'])}")

    passed = candidate_passes_promotion_gates(payload)
    print(f"[promotion] passed gates -> {passed}")
    print(f"[promotion] reason -> {payload['decision']['reason']}")

    if not args.promote_on_pass:
        return 0
    if not passed:
        print("[promotion] skip benchmark refresh + lock bundle (gates failed)")
        return 2

    benchmark_payload = build_market_value_benchmark_payload(
        metrics_path=args.candidate_metrics,
        predictions_path=args.candidate_predictions,
        holdout_metrics_glob=args.candidate_holdout_glob,
        onboarding_json_path=args.onboarding_json,
        ablation_bundle_path=args.ablation_bundle,
    )
    benchmark_paths = write_market_value_benchmark_report(
        benchmark_payload,
        out_json=args.benchmark_out_json,
        out_md=args.benchmark_out_md,
    )
    print(f"[promotion] wrote benchmark json -> {Path(benchmark_paths['json'])}")
    print(f"[promotion] wrote benchmark md   -> {Path(benchmark_paths['markdown'])}")

    build_lock_bundle(
        test_predictions=Path(args.candidate_predictions),
        val_predictions=Path(args.candidate_val_predictions),
        metrics_path=Path(args.candidate_metrics),
        manifest_out=Path(args.manifest_out),
        env_out=Path(args.env_out),
        strict_artifacts=True,
        label=args.label,
        primary_role=args.primary_role,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
