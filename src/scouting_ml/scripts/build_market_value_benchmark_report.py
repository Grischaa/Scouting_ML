from __future__ import annotations

import argparse
from pathlib import Path

from scouting_ml.reporting.market_value_benchmarks import (
    build_market_value_benchmark_payload,
    write_market_value_benchmark_report,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a multi-league benchmark report from current model, holdout, onboarding, and ablation artifacts."
    )
    parser.add_argument("--metrics", default="data/model/big5_predictions_full_v2.metrics.json")
    parser.add_argument("--predictions", default="data/model/big5_predictions_full_v2.csv")
    parser.add_argument("--holdout-metrics-glob", default="data/model/**/*.holdout_*.metrics.json")
    parser.add_argument(
        "--onboarding-json",
        default="data/model/onboarding/non_big5_onboarding_report.json",
    )
    parser.add_argument(
        "--ablation-bundle",
        default=None,
        help="Optional explicit ablation bundle path. Defaults to latest matching ablation_bundle_*.json.",
    )
    parser.add_argument(
        "--ablation-glob",
        default="data/model/ablation/**/ablation_bundle_*.json",
        help="Glob used when --ablation-bundle is not provided.",
    )
    parser.add_argument(
        "--out-json",
        default="data/model/reports/market_value_benchmark_report.json",
    )
    parser.add_argument(
        "--out-md",
        default="data/model/reports/market_value_benchmark_report.md",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = build_market_value_benchmark_payload(
        metrics_path=args.metrics,
        predictions_path=args.predictions,
        holdout_metrics_glob=args.holdout_metrics_glob,
        onboarding_json_path=args.onboarding_json,
        ablation_bundle_path=args.ablation_bundle,
        ablation_glob=args.ablation_glob,
    )
    paths = write_market_value_benchmark_report(
        payload,
        out_json=args.out_json,
        out_md=args.out_md,
    )
    print(f"[bench] wrote json -> {Path(paths['json'])}")
    print(f"[bench] wrote md   -> {Path(paths['markdown'])}")
    print(
        "[bench] holdouts -> "
        f"{payload['league_holdout']['summary'].get('ok_count', 0)} ok / "
        f"{payload['league_holdout']['summary'].get('total', 0)} total"
    )


if __name__ == "__main__":
    main()
