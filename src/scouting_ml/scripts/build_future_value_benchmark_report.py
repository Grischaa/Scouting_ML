from __future__ import annotations

import argparse

from scouting_ml.reporting.future_value_benchmarks import (
    build_future_value_benchmark_payload,
    write_future_value_benchmark_report,
)


def _parse_k_values(raw: str | None) -> list[int]:
    if raw is None:
        return [10, 25, 50]
    out: list[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(float(token))
        except (TypeError, ValueError):
            continue
        if value > 0:
            out.append(value)
    return sorted({*out}) if out else [10, 25, 50]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark scouting-value predictions against next-season market-value growth targets. "
            "The script can read an existing future-target parquet or build targets in memory from the source dataset."
        )
    )
    parser.add_argument("--test-predictions", default=None)
    parser.add_argument("--val-predictions", default=None)
    parser.add_argument("--future-targets", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--score-col", default=None)
    parser.add_argument("--k-values", default="10,25,50")
    parser.add_argument("--cohort-min-labeled", type=int, default=25)
    parser.add_argument("--min-next-minutes", type=float, default=450.0)
    parser.add_argument("--min-minutes", type=float, default=900.0)
    parser.add_argument("--max-age", type=float, default=-1.0, help="Set negative to disable.")
    parser.add_argument("--non-big5-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--top-realized-limit", type=int, default=25)
    parser.add_argument("--out-json", default="data/model/reports/future_value_benchmark_report.json")
    parser.add_argument("--out-md", default="data/model/reports/future_value_benchmark_report.md")
    args = parser.parse_args()

    payload = build_future_value_benchmark_payload(
        test_predictions_path=args.test_predictions,
        val_predictions_path=args.val_predictions,
        future_targets_path=args.future_targets,
        dataset_path=args.dataset,
        score_col=args.score_col,
        k_values=_parse_k_values(args.k_values),
        cohort_min_labeled=args.cohort_min_labeled,
        min_next_minutes=args.min_next_minutes,
        min_minutes=args.min_minutes,
        max_age=None if args.max_age < 0 else args.max_age,
        non_big5_only=args.non_big5_only,
        top_realized_limit=args.top_realized_limit,
    )
    out_paths = write_future_value_benchmark_report(
        payload,
        out_json=args.out_json,
        out_md=args.out_md,
    )
    print(f"[future-benchmark] wrote json -> {out_paths['json']}")
    print(f"[future-benchmark] wrote markdown -> {out_paths['markdown']}")


if __name__ == "__main__":
    main()
