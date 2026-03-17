from __future__ import annotations

import argparse
import json
from pathlib import Path

from scouting_ml.reporting.future_value_diagnostics import (
    build_future_value_diagnostics_payload,
    write_future_value_diagnostics_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a league/position/value-segment diagnostics report from an existing future-value benchmark JSON."
        )
    )
    parser.add_argument(
        "--benchmark-json",
        default="data/model/reports/future_scored/cheap_aggressive_future_benchmark_report.json",
    )
    parser.add_argument(
        "--out-json",
        default="data/model/reports/future_scored/cheap_aggressive_future_diagnostics.json",
    )
    parser.add_argument(
        "--out-md",
        default="data/model/reports/future_scored/cheap_aggressive_future_diagnostics.md",
    )
    parser.add_argument("--k", type=int, default=25)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark_json)
    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    diagnostics = build_future_value_diagnostics_payload(
        payload,
        source_benchmark_json=str(benchmark_path),
        k=args.k,
        top_n=args.top_n,
    )
    out_paths = write_future_value_diagnostics_report(
        diagnostics,
        out_json=args.out_json,
        out_md=args.out_md,
    )
    print(f"[future-diagnostics] wrote json -> {out_paths['json']}")
    print(f"[future-diagnostics] wrote markdown -> {out_paths['markdown']}")


if __name__ == "__main__":
    main()
