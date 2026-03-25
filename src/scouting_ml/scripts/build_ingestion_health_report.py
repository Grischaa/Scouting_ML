from __future__ import annotations

import argparse
import json
from pathlib import Path

from scouting_ml.reporting.operator_health import (
    INGESTION_HEALTH_CSV,
    INGESTION_HEALTH_JSON,
    regenerate_ingestion_health_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the league-season ingestion health report used by operator readiness surfaces."
    )
    parser.add_argument("--clean-dataset", default=None, help="Optional clean dataset parquet used to enrich provider snapshot dates.")
    parser.add_argument("--out-csv", default=str(INGESTION_HEALTH_CSV))
    parser.add_argument("--out-json", default=str(INGESTION_HEALTH_JSON))
    args = parser.parse_args()

    payload = regenerate_ingestion_health_report(
        clean_dataset_path=Path(args.clean_dataset) if args.clean_dataset else None,
        out_csv=Path(args.out_csv),
        out_json=Path(args.out_json),
    )
    summary = payload.get("summary", {})
    print(f"[ingestion-health] wrote csv -> {args.out_csv}")
    print(f"[ingestion-health] wrote json -> {args.out_json}")
    print("[ingestion-health] status counts -> " + json.dumps(summary.get("status_counts", {}), sort_keys=True))


if __name__ == "__main__":
    main()
