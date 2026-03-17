from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scouting_ml.scripts.build_external_feature_coverage_audit import build_external_feature_coverage_audit


INJURY_DERIVED_COLUMNS = (
    "availability_risk_score",
    "durability_score",
    "high_value_readiness_score",
)


@dataclass(frozen=True)
class VariantConfig:
    name: str
    exclude_prefixes: tuple[str, ...] = ()
    exclude_columns: tuple[str, ...] = ()
    note: str = ""


VARIANTS: tuple[VariantConfig, ...] = (
    VariantConfig(
        name="full",
        note="All currently available modeling features.",
    ),
    VariantConfig(
        name="no_injury_block",
        exclude_prefixes=("injury_",),
        exclude_columns=INJURY_DERIVED_COLUMNS,
        note="Remove injury-history features and direct injury-derived composites.",
    ),
)


def _train_market_value_main(**kwargs: Any) -> None:
    from scouting_ml.models.train_market_value_full import main as train_market_value_main

    train_market_value_main(**kwargs)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _segment_metric(metrics: dict[str, Any], split: str, segment: str, key: str) -> float | None:
    for row in metrics.get("segments", {}).get(split, []):
        if str(row.get("segment")) == str(segment):
            value = row.get(key)
            return float(value) if value is not None else None
    return None


def _run_variant(
    *,
    dataset: str,
    val_season: str,
    test_season: str,
    out_dir: Path,
    variant: VariantConfig,
    trials: int,
    holdout_trials: int,
    optimize_metric: str,
    under_5m_weight: float,
    mid_5m_to_20m_weight: float,
    over_20m_weight: float,
) -> dict[str, Any]:
    predictions_path = out_dir / f"{variant.name}.csv"
    val_predictions_path = out_dir / f"{variant.name}_val.csv"
    metrics_path = out_dir / f"{variant.name}.metrics.json"
    quality_path = out_dir / f"{variant.name}.quality.json"

    _train_market_value_main(
        dataset_path=dataset,
        val_season=val_season,
        test_season=test_season,
        output_path=str(predictions_path),
        val_output_path=str(val_predictions_path),
        metrics_output_path=str(metrics_path),
        quality_output_path=str(quality_path),
        n_optuna_trials=int(trials),
        holdout_n_optuna_trials=int(holdout_trials),
        optimize_metric=optimize_metric,
        under_5m_weight=float(under_5m_weight),
        mid_5m_to_20m_weight=float(mid_5m_to_20m_weight),
        over_20m_weight=float(over_20m_weight),
        exclude_prefixes=list(variant.exclude_prefixes),
        exclude_columns=list(variant.exclude_columns),
        save_shap_artifacts=False,
        league_holdouts=[],
        optuna_study_namespace=f"injury_block_compare_{variant.name}",
        optuna_load_if_exists=False,
    )
    metrics = _read_json(metrics_path)
    return {
        "name": variant.name,
        "note": variant.note,
        "exclude_prefixes": list(variant.exclude_prefixes),
        "exclude_columns": list(variant.exclude_columns),
        "artifacts": {
            "predictions_csv": str(predictions_path),
            "val_predictions_csv": str(val_predictions_path),
            "metrics_json": str(metrics_path),
            "quality_json": str(quality_path),
        },
        "metrics": {
            "val_r2": float(metrics["overall"]["val"]["r2"]),
            "val_wmape": float(metrics["overall"]["val"]["wmape"]),
            "test_r2": float(metrics["overall"]["test"]["r2"]),
            "test_wmape": float(metrics["overall"]["test"]["wmape"]),
            "test_under_5m_wmape": _segment_metric(metrics, "test", "under_5m", "wmape"),
            "test_5m_to_20m_wmape": _segment_metric(metrics, "test", "5m_to_20m", "wmape"),
        },
    }


def run_injury_feature_block_compare(
    *,
    dataset: str,
    val_season: str,
    test_season: str,
    out_dir: str,
    audit_json: str | None = None,
    audit_csv: str | None = None,
    audit_md: str | None = None,
    compare_json: str | None = None,
    compare_md: str | None = None,
    trials: int = 5,
    holdout_trials: int = 1,
    optimize_metric: str = "lowmid_wmape",
    under_5m_weight: float = 2.5,
    mid_5m_to_20m_weight: float = 1.75,
    over_20m_weight: float = 0.6,
    min_injury_row_coverage: float = 0.05,
) -> dict[str, Any]:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    audit = build_external_feature_coverage_audit(
        dataset_path=dataset,
        out_json=audit_json,
        out_csv=audit_csv,
        out_md=audit_md,
    )
    injury_summary = audit["family_overall"]["injury"]

    payload: dict[str, Any] = {
        "dataset": dataset,
        "val_season": val_season,
        "test_season": test_season,
        "trials": int(trials),
        "holdout_trials": int(holdout_trials),
        "optimize_metric": optimize_metric,
        "feature_audit": {
            "injury": injury_summary,
            "availability": audit["family_overall"]["availability"],
            "fixture": audit["family_overall"]["fixture"],
        },
        "variants": [],
        "decision": {},
    }

    if float(injury_summary.get("row_coverage_share", 0.0) or 0.0) < float(min_injury_row_coverage):
        payload["decision"] = {
            "status": "skipped",
            "reason": (
                f"injury row coverage {float(injury_summary.get('row_coverage_share', 0.0)):.4f} "
                f"< min_injury_row_coverage {float(min_injury_row_coverage):.4f}"
            ),
        }
    else:
        results = [
            _run_variant(
                dataset=dataset,
                val_season=val_season,
                test_season=test_season,
                out_dir=out_root,
                variant=variant,
                trials=trials,
                holdout_trials=holdout_trials,
                optimize_metric=optimize_metric,
                under_5m_weight=under_5m_weight,
                mid_5m_to_20m_weight=mid_5m_to_20m_weight,
                over_20m_weight=over_20m_weight,
            )
            for variant in VARIANTS
        ]
        payload["variants"] = results
        by_name = {item["name"]: item for item in results}
        full = by_name["full"]["metrics"]
        no_injury = by_name["no_injury_block"]["metrics"]
        payload["decision"] = {
            "status": "completed",
            "winner_by_test_wmape": "full" if full["test_wmape"] <= no_injury["test_wmape"] else "no_injury_block",
            "winner_by_test_r2": "full" if full["test_r2"] >= no_injury["test_r2"] else "no_injury_block",
            "delta_full_minus_no_injury": {
                "test_wmape": float(full["test_wmape"] - no_injury["test_wmape"]),
                "test_r2": float(full["test_r2"] - no_injury["test_r2"]),
                "test_under_5m_wmape": (
                    float(full["test_under_5m_wmape"] - no_injury["test_under_5m_wmape"])
                    if full["test_under_5m_wmape"] is not None and no_injury["test_under_5m_wmape"] is not None
                    else None
                ),
            },
        }

    if compare_json is None:
        compare_json = str(out_root / "injury_feature_block_compare.json")
    compare_json_path = Path(compare_json)
    compare_json_path.parent.mkdir(parents=True, exist_ok=True)
    compare_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if compare_md is None:
        compare_md = str(out_root / "injury_feature_block_compare.md")
    lines = [
        "# Injury Feature Block Compare",
        "",
        f"- Dataset: `{dataset}`",
        f"- Val/Test: `{val_season}` -> `{test_season}`",
        f"- Trials: `{trials}` (holdout trials `{holdout_trials}`)",
        "",
        "## Coverage Snapshot",
        "",
        f"- Injury row coverage: `{float(injury_summary.get('row_coverage_share', 0.0)):.2%}`",
        f"- Availability row coverage: `{float(audit['family_overall']['availability'].get('row_coverage_share', 0.0)):.2%}`",
        f"- Fixture row coverage: `{float(audit['family_overall']['fixture'].get('row_coverage_share', 0.0)):.2%}`",
        "",
    ]
    if payload["decision"].get("status") == "completed":
        lines.extend(
            [
                "## Variant Results",
                "",
                "| Variant | Test WMAPE | Test R2 | Under 5m WMAPE | Note |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for row in payload["variants"]:
            metrics = row["metrics"]
            lines.append(
                "| "
                f"{row['name']} | {metrics['test_wmape']:.2%} | {metrics['test_r2']:.2%} | "
                f"{metrics['test_under_5m_wmape']:.2%} | {row['note']} |"
            )
        lines.extend(
            [
                "",
                "## Decision",
                "",
                f"- Winner by test WMAPE: `{payload['decision']['winner_by_test_wmape']}`",
                f"- Winner by test R2: `{payload['decision']['winner_by_test_r2']}`",
            ]
        )
    else:
        lines.extend(["## Decision", "", f"- {payload['decision'].get('reason', 'comparison skipped')}"])
    Path(compare_md).write_text("\n".join(lines) + "\n", encoding="utf-8")

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the active injury feature block against an injury-free baseline.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--val-season", default="2023/24")
    parser.add_argument("--test-season", default="2024/25")
    parser.add_argument("--out-dir", default="data/model/reports/injury_feature_compare")
    parser.add_argument("--audit-json", default="")
    parser.add_argument("--audit-csv", default="")
    parser.add_argument("--audit-md", default="")
    parser.add_argument("--compare-json", default="")
    parser.add_argument("--compare-md", default="")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--holdout-trials", type=int, default=1)
    parser.add_argument("--optimize-metric", default="lowmid_wmape")
    parser.add_argument("--under-5m-weight", type=float, default=2.5)
    parser.add_argument("--mid-5m-20m-weight", type=float, default=1.75)
    parser.add_argument("--over-20m-weight", type=float, default=0.6)
    parser.add_argument("--min-injury-row-coverage", type=float, default=0.05)
    args = parser.parse_args()

    payload = run_injury_feature_block_compare(
        dataset=args.dataset,
        val_season=args.val_season,
        test_season=args.test_season,
        out_dir=args.out_dir,
        audit_json=args.audit_json or None,
        audit_csv=args.audit_csv or None,
        audit_md=args.audit_md or None,
        compare_json=args.compare_json or None,
        compare_md=args.compare_md or None,
        trials=args.trials,
        holdout_trials=args.holdout_trials,
        optimize_metric=args.optimize_metric,
        under_5m_weight=args.under_5m_weight,
        mid_5m_to_20m_weight=args.mid_5m_20m_weight,
        over_20m_weight=args.over_20m_weight,
        min_injury_row_coverage=args.min_injury_row_coverage,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
