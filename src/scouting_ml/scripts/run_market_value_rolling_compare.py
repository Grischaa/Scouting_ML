from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


@dataclass(frozen=True)
class VariantSpec:
    label: str
    metrics_json: Path
    note: str = ""


DEFAULT_VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec(
        label="active_base",
        metrics_json=Path("data/model/reports/low_value_strategy_focused/cheap_aggressive_2024-25.metrics.json"),
        note="Current active cheap_aggressive valuation configuration.",
    ),
    VariantSpec(
        label="prod60",
        metrics_json=Path("data/model/candidates/cheap_aggressive_prod60.metrics.json"),
        note="Full-budget cheap_aggressive candidate.",
    ),
    VariantSpec(
        label="no_injury_block",
        metrics_json=Path("data/model/reports/injury_feature_compare/no_injury_block.metrics.json"),
        note="cheap_aggressive-style fit without the injury block.",
    ),
)


def _run_rolling_backtest(**kwargs: Any) -> None:
    from scouting_ml.scripts.run_rolling_backtest import run_rolling_backtest

    run_rolling_backtest(**kwargs)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_variant_tokens(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _resolve_variants(tokens: Sequence[str] | None) -> list[VariantSpec]:
    available = {spec.label: spec for spec in DEFAULT_VARIANTS if spec.metrics_json.exists()}
    if not tokens:
        if not available:
            raise ValueError("No default rolling-compare variants are available on disk.")
        return [available[label] for label in available]

    out: list[VariantSpec] = []
    missing: list[str] = []
    for token in tokens:
        spec = available.get(str(token).strip())
        if spec is None:
            missing.append(str(token).strip())
            continue
        out.append(spec)
    if missing:
        raise ValueError(
            "Unknown or unavailable rolling-compare variants: " + ", ".join(missing)
        )
    return out


def _metric_value(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_variant_training_args(metrics_json: Path, *, trials_cap: int | None = None) -> dict[str, Any]:
    payload = _read_json(metrics_json)
    trials = int(payload.get("trials_per_position", 60) or 60)
    if trials_cap is not None:
        trials = min(trials, int(trials_cap))
    return {
        "trials": trials,
        "recency_half_life": float(payload.get("recency_half_life", 2.0) or 2.0),
        "under_5m_weight": float(payload.get("under_5m_weight", 1.0) or 1.0),
        "mid_5m_to_20m_weight": float(
            payload.get("mid_5m_to_20m_weight", payload.get("mid_5m_weight", 1.0)) or 1.0
        ),
        "over_20m_weight": float(payload.get("over_20m_weight", 1.0) or 1.0),
        "optimize_metric": str(payload.get("optimize_metric", "lowmid_wmape")),
        "interval_q": float(payload.get("interval_q", 0.80) or 0.80),
        "strict_leakage_guard": bool(payload.get("strict_leakage_guard", True)),
        "strict_quality_gate": bool(payload.get("strict_quality_gate", False)),
        "two_stage_band_model": bool(payload.get("two_stage_band_model", True)),
        "band_min_samples": int(payload.get("band_min_samples", 160) or 160),
        "band_blend_alpha": float(payload.get("band_blend_alpha", 0.35) or 0.35),
        "exclude_prefixes": list(payload.get("exclude_prefixes", []) or []),
        "exclude_columns": list(payload.get("exclude_columns", []) or []),
        "min_feature_coverage": _metric_value(payload, "min_feature_coverage"),
        "min_provider_feature_coverage": _metric_value(payload, "min_provider_feature_coverage"),
    }


def _build_variant_summary(
    *,
    spec: VariantSpec,
    metrics_json: Path,
    rolling_summary_json: Path,
    rolling_summary_csv: Path,
    training_args: dict[str, Any],
) -> dict[str, Any]:
    candidate_metrics = _read_json(metrics_json)
    rolling = _read_json(rolling_summary_json)
    rolling_rows = pd.read_csv(rolling_summary_csv) if rolling_summary_csv.exists() else pd.DataFrame()
    return {
        "label": spec.label,
        "note": spec.note,
        "metrics_json": str(spec.metrics_json),
        "rolling_summary_json": str(rolling_summary_json),
        "rolling_summary_csv": str(rolling_summary_csv),
        "training_args": training_args,
        "single_split_reference": {
            "test_r2": _metric_value(candidate_metrics.get("overall", {}).get("test", {}), "r2"),
            "test_wmape": _metric_value(candidate_metrics.get("overall", {}).get("test", {}), "wmape"),
            "test_under_5m_wmape": next(
                (
                    float(row.get("wmape"))
                    for row in candidate_metrics.get("segments", {}).get("test", [])
                    if str(row.get("segment")) == "under_5m" and row.get("wmape") is not None
                ),
                None,
            ),
        },
        "rolling": {
            "runs": int(rolling.get("runs", 0)),
            "mean_test_r2": _metric_value(rolling, "mean_test_r2"),
            "std_test_r2": _metric_value(rolling, "std_test_r2"),
            "mean_test_wmape": _metric_value(rolling, "mean_test_wmape"),
            "std_test_wmape": _metric_value(rolling, "std_test_wmape"),
            "mean_test_lowmid_weighted_wmape": _metric_value(rolling, "mean_test_lowmid_weighted_wmape"),
            "std_test_lowmid_weighted_wmape": _metric_value(rolling, "std_test_lowmid_weighted_wmape"),
            "mean_test_segment_weighted_wmape": _metric_value(rolling, "mean_test_segment_weighted_wmape"),
            "quality_gate": rolling.get("quality_gate", {}),
            "skipped_runs": rolling.get("skipped_runs", []),
            "per_season": rolling_rows.to_dict(orient="records"),
        },
    }


def run_market_value_rolling_compare(
    *,
    dataset: str,
    out_dir: str,
    variants: Sequence[str] | None = None,
    compare_json: str | None = None,
    compare_csv: str | None = None,
    compare_md: str | None = None,
    trials_cap: int | None = None,
    test_seasons: Sequence[str] | None = None,
    min_train_seasons: int = 2,
    min_test_samples: int = 300,
    min_test_under5m_samples: int = 50,
    min_test_over20m_samples: int = 25,
    skip_incomplete_test_seasons: bool = True,
) -> dict[str, Any]:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    selected_specs = _resolve_variants(variants)

    variant_payloads: list[dict[str, Any]] = []
    compare_rows: list[dict[str, Any]] = []

    for spec in selected_specs:
        training_args = _load_variant_training_args(spec.metrics_json, trials_cap=trials_cap)
        variant_dir = out_root / spec.label
        _run_rolling_backtest(
            dataset_path=dataset,
            out_dir=str(variant_dir),
            trials=int(training_args["trials"]),
            recency_half_life=float(training_args["recency_half_life"]),
            under_5m_weight=float(training_args["under_5m_weight"]),
            mid_5m_to_20m_weight=float(training_args["mid_5m_to_20m_weight"]),
            over_20m_weight=float(training_args["over_20m_weight"]),
            optimize_metric=str(training_args["optimize_metric"]),
            interval_q=float(training_args["interval_q"]),
            strict_leakage_guard=bool(training_args["strict_leakage_guard"]),
            strict_quality_gate=bool(training_args["strict_quality_gate"]),
            two_stage_band_model=bool(training_args["two_stage_band_model"]),
            band_min_samples=int(training_args["band_min_samples"]),
            band_blend_alpha=float(training_args["band_blend_alpha"]),
            min_test_samples=int(min_test_samples),
            min_test_under5m_samples=int(min_test_under5m_samples),
            min_test_over20m_samples=int(min_test_over20m_samples),
            min_train_seasons=int(min_train_seasons),
            test_seasons=list(test_seasons or []),
            skip_incomplete_test_seasons=bool(skip_incomplete_test_seasons),
            exclude_prefixes=list(training_args["exclude_prefixes"]),
            exclude_columns=list(training_args["exclude_columns"]),
        )
        rolling_json = variant_dir / "rolling_backtest_summary.json"
        rolling_csv = variant_dir / "rolling_backtest_summary.csv"
        variant_payload = _build_variant_summary(
            spec=spec,
            metrics_json=spec.metrics_json,
            rolling_summary_json=rolling_json,
            rolling_summary_csv=rolling_csv,
            training_args=training_args,
        )
        variant_payloads.append(variant_payload)
        compare_rows.append(
            {
                "label": spec.label,
                "note": spec.note,
                "single_split_test_r2": variant_payload["single_split_reference"]["test_r2"],
                "single_split_test_wmape": variant_payload["single_split_reference"]["test_wmape"],
                "rolling_runs": variant_payload["rolling"]["runs"],
                "rolling_mean_test_r2": variant_payload["rolling"]["mean_test_r2"],
                "rolling_std_test_r2": variant_payload["rolling"]["std_test_r2"],
                "rolling_mean_test_wmape": variant_payload["rolling"]["mean_test_wmape"],
                "rolling_std_test_wmape": variant_payload["rolling"]["std_test_wmape"],
                "rolling_mean_lowmid_wmape": variant_payload["rolling"]["mean_test_lowmid_weighted_wmape"],
                "rolling_gate_passed": bool(variant_payload["rolling"]["quality_gate"].get("passed", True)),
                "exclude_prefixes": ",".join(variant_payload["training_args"]["exclude_prefixes"]),
                "exclude_columns": ",".join(variant_payload["training_args"]["exclude_columns"]),
            }
        )

    def _best_by(key: str, *, higher_is_better: bool) -> str | None:
        eligible = [row for row in compare_rows if row.get(key) is not None]
        if not eligible:
            return None
        ranked = sorted(eligible, key=lambda row: float(row[key]), reverse=higher_is_better)
        return str(ranked[0]["label"])

    payload = {
        "dataset": dataset,
        "variants": variant_payloads,
        "decision": {
            "winner_by_rolling_mean_test_r2": _best_by("rolling_mean_test_r2", higher_is_better=True),
            "winner_by_rolling_mean_test_wmape": _best_by("rolling_mean_test_wmape", higher_is_better=False),
            "winner_by_rolling_mean_lowmid_wmape": _best_by("rolling_mean_lowmid_wmape", higher_is_better=False),
            "note": "Rolling compare ranks variants by seasonal stability, not just one split.",
        },
    }

    compare_json_path = Path(compare_json or (out_root / "rolling_compare_summary.json"))
    compare_csv_path = Path(compare_csv or (out_root / "rolling_compare_summary.csv"))
    compare_md_path = Path(compare_md or (out_root / "rolling_compare_summary.md"))
    compare_json_path.parent.mkdir(parents=True, exist_ok=True)
    compare_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(compare_rows).to_csv(compare_csv_path, index=False)

    lines = [
        "# Market Value Rolling Compare",
        "",
        f"- Dataset: `{dataset}`",
        f"- Variants: `{', '.join(row['label'] for row in compare_rows)}`",
        "",
        "| Variant | Single-split test R2 | Rolling mean test R2 | Rolling mean test WMAPE | Rolling mean lowmid WMAPE | Note |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in compare_rows:
        lines.append(
            "| "
            f"{row['label']} | {float(row['single_split_test_r2']):.2%} | "
            f"{float(row['rolling_mean_test_r2']):.2%} | "
            f"{float(row['rolling_mean_test_wmape']):.2%} | "
            f"{float(row['rolling_mean_lowmid_wmape']):.2%} | {row['note']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Winner by rolling mean test R2: `{payload['decision']['winner_by_rolling_mean_test_r2']}`",
            f"- Winner by rolling mean test WMAPE: `{payload['decision']['winner_by_rolling_mean_test_wmape']}`",
            f"- Winner by rolling mean lowmid WMAPE: `{payload['decision']['winner_by_rolling_mean_lowmid_wmape']}`",
        ]
    )
    compare_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rolling backtests across named market-value candidate variants.")
    parser.add_argument("--dataset", default="data/model/tm_context_candidate_clean.parquet")
    parser.add_argument("--out-dir", default="data/model/reports/rolling_compare")
    parser.add_argument(
        "--variants",
        default="",
        help="Optional comma-separated variant labels. Defaults to the available built-ins.",
    )
    parser.add_argument("--compare-json", default="")
    parser.add_argument("--compare-csv", default="")
    parser.add_argument("--compare-md", default="")
    parser.add_argument("--trials-cap", type=int, default=None)
    parser.add_argument("--test-seasons", default="")
    parser.add_argument("--min-train-seasons", type=int, default=2)
    parser.add_argument("--min-test-samples", type=int, default=300)
    parser.add_argument("--min-test-under5m-samples", type=int, default=50)
    parser.add_argument("--min-test-over20m-samples", type=int, default=25)
    parser.add_argument(
        "--skip-incomplete-test-seasons",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    payload = run_market_value_rolling_compare(
        dataset=args.dataset,
        out_dir=args.out_dir,
        variants=_parse_variant_tokens(args.variants),
        compare_json=args.compare_json or None,
        compare_csv=args.compare_csv or None,
        compare_md=args.compare_md or None,
        trials_cap=args.trials_cap,
        test_seasons=_parse_variant_tokens(args.test_seasons),
        min_train_seasons=args.min_train_seasons,
        min_test_samples=args.min_test_samples,
        min_test_under5m_samples=args.min_test_under5m_samples,
        min_test_over20m_samples=args.min_test_over20m_samples,
        skip_incomplete_test_seasons=args.skip_incomplete_test_seasons,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
