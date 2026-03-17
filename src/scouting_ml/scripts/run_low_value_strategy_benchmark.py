from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from scouting_ml.reporting.market_value_experiment_matrix import (
    build_overall_summary,
    build_slice_matrix,
    decorate_slice_matrix,
    load_prediction_frame,
    pick_best_overall,
    safe_float,
    season_slug,
)


@dataclass(frozen=True)
class LowValueStrategy:
    name: str
    note: str
    under_5m_weight: float
    mid_5m_to_20m_weight: float
    over_20m_weight: float
    optimize_metric: str


DEFAULT_STRATEGIES: dict[str, LowValueStrategy] = {
    "baseline_full": LowValueStrategy(
        name="baseline_full",
        note="Current balanced baseline.",
        under_5m_weight=1.0,
        mid_5m_to_20m_weight=1.0,
        over_20m_weight=1.0,
        optimize_metric="hybrid_wmape",
    ),
    "baseline_lowmid_objective": LowValueStrategy(
        name="baseline_lowmid_objective",
        note="Keep baseline weights but optimize the tuner for low/mid bands.",
        under_5m_weight=1.0,
        mid_5m_to_20m_weight=1.0,
        over_20m_weight=1.0,
        optimize_metric="lowmid_wmape",
    ),
    "cheap_balanced": LowValueStrategy(
        name="cheap_balanced",
        note="Moderate cheap-player emphasis with reduced pressure on expensive rows.",
        under_5m_weight=1.75,
        mid_5m_to_20m_weight=1.35,
        over_20m_weight=0.85,
        optimize_metric="lowmid_wmape",
    ),
    "cheap_aggressive": LowValueStrategy(
        name="cheap_aggressive",
        note="Stronger cheap-player weighting to stress under-20m fit.",
        under_5m_weight=2.50,
        mid_5m_to_20m_weight=1.75,
        over_20m_weight=0.60,
        optimize_metric="lowmid_wmape",
    ),
}


def _train_market_value_main(**kwargs) -> None:
    from scouting_ml.models.train_market_value_full import main as train_market_value_main

    train_market_value_main(**kwargs)


def _parse_csv_tokens(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [tok.strip() for tok in str(raw).split(",") if tok.strip()]


def _write_markdown_report(
    *,
    path: Path,
    dataset_path: str,
    val_season: str,
    test_season: str,
    summary: pd.DataFrame,
    bundle: dict[str, Any],
    report_top_n: int,
) -> None:
    lines: list[str] = []
    lines.append("# Low-Value Strategy Benchmark")
    lines.append("")
    lines.append(f"- Dataset: `{dataset_path}`")
    lines.append(f"- Validation season: `{val_season}`")
    lines.append(f"- Test season: `{test_season}`")
    lines.append(f"- Generated: `{bundle.get('generated_at_utc')}`")
    lines.append("")

    lines.append("## Best Strategies")
    for title, item in [
        ("Best overall test strategy", bundle.get("best_overall_test")),
        ("Best under-20m test strategy", bundle.get("best_under_20m_test")),
        ("Best under-5m test strategy", bundle.get("best_under_5m_test")),
        ("Best 5m-20m test strategy", bundle.get("best_5m_to_20m_test")),
    ]:
        if not item:
            lines.append(f"- {title}: unavailable")
            continue
        lines.append(
            f"- {title}: `{item.get('config', 'n/a')}` | {item.get('metric', 'metric')}="
            f"{safe_float(item.get('value')):.4f} | test_r2={safe_float(item.get('test_r2')):.4f}"
        )
    lines.append("")

    lines.append("## Overall Ranking")
    if summary.empty:
        lines.append("No strategy rows produced.")
    else:
        lines.append(
            "| config | objective | under_5m_w | 5m_to_20m_w | over_20m_w | test_wmape | under_20m_wmape | under_5m_wmape | test_r2 |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for _, row in summary.iterrows():
            lines.append(
                f"| {row.get('config')} | {row.get('optimize_metric')} | "
                f"{safe_float(row.get('under_5m_weight')):.2f} | "
                f"{safe_float(row.get('mid_5m_to_20m_weight')):.2f} | "
                f"{safe_float(row.get('over_20m_weight')):.2f} | "
                f"{safe_float(row.get('test_wmape')):.4f} | "
                f"{safe_float(row.get('test_under_20m_wmape')):.4f} | "
                f"{safe_float(row.get('test_under_5m_wmape')):.4f} | "
                f"{safe_float(row.get('test_r2')):.4f} |"
            )
    lines.append("")

    lines.append("## Cheap-Player Tradeoffs")
    tradeoff_rows = bundle.get("tradeoff_rows_test") or []
    if tradeoff_rows:
        lines.append("| config | delta_test_wmape | delta_under_20m | delta_under_5m | test_r2 |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for row in tradeoff_rows[: max(int(report_top_n), 1)]:
            lines.append(
                f"| {row.get('config')} | "
                f"{safe_float(row.get('delta_test_wmape_vs_full')):.4f} | "
                f"{safe_float(row.get('delta_test_under_20m_wmape_vs_full')):.4f} | "
                f"{safe_float(row.get('delta_test_under_5m_wmape_vs_full')):.4f} | "
                f"{safe_float(row.get('test_r2')):.4f} |"
            )
    else:
        lines.append("No tradeoff rows available.")
    lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_low_value_strategy_benchmark(
    *,
    dataset_path: str,
    val_season: str,
    test_season: str,
    out_dir: str,
    strategy_names: Sequence[str],
    trials: int = 60,
    recency_half_life: float = 2.0,
    min_feature_coverage: float = 0.01,
    min_provider_feature_coverage: float = 0.05,
    slice_min_samples: int = 25,
    league_min_samples: int = 40,
    report_top_n: int = 8,
    mape_min_denom_eur: float = 1_000_000.0,
) -> dict[str, Any]:
    strategies: list[LowValueStrategy] = []
    for name in strategy_names:
        key = str(name).strip()
        if key not in DEFAULT_STRATEGIES:
            raise ValueError(f"Unknown low-value strategy '{key}'. Available: {sorted(DEFAULT_STRATEGIES.keys())}")
        strategies.append(DEFAULT_STRATEGIES[key])

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, Any]] = []
    slice_rows: list[dict[str, Any]] = []

    for strategy in strategies:
        print("\n====================================")
        print(f"[low-value] running: {strategy.name}")
        print(f"[low-value] note: {strategy.note}")
        print(
            "[low-value] weights: "
            f"under_5m={strategy.under_5m_weight:.2f}, "
            f"5m_to_20m={strategy.mid_5m_to_20m_weight:.2f}, "
            f"over_20m={strategy.over_20m_weight:.2f}"
        )
        print(f"[low-value] objective: {strategy.optimize_metric}")
        print("====================================")

        stem = f"{strategy.name}_{season_slug(test_season)}"
        pred_path = out_path / f"{stem}.csv"
        val_pred_path = out_path / f"{stem}_val.csv"
        metrics_path = out_path / f"{stem}.metrics.json"

        _train_market_value_main(
            dataset_path=dataset_path,
            val_season=val_season,
            test_season=test_season,
            output_path=str(pred_path),
            val_output_path=str(val_pred_path),
            metrics_output_path=str(metrics_path),
            n_optuna_trials=trials,
            recency_half_life=recency_half_life,
            under_5m_weight=strategy.under_5m_weight,
            mid_5m_to_20m_weight=strategy.mid_5m_to_20m_weight,
            over_20m_weight=strategy.over_20m_weight,
            optimize_metric=strategy.optimize_metric,
            min_feature_coverage=min_feature_coverage,
            min_provider_feature_coverage=min_provider_feature_coverage,
        )

        if not metrics_path.exists():
            raise RuntimeError(f"Low-value run '{strategy.name}' did not produce metrics file: {metrics_path}")
        if not pred_path.exists() or not val_pred_path.exists():
            raise RuntimeError(
                f"Low-value run '{strategy.name}' did not produce prediction outputs: "
                f"test={pred_path.exists()} val={val_pred_path.exists()}"
            )

        run_rows.append(
            {
                "config": strategy.name,
                "note": strategy.note,
                "metrics_path": str(metrics_path),
                "predictions_path": str(pred_path),
                "val_predictions_path": str(val_pred_path),
                "under_5m_weight": strategy.under_5m_weight,
                "mid_5m_to_20m_weight": strategy.mid_5m_to_20m_weight,
                "over_20m_weight": strategy.over_20m_weight,
                "optimize_metric": strategy.optimize_metric,
            }
        )
        for split, path in [("val", val_pred_path), ("test", pred_path)]:
            frame = load_prediction_frame(path)
            slice_rows.extend(
                build_slice_matrix(
                    frame,
                    config=strategy.name,
                    split=split,
                    note=strategy.note,
                    slice_min_samples=slice_min_samples,
                    league_min_samples=league_min_samples,
                    mape_min_denom_eur=mape_min_denom_eur,
                )
            )

    slice_df = pd.DataFrame(slice_rows)
    if slice_df.empty:
        raise RuntimeError("Low-value strategy benchmark produced no slice diagnostics.")
    slice_df = decorate_slice_matrix(slice_df)

    summary = build_overall_summary(run_rows, slice_df)
    if "baseline_full" in set(summary["config"].astype(str)):
        ref = summary.loc[summary["config"].astype(str) == "baseline_full"].iloc[0]
        metric_cols = [
            col
            for col in summary.columns
            if col.startswith(("val_", "test_"))
            and (
                col.endswith("_r2")
                or col.endswith("_mae_eur")
                or col.endswith("_mape")
                or col.endswith("_wmape")
            )
        ]
        for col in metric_cols:
            summary[f"delta_{col}_vs_full"] = pd.to_numeric(summary[col], errors="coerce") - safe_float(ref[col])

    summary_path = out_path / f"low_value_strategy_summary_{season_slug(test_season)}.csv"
    slices_path = out_path / f"low_value_strategy_slices_{season_slug(test_season)}.csv"
    bundle_path = out_path / f"low_value_strategy_bundle_{season_slug(test_season)}.json"
    report_path = out_path / f"low_value_strategy_report_{season_slug(test_season)}.md"

    summary.to_csv(summary_path, index=False)
    slice_df.to_csv(slices_path, index=False)

    tradeoff_rows = summary[summary["config"].astype(str) != "baseline_full"].copy()
    tradeoff_rows = tradeoff_rows.sort_values(
        ["test_under_20m_wmape", "test_under_5m_wmape", "test_wmape", "test_r2"],
        ascending=[True, True, True, False],
        na_position="last",
    )
    bundle = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": dataset_path,
        "val_season": val_season,
        "test_season": test_season,
        "strategies": list(strategy_names),
        "best_overall_test": pick_best_overall(summary, "test_wmape"),
        "best_under_20m_test": pick_best_overall(summary, "test_under_20m_wmape"),
        "best_under_5m_test": pick_best_overall(summary, "test_under_5m_wmape"),
        "best_5m_to_20m_test": pick_best_overall(summary, "test_5m_to_20m_wmape"),
        "tradeoff_rows_test": tradeoff_rows.head(max(int(report_top_n), 1)).to_dict(orient="records"),
        "artifacts": {
            "overall_summary_csv": str(summary_path),
            "slice_matrix_csv": str(slices_path),
            "bundle_json": str(bundle_path),
            "report_md": str(report_path),
        },
    }
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    _write_markdown_report(
        path=report_path,
        dataset_path=dataset_path,
        val_season=val_season,
        test_season=test_season,
        summary=summary,
        bundle=bundle,
        report_top_n=report_top_n,
    )

    print("\n========== LOW-VALUE STRATEGY SUMMARY ==========")
    for _, row in summary.iterrows():
        print(
            f"{row['config']:>24} | "
            f"test R2 {safe_float(row['test_r2'])*100:6.2f}% | "
            f"WMAPE {safe_float(row['test_wmape'])*100:6.2f}% | "
            f"under_20m WMAPE {safe_float(row.get('test_under_20m_wmape'))*100:6.2f}% | "
            f"under_5m WMAPE {safe_float(row.get('test_under_5m_wmape'))*100:6.2f}%"
        )
    print(f"[low-value] wrote overall summary -> {summary_path}")
    print(f"[low-value] wrote slice matrix -> {slices_path}")
    print(f"[low-value] wrote bundle -> {bundle_path}")
    print(f"[low-value] wrote report -> {report_path}")
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark low-value weighting/objective strategies for the market-value pipeline."
    )
    parser.add_argument("--dataset", default="data/model/big5_players_clean.parquet")
    parser.add_argument("--val-season", default="2023/24")
    parser.add_argument("--test-season", default="2024/25")
    parser.add_argument("--out-dir", default="data/model/reports/low_value_strategy")
    parser.add_argument(
        "--strategies",
        default="baseline_full,baseline_lowmid_objective,cheap_balanced,cheap_aggressive",
        help=f"Comma-separated strategy names. Available: {','.join(sorted(DEFAULT_STRATEGIES.keys()))}",
    )
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--recency-half-life", type=float, default=2.0)
    parser.add_argument("--min-feature-coverage", type=float, default=0.01)
    parser.add_argument("--min-provider-feature-coverage", type=float, default=0.05)
    parser.add_argument("--slice-min-samples", type=int, default=25)
    parser.add_argument("--league-min-samples", type=int, default=40)
    parser.add_argument("--report-top-n", type=int, default=8)
    parser.add_argument("--mape-min-denom-eur", type=float, default=1_000_000.0)
    args = parser.parse_args()

    run_low_value_strategy_benchmark(
        dataset_path=args.dataset,
        val_season=args.val_season,
        test_season=args.test_season,
        out_dir=args.out_dir,
        strategy_names=_parse_csv_tokens(args.strategies),
        trials=args.trials,
        recency_half_life=args.recency_half_life,
        min_feature_coverage=args.min_feature_coverage,
        min_provider_feature_coverage=args.min_provider_feature_coverage,
        slice_min_samples=args.slice_min_samples,
        league_min_samples=args.league_min_samples,
        report_top_n=args.report_top_n,
        mape_min_denom_eur=args.mape_min_denom_eur,
    )


if __name__ == "__main__":
    main()
