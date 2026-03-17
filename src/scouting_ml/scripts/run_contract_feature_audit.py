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
    slugify,
)
from scouting_ml.utils.data_utils import load_dataset


COMPACT_CONTRACT_COLUMNS = (
    "contract_years_left",
    "contract_expiring_within_1y",
    "contract_long_term_flag",
    "contract_security_score",
    "contract_agent_known_flag",
    "contract_loan_context_flag",
)

EXTENDED_CONTRACT_COLUMNS = COMPACT_CONTRACT_COLUMNS + (
    "contract_until_year",
    "contract_joined_year",
    "contract_release_clause_eur",
    "contract_loan_flag",
)


@dataclass(frozen=True)
class ContractAuditSpec:
    config: str
    note: str
    column: str | None = None


def _train_market_value_main(**kwargs) -> None:
    from scouting_ml.models.train_market_value_full import main as train_market_value_main

    train_market_value_main(**kwargs)


def _parse_csv_tokens(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [tok.strip() for tok in str(raw).split(",") if tok.strip()]


def _resolve_contract_columns(
    *,
    dataset_path: str,
    columns: Sequence[str] | None,
    audit_set: str,
) -> list[str]:
    requested = [str(col).strip() for col in (columns or []) if str(col).strip()]
    if requested:
        selected = requested
    elif str(audit_set).strip().casefold() == "extended":
        selected = list(EXTENDED_CONTRACT_COLUMNS)
    else:
        selected = list(COMPACT_CONTRACT_COLUMNS)

    dataset = load_dataset(dataset_path)
    available = set(dataset.columns.astype(str))
    resolved = [col for col in selected if col in available]
    missing = [col for col in selected if col not in available]
    if missing:
        print(
            "[contract-audit] skipped missing columns: "
            + ", ".join(missing[:12])
            + (" ..." if len(missing) > 12 else "")
        )
    if not resolved:
        raise ValueError("No requested contract columns exist in the dataset.")
    return resolved


def _specs_for_columns(columns: Sequence[str]) -> list[ContractAuditSpec]:
    specs = [ContractAuditSpec(config="full", note="Baseline with all current features.")]
    for col in columns:
        specs.append(
            ContractAuditSpec(
                config=f"drop_{slugify(col)}",
                column=col,
                note=f"Drop exact column `{col}` while keeping the rest of the contract block.",
            )
        )
    return specs


def _contract_signal_rows(summary: pd.DataFrame, *, direction: str, limit: int) -> list[dict[str, Any]]:
    work = summary[summary["config"].astype(str) != "full"].copy()
    if work.empty:
        return []
    sort_ascending = str(direction) == "suspect"
    work = work[np.isfinite(pd.to_numeric(work["delta_test_wmape_vs_full"], errors="coerce"))].copy()
    if work.empty:
        return []
    work = work.sort_values(
        ["delta_test_wmape_vs_full", "delta_test_under_20m_wmape_vs_full", "test_r2"],
        ascending=[sort_ascending, sort_ascending, not sort_ascending],
        na_position="last",
    )
    cols = [
        "column",
        "config",
        "test_wmape",
        "delta_test_wmape_vs_full",
        "test_under_20m_wmape",
        "delta_test_under_20m_wmape_vs_full",
        "test_under_5m_wmape",
        "delta_test_under_5m_wmape_vs_full",
        "test_r2",
        "note",
    ]
    return work.loc[:, [col for col in cols if col in work.columns]].head(max(int(limit), 1)).to_dict(orient="records")


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
    lines.append("# Contract Feature Audit")
    lines.append("")
    lines.append(f"- Dataset: `{dataset_path}`")
    lines.append(f"- Validation season: `{val_season}`")
    lines.append(f"- Test season: `{test_season}`")
    lines.append(f"- Generated: `{bundle.get('generated_at_utc')}`")
    lines.append("")

    baseline = bundle.get("baseline_test") or {}
    if baseline:
        lines.append("## Baseline")
        lines.append(
            f"- `full`: test_wmape={safe_float(baseline.get('test_wmape')):.4f} | "
            f"test_under_20m_wmape={safe_float(baseline.get('test_under_20m_wmape')):.4f} | "
            f"test_under_5m_wmape={safe_float(baseline.get('test_under_5m_wmape')):.4f} | "
            f"test_r2={safe_float(baseline.get('test_r2')):.4f}"
        )
        lines.append("")

    def _best_line(title: str, item: dict[str, Any] | None) -> None:
        if not item:
            lines.append(f"- {title}: unavailable")
            return
        lines.append(
            f"- {title}: `{item.get('column', item.get('config', 'n/a'))}` | "
            f"{item.get('metric', 'metric')}={safe_float(item.get('value')):.4f} | "
            f"test_r2={safe_float(item.get('test_r2')):.4f}"
        )

    lines.append("## Top Drops")
    _best_line("Best overall drop", bundle.get("best_overall_drop_test"))
    _best_line("Best under-20m drop", bundle.get("best_under_20m_drop_test"))
    _best_line("Best under-5m drop", bundle.get("best_under_5m_drop_test"))
    lines.append("")

    lines.append("## Overall Ranking")
    if summary.empty:
        lines.append("No audit rows produced.")
    else:
        cols = [
            "column",
            "config",
            "test_wmape",
            "delta_test_wmape_vs_full",
            "test_under_20m_wmape",
            "delta_test_under_20m_wmape_vs_full",
            "test_under_5m_wmape",
            "delta_test_under_5m_wmape_vs_full",
            "test_r2",
        ]
        work = summary.loc[:, [col for col in cols if col in summary.columns]].copy()
        lines.append(
            "| column | config | test_wmape | delta_vs_full | under_20m_wmape | delta_under_20m | under_5m_wmape | delta_under_5m | test_r2 |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for _, row in work.iterrows():
            lines.append(
                f"| {row.get('column') or '-'} | {row.get('config')} | "
                f"{safe_float(row.get('test_wmape')):.4f} | {safe_float(row.get('delta_test_wmape_vs_full')):.4f} | "
                f"{safe_float(row.get('test_under_20m_wmape')):.4f} | {safe_float(row.get('delta_test_under_20m_wmape_vs_full')):.4f} | "
                f"{safe_float(row.get('test_under_5m_wmape')):.4f} | {safe_float(row.get('delta_test_under_5m_wmape_vs_full')):.4f} | "
                f"{safe_float(row.get('test_r2')):.4f} |"
            )
    lines.append("")

    for title, key in [
        ("Most Suspect Contract Features", "most_suspect_contract_features_test"),
        ("Most Helpful Contract Features", "most_helpful_contract_features_test"),
    ]:
        lines.append(f"## {title}")
        rows = bundle.get(key) or []
        if not rows:
            lines.append("No rows available.")
            lines.append("")
            continue
        lines.append("| column | delta_test_wmape | delta_under_20m | delta_under_5m | test_r2 |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for row in rows[: max(int(report_top_n), 1)]:
            lines.append(
                f"| {row.get('column') or row.get('config')} | "
                f"{safe_float(row.get('delta_test_wmape_vs_full')):.4f} | "
                f"{safe_float(row.get('delta_test_under_20m_wmape_vs_full')):.4f} | "
                f"{safe_float(row.get('delta_test_under_5m_wmape_vs_full')):.4f} | "
                f"{safe_float(row.get('test_r2')):.4f} |"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_contract_feature_audit(
    *,
    dataset_path: str,
    val_season: str,
    test_season: str,
    out_dir: str,
    columns: Sequence[str] | None = None,
    audit_set: str = "compact",
    trials: int = 60,
    recency_half_life: float = 2.0,
    under_5m_weight: float = 1.0,
    mid_5m_to_20m_weight: float = 1.0,
    over_20m_weight: float = 1.0,
    optimize_metric: str = "hybrid_wmape",
    min_feature_coverage: float = 0.01,
    min_provider_feature_coverage: float = 0.05,
    slice_min_samples: int = 25,
    league_min_samples: int = 40,
    report_top_n: int = 8,
    mape_min_denom_eur: float = 1_000_000.0,
) -> dict[str, Any]:
    contract_columns = _resolve_contract_columns(
        dataset_path=dataset_path,
        columns=columns,
        audit_set=audit_set,
    )
    specs = _specs_for_columns(contract_columns)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, Any]] = []
    slice_rows: list[dict[str, Any]] = []

    for spec in specs:
        print("\n====================================")
        print(f"[contract-audit] running: {spec.config}")
        print(f"[contract-audit] note: {spec.note}")
        if spec.column:
            print(f"[contract-audit] dropped column: {spec.column}")
        print("====================================")

        stem = f"{spec.config}_{season_slug(test_season)}"
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
            under_5m_weight=under_5m_weight,
            mid_5m_to_20m_weight=mid_5m_to_20m_weight,
            over_20m_weight=over_20m_weight,
            exclude_columns=[spec.column] if spec.column else [],
            optimize_metric=optimize_metric,
            min_feature_coverage=min_feature_coverage,
            min_provider_feature_coverage=min_provider_feature_coverage,
        )

        if not metrics_path.exists():
            raise RuntimeError(f"Contract-audit run '{spec.config}' did not produce metrics file: {metrics_path}")
        if not pred_path.exists() or not val_pred_path.exists():
            raise RuntimeError(
                f"Contract-audit run '{spec.config}' did not produce prediction outputs: "
                f"test={pred_path.exists()} val={val_pred_path.exists()}"
            )

        run_rows.append(
            {
                "config": spec.config,
                "column": spec.column or "",
                "exclude_columns": spec.column or "",
                "note": spec.note,
                "metrics_path": str(metrics_path),
                "predictions_path": str(pred_path),
                "val_predictions_path": str(val_pred_path),
            }
        )
        for split, path in [("val", val_pred_path), ("test", pred_path)]:
            frame = load_prediction_frame(path)
            slice_rows.extend(
                build_slice_matrix(
                    frame,
                    config=spec.config,
                    split=split,
                    note=spec.note,
                    slice_min_samples=slice_min_samples,
                    league_min_samples=league_min_samples,
                    mape_min_denom_eur=mape_min_denom_eur,
                )
            )

    slice_df = pd.DataFrame(slice_rows)
    if slice_df.empty:
        raise RuntimeError("Contract feature audit produced no slice diagnostics.")
    slice_df = decorate_slice_matrix(slice_df)

    summary = build_overall_summary(run_rows, slice_df)
    summary["audit_set"] = str(audit_set)

    summary_path = out_path / f"contract_feature_audit_summary_{season_slug(test_season)}.csv"
    slices_path = out_path / f"contract_feature_audit_slices_{season_slug(test_season)}.csv"
    bundle_path = out_path / f"contract_feature_audit_bundle_{season_slug(test_season)}.json"
    report_path = out_path / f"contract_feature_audit_report_{season_slug(test_season)}.md"

    summary.to_csv(summary_path, index=False)
    slice_df.to_csv(slices_path, index=False)

    baseline = summary.loc[summary["config"].astype(str) == "full"]
    baseline_row = baseline.iloc[0].to_dict() if not baseline.empty else {}
    bundle = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": dataset_path,
        "val_season": val_season,
        "test_season": test_season,
        "audit_set": audit_set,
        "columns": contract_columns,
        "baseline_test": {
            "test_wmape": safe_float(baseline_row.get("test_wmape")),
            "test_under_20m_wmape": safe_float(baseline_row.get("test_under_20m_wmape")),
            "test_under_5m_wmape": safe_float(baseline_row.get("test_under_5m_wmape")),
            "test_r2": safe_float(baseline_row.get("test_r2")),
        } if baseline_row else {},
        "best_overall_drop_test": pick_best_overall(summary, "test_wmape", exclude_configs=("full",)),
        "best_under_20m_drop_test": pick_best_overall(summary, "test_under_20m_wmape", exclude_configs=("full",)),
        "best_under_5m_drop_test": pick_best_overall(summary, "test_under_5m_wmape", exclude_configs=("full",)),
        "most_suspect_contract_features_test": _contract_signal_rows(summary, direction="suspect", limit=report_top_n),
        "most_helpful_contract_features_test": _contract_signal_rows(summary, direction="helpful", limit=report_top_n),
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

    print("\n========== CONTRACT FEATURE AUDIT ==========")
    for _, row in summary.iterrows():
        label = row["column"] or "BASELINE"
        print(
            f"{label:>28} | "
            f"test R2 {safe_float(row['test_r2'])*100:6.2f}% | "
            f"WMAPE {safe_float(row['test_wmape'])*100:6.2f}% | "
            f"under_20m WMAPE {safe_float(row.get('test_under_20m_wmape'))*100:6.2f}% | "
            f"delta vs full {safe_float(row.get('delta_test_wmape_vs_full'))*100:6.2f}%"
        )
    print(f"[contract-audit] wrote overall summary -> {summary_path}")
    print(f"[contract-audit] wrote slice matrix -> {slices_path}")
    print(f"[contract-audit] wrote bundle -> {bundle_path}")
    print(f"[contract-audit] wrote report -> {report_path}")
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run exact-column contract feature audits for the market-value pipeline."
    )
    parser.add_argument("--dataset", default="data/model/big5_players_clean.parquet")
    parser.add_argument("--val-season", default="2023/24")
    parser.add_argument("--test-season", default="2024/25")
    parser.add_argument("--out-dir", default="data/model/reports/contract_audit")
    parser.add_argument(
        "--columns",
        default="",
        help="Optional comma-separated exact contract columns to audit. Overrides --audit-set.",
    )
    parser.add_argument(
        "--audit-set",
        default="compact",
        choices=["compact", "extended"],
        help="Compact audits only derived contract signals; extended adds raw numeric fields.",
    )
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--recency-half-life", type=float, default=2.0)
    parser.add_argument("--under-5m-weight", type=float, default=1.0)
    parser.add_argument("--mid-5m-20m-weight", type=float, default=1.0)
    parser.add_argument("--over-20m-weight", type=float, default=1.0)
    parser.add_argument(
        "--optimize-metric",
        default="hybrid_wmape",
        choices=["mae", "rmse", "overall_wmape", "band_wmape", "lowmid_wmape", "hybrid_wmape"],
    )
    parser.add_argument("--min-feature-coverage", type=float, default=0.01)
    parser.add_argument("--min-provider-feature-coverage", type=float, default=0.05)
    parser.add_argument("--slice-min-samples", type=int, default=25)
    parser.add_argument("--league-min-samples", type=int, default=40)
    parser.add_argument("--report-top-n", type=int, default=8)
    parser.add_argument("--mape-min-denom-eur", type=float, default=1_000_000.0)
    args = parser.parse_args()

    run_contract_feature_audit(
        dataset_path=args.dataset,
        val_season=args.val_season,
        test_season=args.test_season,
        out_dir=args.out_dir,
        columns=_parse_csv_tokens(args.columns),
        audit_set=args.audit_set,
        trials=args.trials,
        recency_half_life=args.recency_half_life,
        under_5m_weight=args.under_5m_weight,
        mid_5m_to_20m_weight=args.mid_5m_20m_weight,
        over_20m_weight=args.over_20m_weight,
        optimize_metric=args.optimize_metric,
        min_feature_coverage=args.min_feature_coverage,
        min_provider_feature_coverage=args.min_provider_feature_coverage,
        slice_min_samples=args.slice_min_samples,
        league_min_samples=args.league_min_samples,
        report_top_n=args.report_top_n,
        mape_min_denom_eur=args.mape_min_denom_eur,
    )


if __name__ == "__main__":
    main()
