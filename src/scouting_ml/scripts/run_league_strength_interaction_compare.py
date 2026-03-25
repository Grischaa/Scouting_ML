from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


LEAGUE_STRENGTH_INTERACTION_COLUMNS = (
    "league_strength_blend",
    "club_league_strength_interaction",
    "international_league_strength_interaction",
    "elite_context_league_interaction",
)

DEFAULT_HOLDOUTS = (
    "Austrian Bundesliga",
    "Belgian Pro League",
    "Czech Fortuna Liga",
    "Danish Superliga",
    "Eredivisie",
    "Estonian Meistriliiga",
    "Greek Super League",
    "Primeira Liga",
    "Scottish Premiership",
    "Turkish Super Lig",
)

KEY_HOLDOUTS = (
    "Estonian Meistriliiga",
    "Czech Fortuna Liga",
)


@dataclass(frozen=True)
class VariantConfig:
    name: str
    exclude_columns: tuple[str, ...] = ()
    note: str = ""


VARIANTS: tuple[VariantConfig, ...] = (
    VariantConfig(
        name="baseline",
        exclude_columns=LEAGUE_STRENGTH_INTERACTION_COLUMNS,
        note="Baseline valuation model without explicit league-strength interaction features.",
    ),
    VariantConfig(
        name="league_strength_interactions",
        note="Challenger variant with league-strength interaction features enabled.",
    ),
)


def _train_market_value_main(**kwargs: Any) -> None:
    from scouting_ml.models.train_market_value_full import main as train_market_value_main

    train_market_value_main(**kwargs)


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")


def _parse_csv_tokens(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dataset_study_suffix(dataset: str) -> str:
    dataset_path = Path(dataset)
    try:
        stat = dataset_path.stat()
    except OSError:
        return "nosig"
    signature = (
        f"{dataset_path.resolve()}::{stat.st_size}::{stat.st_mtime_ns}"
    ).encode("utf-8")
    return hashlib.sha1(signature).hexdigest()[:10]


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def _overall_metric(metrics: dict[str, Any], split: str, key: str) -> float | None:
    block = metrics.get("overall", {}).get(split, {})
    if not isinstance(block, dict):
        return None
    return _safe_float(block.get(key))


def _holdout_map(metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in metrics.get("league_holdout") or []:
        if not isinstance(row, dict):
            continue
        league = str(row.get("league") or "").strip()
        if not league:
            continue
        out[_slugify(league)] = row
    return out


def _holdout_metric(metrics: dict[str, Any], league: str, key: str) -> float | None:
    row = _holdout_map(metrics).get(_slugify(league))
    if not row:
        return None
    overall = row.get("overall") if isinstance(row.get("overall"), dict) else {}
    return _safe_float(overall.get(key))


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
    league_holdouts: Sequence[str],
) -> dict[str, Any]:
    predictions_path = out_dir / f"{variant.name}.csv"
    val_predictions_path = out_dir / f"{variant.name}_val.csv"
    metrics_path = out_dir / f"{variant.name}.metrics.json"
    quality_path = out_dir / f"{variant.name}.quality.json"
    dataset_signature = _dataset_study_suffix(dataset)

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
        exclude_columns=list(variant.exclude_columns),
        save_shap_artifacts=False,
        league_holdouts=list(league_holdouts),
        optuna_study_namespace=f"league_strength_compare_{variant.name}_{dataset_signature}",
        optuna_load_if_exists=True,
    )

    metrics = _read_json(metrics_path)
    return {
        "name": variant.name,
        "note": variant.note,
        "exclude_columns": list(variant.exclude_columns),
        "artifacts": {
            "predictions_csv": str(predictions_path),
            "val_predictions_csv": str(val_predictions_path),
            "metrics_json": str(metrics_path),
            "quality_json": str(quality_path),
        },
        "metrics": metrics,
    }


def run_league_strength_interaction_compare(
    *,
    dataset: str,
    val_season: str,
    test_season: str,
    out_dir: str,
    compare_json: str | None = None,
    compare_md: str | None = None,
    trials: int = 20,
    holdout_trials: int = 5,
    optimize_metric: str = "lowmid_wmape",
    under_5m_weight: float = 1.0,
    mid_5m_to_20m_weight: float = 1.0,
    over_20m_weight: float = 1.0,
    league_holdouts: Sequence[str] | None = None,
) -> dict[str, Any]:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    requested_holdouts = list(league_holdouts or DEFAULT_HOLDOUTS)
    variant_results = [
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
            league_holdouts=requested_holdouts,
        )
        for variant in VARIANTS
    ]

    by_name = {item["name"]: item for item in variant_results}
    baseline_metrics = by_name["baseline"]["metrics"]
    challenger_metrics = by_name["league_strength_interactions"]["metrics"]

    holdout_rows: list[dict[str, Any]] = []
    for league in requested_holdouts:
        baseline_wmape = _holdout_metric(baseline_metrics, league, "wmape")
        challenger_wmape = _holdout_metric(challenger_metrics, league, "wmape")
        baseline_r2 = _holdout_metric(baseline_metrics, league, "r2")
        challenger_r2 = _holdout_metric(challenger_metrics, league, "r2")
        wmape_delta = (
            challenger_wmape - baseline_wmape
            if baseline_wmape is not None and challenger_wmape is not None
            else None
        )
        wmape_improvement_ratio = (
            (baseline_wmape - challenger_wmape) / baseline_wmape
            if baseline_wmape not in (None, 0.0) and challenger_wmape is not None
            else None
        )
        holdout_rows.append(
            {
                "league": league,
                "baseline_wmape": baseline_wmape,
                "challenger_wmape": challenger_wmape,
                "wmape_delta": wmape_delta,
                "wmape_improvement_ratio": wmape_improvement_ratio,
                "baseline_r2": baseline_r2,
                "challenger_r2": challenger_r2,
                "r2_delta": (
                    challenger_r2 - baseline_r2
                    if baseline_r2 is not None and challenger_r2 is not None
                    else None
                ),
            }
        )

    baseline_test_wmape = _overall_metric(baseline_metrics, "test", "wmape")
    challenger_test_wmape = _overall_metric(challenger_metrics, "test", "wmape")
    baseline_test_r2 = _overall_metric(baseline_metrics, "test", "r2")
    challenger_test_r2 = _overall_metric(challenger_metrics, "test", "r2")

    key_checks: dict[str, dict[str, Any]] = {}
    key_pass = True
    for league in KEY_HOLDOUTS:
        row = next((item for item in holdout_rows if item["league"] == league), None)
        improvement = row.get("wmape_improvement_ratio") if row else None
        passed = improvement is not None and improvement >= 0.25
        key_checks[league] = {
            "wmape_improvement_ratio": improvement,
            "passed": bool(passed),
        }
        key_pass = key_pass and bool(passed)

    overall_pass = (
        baseline_test_wmape is not None
        and challenger_test_wmape is not None
        and baseline_test_r2 is not None
        and challenger_test_r2 is not None
        and (challenger_test_wmape - baseline_test_wmape) <= 0.01
        and (challenger_test_r2 - baseline_test_r2) >= -0.02
    )

    decision = {
        "status": "promotable" if key_pass and overall_pass else "hold_runtime_shrinkage",
        "reason": (
            "Challenger cleared Estonia/Czech improvement checks and preserved aggregate test performance."
            if key_pass and overall_pass
            else "Keep the runtime league-adjusted shrinkage in production; the challenger did not clear promotion gates."
        ),
        "promotion_criteria": {
            "key_holdouts_improve_by_25pct_relative_wmape": key_checks,
            "overall_test_wmape_delta_max": 0.01,
            "overall_test_r2_delta_min": -0.02,
        },
        "overall_deltas": {
            "test_wmape_delta": (
                challenger_test_wmape - baseline_test_wmape
                if baseline_test_wmape is not None and challenger_test_wmape is not None
                else None
            ),
            "test_r2_delta": (
                challenger_test_r2 - baseline_test_r2
                if baseline_test_r2 is not None and challenger_test_r2 is not None
                else None
            ),
        },
    }

    payload = {
        "dataset": dataset,
        "val_season": val_season,
        "test_season": test_season,
        "trials": int(trials),
        "holdout_trials": int(holdout_trials),
        "optimize_metric": optimize_metric,
        "league_holdouts": requested_holdouts,
        "variants": [
            {
                "name": item["name"],
                "note": item["note"],
                "exclude_columns": item["exclude_columns"],
                "artifacts": item["artifacts"],
                "overall": {
                    "test_wmape": _overall_metric(item["metrics"], "test", "wmape"),
                    "test_r2": _overall_metric(item["metrics"], "test", "r2"),
                    "val_wmape": _overall_metric(item["metrics"], "val", "wmape"),
                    "val_r2": _overall_metric(item["metrics"], "val", "r2"),
                },
            }
            for item in variant_results
        ],
        "holdout_comparison": holdout_rows,
        "decision": decision,
    }

    if compare_json is None:
        compare_json = str(out_root / "league_strength_interaction_compare.json")
    compare_json_path = Path(compare_json)
    compare_json_path.parent.mkdir(parents=True, exist_ok=True)
    compare_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if compare_md is None:
        compare_md = str(out_root / "league_strength_interaction_compare.md")
    lines = [
        "# League Strength Interaction Compare",
        "",
        f"- Dataset: `{dataset}`",
        f"- Val/Test: `{val_season}` -> `{test_season}`",
        f"- Trials: `{trials}` (holdout trials `{holdout_trials}`)",
        f"- Optimize metric: `{optimize_metric}`",
        "",
        "## Variant Summary",
        "",
        "| Variant | Test WMAPE | Test R2 | Note |",
        "|---|---:|---:|---|",
    ]
    for item in payload["variants"]:
        overall = item["overall"]
        lines.append(
            f"| {item['name']} | "
            f"{_fmt_pct(overall['test_wmape'])} | "
            f"{_fmt_pct(overall['test_r2'])} | "
            f"{item['note']} |"
        )
    lines.extend(
        [
            "",
            "## Key Holdouts",
            "",
            "| League | Baseline WMAPE | Challenger WMAPE | Relative Improvement | Baseline R2 | Challenger R2 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in holdout_rows:
        lines.append(
            f"| {row['league']} | "
            f"{_fmt_pct(row['baseline_wmape'])} | "
            f"{_fmt_pct(row['challenger_wmape'])} | "
            f"{_fmt_pct(row['wmape_improvement_ratio'])} | "
            f"{_fmt_pct(row['baseline_r2'])} | "
            f"{_fmt_pct(row['challenger_r2'])} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Status: `{decision['status']}`",
            f"- Reason: {decision['reason']}",
        ]
    )
    Path(compare_md).write_text("\n".join(lines) + "\n", encoding="utf-8")

    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the active valuation model against a challenger with league-strength interaction features."
    )
    parser.add_argument("--dataset", default="data/model/champion_players_clean.parquet")
    parser.add_argument("--val-season", default="2023/24")
    parser.add_argument("--test-season", default="2024/25")
    parser.add_argument("--out-dir", default="data/model/reports/league_strength_interaction_compare")
    parser.add_argument("--compare-json", default=None)
    parser.add_argument("--compare-md", default=None)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--holdout-trials", type=int, default=5)
    parser.add_argument("--optimize-metric", default="lowmid_wmape")
    parser.add_argument("--under-5m-weight", type=float, default=1.0)
    parser.add_argument("--mid-5m-to-20m-weight", type=float, default=1.0)
    parser.add_argument("--over-20m-weight", type=float, default=1.0)
    parser.add_argument(
        "--league-holdouts",
        default=",".join(DEFAULT_HOLDOUTS),
        help="Comma-separated league holdouts for the comparison run.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_league_strength_interaction_compare(
        dataset=args.dataset,
        val_season=args.val_season,
        test_season=args.test_season,
        out_dir=args.out_dir,
        compare_json=args.compare_json,
        compare_md=args.compare_md,
        trials=args.trials,
        holdout_trials=args.holdout_trials,
        optimize_metric=args.optimize_metric,
        under_5m_weight=args.under_5m_weight,
        mid_5m_to_20m_weight=args.mid_5m_to_20m_weight,
        over_20m_weight=args.over_20m_weight,
        league_holdouts=_parse_csv_tokens(args.league_holdouts),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
