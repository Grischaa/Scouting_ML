from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scouting_ml.reporting.future_value_benchmarks import attach_future_targets, load_future_targets_frame


NUMERIC_CANDIDATES = [
    "undervaluation_score",
    "undervaluation_confidence",
    "value_gap_capped_eur",
    "value_gap_conservative_eur",
    "value_gap_eur",
    "market_value_eur",
    "expected_value_eur",
    "fair_value_eur",
    "age",
    "minutes",
    "sofa_minutesPlayed",
    "history_strength_score",
    "history_strength_coverage",
]

CATEGORICAL_CANDIDATES = [
    "model_position",
    "position_group",
    "league",
]

LABEL_MAP = {
    "positive_growth": "value_growth_positive_flag",
    "growth_gt25pct": "value_growth_gt25pct_flag",
}


def _resolve_base_rank_col(frame: pd.DataFrame) -> str:
    for col in ("scout_target_score", "undervaluation_score", "value_gap_capped_eur", "value_gap_conservative_eur", "value_gap_eur"):
        if col in frame.columns:
            return col
    raise ValueError("No suitable ranking column found for future scout score blending.")


def _minutes_series(frame: pd.DataFrame) -> pd.Series:
    if "minutes" in frame.columns:
        return pd.to_numeric(frame["minutes"], errors="coerce")
    if "sofa_minutesPlayed" in frame.columns:
        return pd.to_numeric(frame["sofa_minutesPlayed"], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _league_norm_series(frame: pd.DataFrame) -> pd.Series:
    if "league" not in frame.columns:
        return pd.Series("unknown", index=frame.index, dtype=object)
    return frame["league"].astype(str).str.strip().str.casefold()


def _parse_positions(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    out = {token.strip().upper() for token in str(raw).split(",") if token.strip()}
    return out or None


def _rank_percentile(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(0.5, index=series.index, dtype=float)
    ranks = values.rank(method="average", pct=True)
    return ranks.fillna(ranks.median() if ranks.notna().any() else 0.5).clip(lower=0.0, upper=1.0)


def _precision_at_k(frame: pd.DataFrame, *, score_col: str, label_col: str, k: int) -> float | None:
    work = frame.copy()
    work["_score"] = pd.to_numeric(work[score_col], errors="coerce")
    work["_label"] = pd.to_numeric(work[label_col], errors="coerce")
    work = work[work["_score"].notna() & work["_label"].notna()].copy()
    if work.empty:
        return None
    work = work.sort_values("_score", ascending=False).head(max(int(k), 1))
    if work.empty:
        return None
    return float((work["_label"] > 0).mean())


def _prepare_rows(
    frame: pd.DataFrame,
    *,
    min_minutes: float,
    max_age: float | None,
    positions: set[str] | None,
    include_leagues: set[str] | None,
    exclude_leagues: set[str] | None,
) -> pd.DataFrame:
    work = frame.copy()
    work["_minutes_used"] = _minutes_series(work).fillna(0.0)
    work["_age_num"] = pd.to_numeric(work.get("age"), errors="coerce")
    work["_league_norm"] = _league_norm_series(work)

    work = work[work["_minutes_used"] >= float(min_minutes)].copy()
    if max_age is not None:
        work = work[work["_age_num"].fillna(999.0) <= float(max_age)].copy()
    if positions:
        pos_series = (
            work["model_position"].astype(str).str.upper()
            if "model_position" in work.columns
            else work.get("position_group", pd.Series("", index=work.index)).astype(str).str.upper()
        )
        work = work[pos_series.isin(positions)].copy()
    if include_leagues:
        work = work[work["_league_norm"].isin(include_leagues)].copy()
    if exclude_leagues:
        work = work[~work["_league_norm"].isin(exclude_leagues)].copy()
    return work


def _feature_lists(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [col for col in NUMERIC_CANDIDATES if col in frame.columns]
    categorical = [col for col in CATEGORICAL_CANDIDATES if col in frame.columns]
    if not numeric and not categorical:
        raise ValueError("No supported feature columns available for future scout score.")
    return numeric, categorical


def _top_coefficients(model: Pipeline, *, limit: int = 20) -> list[dict[str, Any]]:
    try:
        pre = model.named_steps["preprocess"]
        clf = model.named_steps["model"]
        feature_names = pre.get_feature_names_out()
        coefs = clf.coef_[0]
    except Exception:
        return []
    rows = [{"feature": str(name), "coefficient": float(weight)} for name, weight in zip(feature_names, coefs)]
    rows.sort(key=lambda row: abs(row["coefficient"]), reverse=True)
    return rows[: max(int(limit), 1)]


def build_future_scout_score(
    *,
    val_predictions_path: str,
    test_predictions_path: str | None,
    out_val_path: str,
    out_test_path: str | None,
    diagnostics_out: str,
    future_targets_path: str | None = None,
    dataset_path: str | None = None,
    min_next_minutes: float = 450.0,
    min_minutes: float = 900.0,
    max_age: float | None = None,
    positions: set[str] | None = None,
    include_leagues: set[str] | None = None,
    exclude_leagues: set[str] | None = None,
    label_mode: str = "positive_growth",
    k_eval: int = 25,
) -> dict[str, Any]:
    if label_mode not in LABEL_MAP:
        raise ValueError(f"Unknown label_mode '{label_mode}'. Expected one of: {sorted(LABEL_MAP)}")
    label_col = LABEL_MAP[label_mode]

    targets, target_meta = load_future_targets_frame(
        future_targets_path=future_targets_path,
        dataset_path=dataset_path,
        min_next_minutes=min_next_minutes,
    )

    val_frame = pd.read_csv(val_predictions_path, low_memory=False)
    val_joined, val_join_meta = attach_future_targets(val_frame, targets)
    train_rows = _prepare_rows(
        val_joined,
        min_minutes=min_minutes,
        max_age=max_age,
        positions=positions,
        include_leagues=include_leagues,
        exclude_leagues=exclude_leagues,
    )
    train_rows = train_rows[pd.to_numeric(train_rows.get("has_next_season_target"), errors="coerce") == 1].copy()
    train_rows["_label"] = pd.to_numeric(train_rows.get(label_col), errors="coerce")
    train_rows = train_rows[train_rows["_label"].notna()].copy()
    if train_rows["_label"].nunique() < 2:
        raise ValueError("Future scout score needs both positive and negative labeled rows in the training split.")

    numeric_features, categorical_features = _feature_lists(train_rows)
    transformers = []
    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            )
        )

    model = Pipeline(
        [
            ("preprocess", ColumnTransformer(transformers=transformers, remainder="drop")),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    feature_cols = [*numeric_features, *categorical_features]
    model.fit(train_rows[feature_cols], train_rows["_label"].astype(int))

    base_rank_col = _resolve_base_rank_col(val_joined)

    def _apply(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        joined, join_meta = attach_future_targets(frame, targets)
        prob = model.predict_proba(joined[feature_cols])[:, 1]
        joined["future_growth_probability"] = prob
        joined["future_scout_score"] = prob
        base_rank_pct = _rank_percentile(joined[base_rank_col])
        joined["future_scout_blend_score"] = joined["future_growth_probability"] * (0.50 + base_rank_pct)
        return joined, join_meta

    val_scored, val_scored_join = _apply(val_frame)
    val_labeled = _prepare_rows(
        val_scored,
        min_minutes=min_minutes,
        max_age=max_age,
        positions=positions,
        include_leagues=include_leagues,
        exclude_leagues=exclude_leagues,
    )
    val_labeled = val_labeled[pd.to_numeric(val_labeled.get("has_next_season_target"), errors="coerce") == 1].copy()
    val_labeled["_label"] = pd.to_numeric(val_labeled.get(label_col), errors="coerce")
    val_labeled = val_labeled[val_labeled["_label"].notna()].copy()

    out_val = Path(out_val_path)
    out_val.parent.mkdir(parents=True, exist_ok=True)
    val_scored.to_csv(out_val, index=False)

    test_payload: dict[str, Any] | None = None
    if test_predictions_path and out_test_path:
        test_frame = pd.read_csv(test_predictions_path, low_memory=False)
        test_scored, test_scored_join = _apply(test_frame)
        out_test = Path(out_test_path)
        out_test.parent.mkdir(parents=True, exist_ok=True)
        test_scored.to_csv(out_test, index=False)
        test_payload = {
            "predictions_path": test_predictions_path,
            "output_path": str(out_test),
            "join": test_scored_join,
        }

    y_true = val_labeled["_label"].astype(int)
    y_pred = pd.to_numeric(val_labeled["future_growth_probability"], errors="coerce")
    diagnostics = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_source": target_meta,
        "label_mode": label_mode,
        "label_column": label_col,
        "training_filters": {
            "min_minutes": float(min_minutes),
            "max_age": None if max_age is None else float(max_age),
            "positions": sorted(positions) if positions else [],
            "include_leagues": sorted(include_leagues) if include_leagues else [],
            "exclude_leagues": sorted(exclude_leagues) if exclude_leagues else [],
        },
        "features": {
            "numeric": numeric_features,
            "categorical": categorical_features,
            "base_rank_column": base_rank_col,
        },
        "training_rows": int(len(train_rows)),
        "training_positive_rate": float(y_true.mean()),
        "val_predictions_path": val_predictions_path,
        "val_output_path": str(out_val),
        "val_join": val_join_meta,
        "val_scored_join": val_scored_join,
        "val_metrics": {
            "roc_auc": float(roc_auc_score(y_true, y_pred)) if y_true.nunique() >= 2 else None,
            "average_precision": float(average_precision_score(y_true, y_pred)),
            "precision_at_k_probability": _precision_at_k(val_labeled, score_col="future_growth_probability", label_col="_label", k=k_eval),
            "precision_at_k_blend": _precision_at_k(val_labeled, score_col="future_scout_blend_score", label_col="_label", k=k_eval),
            "precision_at_k_base_rank": _precision_at_k(val_labeled, score_col=base_rank_col, label_col="_label", k=k_eval),
            "k_eval": int(k_eval),
        },
        "top_coefficients": _top_coefficients(model),
        "test": test_payload,
    }

    diagnostics_path = Path(diagnostics_out)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    print(f"[future-score] wrote val scored predictions -> {out_val}")
    if test_payload:
        print(f"[future-score] wrote test scored predictions -> {test_payload['output_path']}")
    print(f"[future-score] wrote diagnostics -> {diagnostics_path}")
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a future-target-tuned scouting score on labeled validation rows and write enriched prediction files "
            "with future_growth_probability and future_scout_blend_score."
        )
    )
    parser.add_argument("--val-predictions", required=True)
    parser.add_argument("--test-predictions", default=None)
    parser.add_argument("--out-val", required=True)
    parser.add_argument("--out-test", default=None)
    parser.add_argument("--diagnostics-out", required=True)
    parser.add_argument("--future-targets", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--min-next-minutes", type=float, default=450.0)
    parser.add_argument("--min-minutes", type=float, default=900.0)
    parser.add_argument("--max-age", type=float, default=-1.0, help="Set negative to disable.")
    parser.add_argument("--positions", default="")
    parser.add_argument("--include-leagues", default="")
    parser.add_argument("--exclude-leagues", default="")
    parser.add_argument("--label-mode", default="positive_growth", choices=sorted(LABEL_MAP))
    parser.add_argument("--k-eval", type=int, default=25)
    args = parser.parse_args()

    build_future_scout_score(
        val_predictions_path=args.val_predictions,
        test_predictions_path=args.test_predictions,
        out_val_path=args.out_val,
        out_test_path=args.out_test,
        diagnostics_out=args.diagnostics_out,
        future_targets_path=args.future_targets,
        dataset_path=args.dataset,
        min_next_minutes=args.min_next_minutes,
        min_minutes=args.min_minutes,
        max_age=None if args.max_age < 0 else args.max_age,
        positions=_parse_positions(args.positions),
        include_leagues={token.strip().casefold() for token in str(args.include_leagues).split(",") if token.strip()} or None,
        exclude_leagues={token.strip().casefold() for token in str(args.exclude_leagues).split(",") if token.strip()} or None,
        label_mode=args.label_mode,
        k_eval=args.k_eval,
    )


if __name__ == "__main__":
    main()
