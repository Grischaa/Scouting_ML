from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class FeatureFamily:
    name: str
    prefixes: tuple[str, ...] = ()
    columns: tuple[str, ...] = ()
    note: str = ""


FAMILIES: tuple[FeatureFamily, ...] = (
    FeatureFamily(
        "injury",
        prefixes=("injury_",),
        columns=("availability_risk_score", "durability_score"),
        note="Transfermarkt injury-history features and direct injury-derived durability scores.",
    ),
    FeatureFamily(
        "availability",
        prefixes=("avail_",),
        columns=(
            "availability_selection_score",
            "availability_performance_hint",
            "availability_trust_score",
        ),
        note="Provider-derived selection, usage, and availability trust signals.",
    ),
    FeatureFamily(
        "fixture",
        prefixes=("fixture_",),
        columns=("fixture_team_form_score", "fixture_environment_score"),
        note="Provider-derived team-form and scoring-environment context.",
    ),
    FeatureFamily(
        "statsbomb",
        prefixes=("sb_",),
        columns=(),
        note="StatsBomb provider season features.",
    ),
    FeatureFamily(
        "odds",
        prefixes=("odds_",),
        columns=("odds_strength_score",),
        note="Odds-derived market context features.",
    ),
)


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _match_family_columns(columns: list[str], family: FeatureFamily) -> list[str]:
    matched: list[str] = []
    explicit = set(family.columns)
    for col in columns:
        if col in explicit or any(col.startswith(prefix) for prefix in family.prefixes):
            matched.append(col)
    return sorted(set(matched))


def _present_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype):
        cleaned = (
            series.astype(str)
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA, "<NA>": pd.NA})
        )
        return cleaned.notna()
    return series.notna()


def _column_rows(frame: pd.DataFrame, *, family: str, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for col in frame.columns:
        mask = _present_mask(frame[col])
        present_rows = int(mask.sum())
        total_rows = int(len(frame))
        rows.append(
            {
                "section": "feature_coverage",
                "family": family,
                "split": split,
                "column": col,
                "rows": total_rows,
                "present_rows": present_rows,
                "coverage_0_to_1": float(present_rows / total_rows) if total_rows else 0.0,
            }
        )
    return rows


def _family_summary(frame: pd.DataFrame, *, family: FeatureFamily, split: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cols = _match_family_columns(list(frame.columns), family)
    if not cols:
        return (
            {
                "section": "family_summary",
                "family": family.name,
                "split": split,
                "note": family.note,
                "feature_count": 0,
                "rows": int(len(frame)),
                "row_coverage_share": 0.0,
                "mean_feature_coverage": 0.0,
                "median_feature_coverage": 0.0,
                "high_coverage_features": 0,
            },
            [],
        )

    work = frame[cols].copy()
    present = pd.DataFrame({col: _present_mask(work[col]) for col in cols}, index=work.index)
    per_col = present.mean(axis=0).astype(float)
    row_cov = present.any(axis=1).mean() if len(present) else 0.0
    col_rows = _column_rows(work, family=family.name, split=split)
    summary = {
        "section": "family_summary",
        "family": family.name,
        "split": split,
        "note": family.note,
        "feature_count": int(len(cols)),
        "rows": int(len(frame)),
        "row_coverage_share": float(row_cov),
        "mean_feature_coverage": float(per_col.mean()) if len(per_col) else 0.0,
        "median_feature_coverage": float(per_col.median()) if len(per_col) else 0.0,
        "high_coverage_features": int((per_col >= 0.50).sum()),
        "columns": cols,
    }
    return summary, col_rows


def build_external_feature_coverage_audit(
    *,
    dataset_path: str,
    out_json: str | None = None,
    out_csv: str | None = None,
    out_md: str | None = None,
) -> dict[str, Any]:
    path = Path(dataset_path)
    frame = _read_table(path)
    if "season" in frame.columns:
        frame["season"] = frame["season"].astype(str)

    family_overall: dict[str, Any] = {}
    family_by_season: dict[str, list[dict[str, Any]]] = {}
    csv_rows: list[dict[str, Any]] = []

    for family in FAMILIES:
        overall_summary, overall_columns = _family_summary(frame, family=family, split="overall")
        family_overall[family.name] = overall_summary
        csv_rows.append(overall_summary)
        csv_rows.extend(overall_columns)

        season_rows: list[dict[str, Any]] = []
        if "season" in frame.columns:
            for season, group in frame.groupby("season", dropna=False):
                split_label = str(season)
                season_summary, season_columns = _family_summary(group, family=family, split=split_label)
                season_rows.append(season_summary)
                csv_rows.append(season_summary)
                csv_rows.extend(season_columns)
        family_by_season[family.name] = season_rows

    payload = {
        "dataset_path": str(path),
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "seasons": sorted(frame["season"].astype(str).dropna().unique().tolist()) if "season" in frame.columns else [],
        "leagues": int(frame["league"].astype(str).nunique()) if "league" in frame.columns else 0,
        "family_overall": family_overall,
        "family_by_season": family_by_season,
    }

    if out_json:
        out_path = Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if out_csv:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(csv_rows).to_csv(out_path, index=False)

    if out_md:
        lines = [
            "# External Feature Coverage Audit",
            "",
            f"- Dataset: `{path}`",
            f"- Rows: `{len(frame):,}`",
            f"- Columns: `{len(frame.columns):,}`",
            f"- Seasons: `{', '.join(payload['seasons'])}`" if payload["seasons"] else "- Seasons: `n/a`",
            "",
            "## Family Coverage",
            "",
            "| Family | Features | Row coverage | Mean feature coverage | Notes |",
            "|---|---:|---:|---:|---|",
        ]
        for family in FAMILIES:
            summary = family_overall[family.name]
            lines.append(
                "| "
                f"{family.name} | {summary['feature_count']} | {summary['row_coverage_share']:.2%} | "
                f"{summary['mean_feature_coverage']:.2%} | {family.note} |"
            )
        out_path = Path(out_md)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit external/provider feature coverage in the modeling dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-json", default="data/model/reports/external_feature_coverage_audit.json")
    parser.add_argument("--out-csv", default="data/model/reports/external_feature_coverage_audit.csv")
    parser.add_argument("--out-md", default="data/model/reports/external_feature_coverage_audit.md")
    args = parser.parse_args()

    payload = build_external_feature_coverage_audit(
        dataset_path=args.dataset,
        out_json=args.out_json,
        out_csv=args.out_csv,
        out_md=args.out_md,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
