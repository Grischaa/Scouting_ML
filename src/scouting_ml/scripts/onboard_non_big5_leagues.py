from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


BIG5_SLUGS = {
    "english_premier_league",
    "spanish_la_liga",
    "german_bundesliga",
    "italian_serie_a",
    "french_ligue_1",
}

def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")


def _season_sort_key(season: str) -> int:
    m = re.match(r"^(\d{4})[-/](\d{2,4})$", str(season).strip())
    if not m:
        return -1
    return int(m.group(1))


def _league_slug_aliases(league_slug: str) -> list[str]:
    slug = _slugify(league_slug)
    tokens = slug.split("_")
    aliases = [slug]
    if len(tokens) > 1:
        aliases.append("_".join(tokens[1:]))
    return [a for a in dict.fromkeys(aliases) if a]


def _load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    frame = pd.read_csv(path)
    required = {"league_slug", "season", "country"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Manifest missing required columns: {', '.join(missing)}")
    frame["league_slug"] = frame["league_slug"].astype(str).str.strip().str.lower()
    frame["season"] = frame["season"].astype(str).str.strip()
    frame["country"] = frame["country"].astype(str).str.strip().str.lower()
    return frame


def _find_holdout_metrics(glob_pattern: str) -> list[Path]:
    raw_matches = glob.glob(glob_pattern, recursive=True)
    if not raw_matches:
        raw_matches = glob.glob(str(Path.cwd() / glob_pattern), recursive=True)
    return sorted({Path(m).resolve() for m in raw_matches if Path(m).is_file()})


def _load_holdout_index(paths: list[Path]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        league = payload.get("league")
        if not league:
            continue
        slug = _slugify(str(league))
        overall = payload.get("overall", {}) if isinstance(payload, dict) else {}
        domain_shift = payload.get("domain_shift", {}) if isinstance(payload, dict) else {}
        out[slug] = {
            "league_label": str(league),
            "r2": overall.get("r2"),
            "wmape": overall.get("wmape"),
            "mape": overall.get("mape"),
            "n_samples": overall.get("n_samples"),
            "domain_shift_mean_abs_z": domain_shift.get("mean_abs_shift_z"),
            "metrics_json": str(path),
        }
    return out


def build_onboarding_report(
    *,
    manifest_path: str,
    holdout_metrics_glob: str,
    out_json: str,
    out_csv: str,
    min_seasons: int = 2,
    min_files: int = 2,
    max_domain_shift_z: float = 1.25,
    min_holdout_r2: float = 0.35,
) -> dict[str, Any]:
    manifest = _load_manifest(Path(manifest_path))
    manifest = manifest[~manifest["league_slug"].isin(BIG5_SLUGS)].copy()
    if manifest.empty:
        raise ValueError("No non-Big5 leagues found in manifest.")

    holdout_paths = _find_holdout_metrics(holdout_metrics_glob)
    holdout_index = _load_holdout_index(holdout_paths)

    grouped = manifest.groupby("league_slug", dropna=False)
    records: list[dict[str, Any]] = []
    status_counts = defaultdict(int)

    for league_slug, g in grouped:
        seasons = sorted({str(s) for s in g["season"].dropna().astype(str)}, key=_season_sort_key)
        countries = sorted({str(c) for c in g["country"].dropna().astype(str)})
        files_total = int(len(g))
        season_count = int(len(seasons))
        latest_season = seasons[-1] if seasons else None
        country = countries[0] if countries else "unknown"

        reasons: list[str] = []
        if files_total < int(min_files):
            reasons.append(f"files_total {files_total} < min_files {int(min_files)}")
        if season_count < int(min_seasons):
            reasons.append(f"season_count {season_count} < min_seasons {int(min_seasons)}")

        holdout = None
        matched_holdout_slug = None
        for alias in _league_slug_aliases(str(league_slug)):
            if alias in holdout_index:
                holdout = holdout_index[alias]
                matched_holdout_slug = alias
                break
        holdout_available = holdout is not None
        r2 = None
        wmape = None
        shift_z = None
        metrics_json = None
        if holdout is None:
            reasons.append("missing_holdout_metrics")
        else:
            metrics_json = holdout.get("metrics_json")
            r2 = holdout.get("r2")
            wmape = holdout.get("wmape")
            shift_z = holdout.get("domain_shift_mean_abs_z")

            try:
                r2_num = float(r2)
            except (TypeError, ValueError):
                r2_num = float("nan")
            try:
                shift_num = float(shift_z)
            except (TypeError, ValueError):
                shift_num = float("nan")

            if not (r2_num == r2_num) or r2_num < float(min_holdout_r2):
                reasons.append(f"holdout_r2 {r2} < min_holdout_r2 {float(min_holdout_r2):.2f}")
            if shift_num == shift_num and shift_num > float(max_domain_shift_z):
                reasons.append(
                    f"domain_shift_mean_abs_z {shift_num:.3f} > max_domain_shift_z {float(max_domain_shift_z):.3f}"
                )

        if reasons:
            if "missing_holdout_metrics" in reasons:
                status = "watch"
            else:
                status = "blocked"
        else:
            status = "ready"
        status_counts[status] += 1

        records.append(
            {
                "league_slug": str(league_slug),
                "country": country,
                "files_total": files_total,
                "season_count": season_count,
                "latest_season": latest_season,
                "holdout_available": holdout_available,
                "matched_holdout_slug": matched_holdout_slug,
                "holdout_r2": r2,
                "holdout_wmape": wmape,
                "domain_shift_mean_abs_z": shift_z,
                "status": status,
                "reasons": "; ".join(reasons),
                "metrics_json": metrics_json,
            }
        )

    out_frame = pd.DataFrame(records).sort_values(["status", "league_slug"])
    out_json_path = Path(out_json)
    out_csv_path = Path(out_csv)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_frame.to_csv(out_csv_path, index=False)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(Path(manifest_path)),
        "holdout_metrics_glob": holdout_metrics_glob,
        "thresholds": {
            "min_seasons": int(min_seasons),
            "min_files": int(min_files),
            "max_domain_shift_z": float(max_domain_shift_z),
            "min_holdout_r2": float(min_holdout_r2),
        },
        "status_counts": dict(status_counts),
        "csv_path": str(out_csv_path),
        "items": out_frame.to_dict(orient="records"),
    }
    out_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[onboard] wrote csv -> {out_csv_path}")
    print(f"[onboard] wrote json -> {out_json_path}")
    print(
        "[onboard] status counts -> "
        + ", ".join(f"{k}:{v}" for k, v in sorted(status_counts.items()))
    )
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate non-Big5 onboarding readiness from organized manifest and holdout "
            "domain-shift metrics."
        )
    )
    parser.add_argument("--manifest", default="data/processed/organization_manifest.csv")
    parser.add_argument(
        "--holdout-metrics-glob",
        default="data/model/**/*.holdout_*.metrics.json",
        help="Glob for league holdout metrics json files.",
    )
    parser.add_argument("--out-json", default="data/model/onboarding/non_big5_onboarding_report.json")
    parser.add_argument("--out-csv", default="data/model/onboarding/non_big5_onboarding_report.csv")
    parser.add_argument("--min-seasons", type=int, default=2)
    parser.add_argument("--min-files", type=int, default=2)
    parser.add_argument("--max-domain-shift-z", type=float, default=1.25)
    parser.add_argument("--min-holdout-r2", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_onboarding_report(
        manifest_path=args.manifest,
        holdout_metrics_glob=args.holdout_metrics_glob,
        out_json=args.out_json,
        out_csv=args.out_csv,
        min_seasons=args.min_seasons,
        min_files=args.min_files,
        max_domain_shift_z=args.max_domain_shift_z,
        min_holdout_r2=args.min_holdout_r2,
    )


if __name__ == "__main__":
    main()
