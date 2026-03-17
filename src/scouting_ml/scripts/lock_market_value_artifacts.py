from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

TEST_ENV = "SCOUTING_TEST_PREDICTIONS_PATH"
VAL_ENV = "SCOUTING_VAL_PREDICTIONS_PATH"
METRICS_ENV = "SCOUTING_METRICS_PATH"
MANIFEST_ENV = "SCOUTING_MODEL_MANIFEST_PATH"
STRICT_ENV = "SCOUTING_STRICT_ARTIFACTS"
VALUATION_TEST_ENV = "SCOUTING_VALUATION_TEST_PREDICTIONS_PATH"
VALUATION_VAL_ENV = "SCOUTING_VALUATION_VAL_PREDICTIONS_PATH"
VALUATION_METRICS_ENV = "SCOUTING_VALUATION_METRICS_PATH"
FUTURE_TEST_ENV = "SCOUTING_FUTURE_SHORTLIST_TEST_PREDICTIONS_PATH"
FUTURE_VAL_ENV = "SCOUTING_FUTURE_SHORTLIST_VAL_PREDICTIONS_PATH"
FUTURE_METRICS_ENV = "SCOUTING_FUTURE_SHORTLIST_METRICS_PATH"

DEFAULT_TEST = Path("data/model/big5_predictions_full_v2.csv")
DEFAULT_VAL = Path("data/model/big5_predictions_full_v2_val.csv")
DEFAULT_METRICS = Path("data/model/big5_predictions_full_v2.metrics.json")
DEFAULT_MANIFEST = Path("data/model/model_manifest.json")
DEFAULT_ENV_OUT = Path("data/model/model_artifacts.env")

ChampionRole = Literal["valuation", "future_shortlist"]
ROLE_TO_KEY: dict[ChampionRole, str] = {
    "valuation": "valuation_champion",
    "future_shortlist": "future_shortlist_champion",
}
ROLE_ENV_NAMES: dict[ChampionRole, dict[str, str]] = {
    "valuation": {
        "test_predictions": VALUATION_TEST_ENV,
        "val_predictions": VALUATION_VAL_ENV,
        "metrics": VALUATION_METRICS_ENV,
    },
    "future_shortlist": {
        "test_predictions": FUTURE_TEST_ENV,
        "val_predictions": FUTURE_VAL_ENV,
        "metrics": FUTURE_METRICS_ENV,
    },
}


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _meta(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "sha256": _sha256_file(path),
    }


def _resolve_path(cli_value: str | None, env_name: str, default: Path) -> Path:
    if cli_value and cli_value.strip():
        return Path(cli_value.strip())
    env_value = os.getenv(env_name, "").strip()
    if env_value:
        return Path(env_value)
    return default


def _load_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _config_from_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": metrics.get("dataset"),
        "val_season": metrics.get("val_season"),
        "test_season": metrics.get("test_season"),
        "trials_per_position": metrics.get("trials_per_position"),
        "optimize_metric": metrics.get("optimize_metric"),
        "interval_q": metrics.get("interval_q"),
        "two_stage_band_model": metrics.get("two_stage_band_model"),
        "band_min_samples": metrics.get("band_min_samples"),
        "band_blend_alpha": metrics.get("band_blend_alpha"),
        "strict_leakage_guard": metrics.get("strict_leakage_guard"),
    }


def _summary_from_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "overall": metrics.get("overall"),
        "segments": metrics.get("segments"),
        "league_holdout": metrics.get("league_holdout"),
    }


def _build_champion_section(
    *,
    role: ChampionRole,
    label: str,
    test_predictions: Path,
    val_predictions: Path,
    metrics_path: Path,
) -> dict[str, Any]:
    for required in (test_predictions, val_predictions, metrics_path):
        if not required.exists():
            raise FileNotFoundError(f"Required artifact not found: {required}")

    metrics = _load_metrics(metrics_path)
    return {
        "role": role,
        "label": label,
        "artifacts": {
            "test_predictions": _meta(test_predictions),
            "val_predictions": _meta(val_predictions),
            "metrics": _meta(metrics_path),
        },
        "config": _config_from_metrics(metrics),
        "summary": _summary_from_metrics(metrics),
    }


def _load_existing_manifest(manifest_out: Path) -> dict[str, Any] | None:
    if not manifest_out.exists():
        return None
    try:
        payload = json.loads(manifest_out.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _coerce_champion_section(payload: dict[str, Any] | None, role: ChampionRole) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    section = payload.get(ROLE_TO_KEY[role])
    return section if isinstance(section, dict) else None


def build_lock_bundle(
    *,
    test_predictions: Path,
    val_predictions: Path,
    metrics_path: Path,
    manifest_out: Path,
    env_out: Path,
    strict_artifacts: bool,
    label: str,
    primary_role: ChampionRole = "valuation",
    valuation_test_predictions: Path | None = None,
    valuation_val_predictions: Path | None = None,
    valuation_metrics_path: Path | None = None,
    valuation_label: str | None = None,
    future_test_predictions: Path | None = None,
    future_val_predictions: Path | None = None,
    future_metrics_path: Path | None = None,
    future_shortlist_label: str | None = None,
) -> None:
    primary_section = _build_champion_section(
        role=primary_role,
        label=label,
        test_predictions=test_predictions,
        val_predictions=val_predictions,
        metrics_path=metrics_path,
    )
    existing_manifest = _load_existing_manifest(manifest_out)
    has_explicit_dual_inputs = any(
        value is not None
        for value in (
            valuation_test_predictions,
            valuation_val_predictions,
            valuation_metrics_path,
            future_test_predictions,
            future_val_predictions,
            future_metrics_path,
        )
    )
    dual_role_enabled = has_explicit_dual_inputs or any(
        _coerce_champion_section(existing_manifest, role) is not None for role in ("valuation", "future_shortlist")
    )

    valuation_section: dict[str, Any] | None = None
    future_section: dict[str, Any] | None = None

    if dual_role_enabled:
        if primary_role == "valuation":
            valuation_section = primary_section
        else:
            future_section = primary_section

        if all(value is not None for value in (valuation_test_predictions, valuation_val_predictions, valuation_metrics_path)):
            valuation_section = _build_champion_section(
                role="valuation",
                label=valuation_label or "valuation_champion",
                test_predictions=Path(valuation_test_predictions),
                val_predictions=Path(valuation_val_predictions),
                metrics_path=Path(valuation_metrics_path),
            )
        elif valuation_section is None:
            valuation_section = _coerce_champion_section(existing_manifest, "valuation")

        if all(value is not None for value in (future_test_predictions, future_val_predictions, future_metrics_path)):
            future_section = _build_champion_section(
                role="future_shortlist",
                label=future_shortlist_label or "future_shortlist_champion",
                test_predictions=Path(future_test_predictions),
                val_predictions=Path(future_val_predictions),
                metrics_path=Path(future_metrics_path),
            )
        elif future_section is None:
            future_section = _coerce_champion_section(existing_manifest, "future_shortlist")

    payload = {
        "registry_version": 2 if dual_role_enabled else 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "legacy_default_role": primary_role,
        "artifacts": primary_section["artifacts"],
        "config": primary_section["config"],
        "summary": primary_section["summary"],
    }
    if valuation_section is not None:
        payload["valuation_champion"] = valuation_section
    if future_section is not None:
        payload["future_shortlist_champion"] = future_section

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    env_lines = [
        f"{TEST_ENV}={test_predictions}",
        f"{VAL_ENV}={val_predictions}",
        f"{METRICS_ENV}={metrics_path}",
        f"{MANIFEST_ENV}={manifest_out}",
        f"{STRICT_ENV}={'1' if strict_artifacts else '0'}",
    ]
    for role, section in (("valuation", valuation_section), ("future_shortlist", future_section)):
        if not isinstance(section, dict):
            continue
        artifacts = section.get("artifacts")
        if not isinstance(artifacts, dict):
            continue
        env_names = ROLE_ENV_NAMES[role]
        for key, env_name in env_names.items():
            artifact = artifacts.get(key)
            if isinstance(artifact, dict) and artifact.get("path"):
                env_lines.append(f"{env_name}={artifact['path']}")

    env_out.parent.mkdir(parents=True, exist_ok=True)
    env_out.write_text("\n".join(env_lines) + "\n", encoding="utf-8")

    print(f"[lock] wrote manifest -> {manifest_out}")
    print(f"[lock] wrote env file -> {env_out}")
    print("[lock] export this file before starting the API:")
    print(f"       source {env_out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze model artifacts into a manifest + env lock file for reproducible API serving."
    )
    parser.add_argument("--test-predictions", default=None)
    parser.add_argument("--val-predictions", default=None)
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--manifest-out", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--env-out", default=str(DEFAULT_ENV_OUT))
    parser.add_argument("--label", default="market_value_main")
    parser.add_argument("--primary-role", choices=["valuation", "future_shortlist"], default="valuation")
    parser.add_argument("--valuation-test-predictions", default=None)
    parser.add_argument("--valuation-val-predictions", default=None)
    parser.add_argument("--valuation-metrics", default=None)
    parser.add_argument("--valuation-label", default=None)
    parser.add_argument("--future-test-predictions", default=None)
    parser.add_argument("--future-val-predictions", default=None)
    parser.add_argument("--future-metrics", default=None)
    parser.add_argument("--future-shortlist-label", default=None)
    parser.add_argument(
        "--strict-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write SCOUTING_STRICT_ARTIFACTS=1 in lock env file.",
    )
    args = parser.parse_args()

    test_predictions = _resolve_path(args.test_predictions, TEST_ENV, DEFAULT_TEST)
    val_predictions = _resolve_path(args.val_predictions, VAL_ENV, DEFAULT_VAL)
    metrics_path = _resolve_path(args.metrics, METRICS_ENV, DEFAULT_METRICS)

    build_lock_bundle(
        test_predictions=test_predictions,
        val_predictions=val_predictions,
        metrics_path=metrics_path,
        manifest_out=Path(args.manifest_out),
        env_out=Path(args.env_out),
        strict_artifacts=bool(args.strict_artifacts),
        label=str(args.label),
        primary_role=str(args.primary_role),
        valuation_test_predictions=Path(args.valuation_test_predictions) if args.valuation_test_predictions else None,
        valuation_val_predictions=Path(args.valuation_val_predictions) if args.valuation_val_predictions else None,
        valuation_metrics_path=Path(args.valuation_metrics) if args.valuation_metrics else None,
        valuation_label=str(args.valuation_label).strip() if args.valuation_label else None,
        future_test_predictions=Path(args.future_test_predictions) if args.future_test_predictions else None,
        future_val_predictions=Path(args.future_val_predictions) if args.future_val_predictions else None,
        future_metrics_path=Path(args.future_metrics) if args.future_metrics else None,
        future_shortlist_label=str(args.future_shortlist_label).strip() if args.future_shortlist_label else None,
    )


if __name__ == "__main__":
    main()
