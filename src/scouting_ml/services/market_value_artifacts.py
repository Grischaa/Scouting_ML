"""Artifact path, manifest, and readiness helpers for market-value services."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


Split = Literal["test", "val"]
ChampionRole = Literal["valuation", "future_shortlist"]

TEST_PRED_ENV = "SCOUTING_TEST_PREDICTIONS_PATH"
VAL_PRED_ENV = "SCOUTING_VAL_PREDICTIONS_PATH"
METRICS_ENV = "SCOUTING_METRICS_PATH"
MODEL_MANIFEST_ENV = "SCOUTING_MODEL_MANIFEST_PATH"
BENCHMARK_REPORT_ENV = "SCOUTING_BENCHMARK_REPORT_PATH"
ENABLE_RESIDUAL_CALIBRATION_ENV = "SCOUTING_ENABLE_RESIDUAL_CALIBRATION"
CALIBRATION_MIN_SAMPLES_ENV = "SCOUTING_CALIBRATION_MIN_SAMPLES"
WATCHLIST_PATH_ENV = "SCOUTING_WATCHLIST_PATH"
VALUATION_TEST_PRED_ENV = "SCOUTING_VALUATION_TEST_PREDICTIONS_PATH"
VALUATION_VAL_PRED_ENV = "SCOUTING_VALUATION_VAL_PREDICTIONS_PATH"
VALUATION_METRICS_ENV = "SCOUTING_VALUATION_METRICS_PATH"
FUTURE_TEST_PRED_ENV = "SCOUTING_FUTURE_SHORTLIST_TEST_PREDICTIONS_PATH"
FUTURE_VAL_PRED_ENV = "SCOUTING_FUTURE_SHORTLIST_VAL_PREDICTIONS_PATH"
FUTURE_METRICS_ENV = "SCOUTING_FUTURE_SHORTLIST_METRICS_PATH"

DEFAULT_TEST_PRED = Path("data/model/big5_predictions_full_v2.csv")
DEFAULT_VAL_PRED = Path("data/model/big5_predictions_full_v2_val.csv")
DEFAULT_METRICS = Path("data/model/big5_predictions_full_v2.metrics.json")
DEFAULT_MODEL_MANIFEST = Path("data/model/model_manifest.json")
DEFAULT_BENCHMARK_REPORT = Path("data/model/reports/market_value_benchmark_report.json")
DEFAULT_WATCHLIST_PATH = Path("data/model/scout_watchlist.jsonl")

SPLIT_TO_PATH = {
    "test": (TEST_PRED_ENV, DEFAULT_TEST_PRED),
    "val": (VAL_PRED_ENV, DEFAULT_VAL_PRED),
}
ROLE_TO_MANIFEST_KEY: dict[ChampionRole, str] = {
    "valuation": "valuation_champion",
    "future_shortlist": "future_shortlist_champion",
}
ROLE_ENV_NAMES: dict[ChampionRole, dict[str, str]] = {
    "valuation": {
        "test": VALUATION_TEST_PRED_ENV,
        "val": VALUATION_VAL_PRED_ENV,
        "metrics": VALUATION_METRICS_ENV,
    },
    "future_shortlist": {
        "test": FUTURE_TEST_PRED_ENV,
        "val": FUTURE_VAL_PRED_ENV,
        "metrics": FUTURE_METRICS_ENV,
    },
}
REQUIRED_ARTIFACT_ENVS = (TEST_PRED_ENV, VAL_PRED_ENV, METRICS_ENV)


def resolve_path(env_var: str, default_path: Path) -> Path:
    value = os.getenv(env_var, "").strip()
    return Path(value) if value else default_path


def manifest_path() -> Path:
    manifest_env_raw = os.getenv(MODEL_MANIFEST_ENV, "").strip()
    return Path(manifest_env_raw) if manifest_env_raw else DEFAULT_MODEL_MANIFEST


def load_manifest_payload() -> dict[str, Any] | None:
    path = manifest_path()
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def manifest_role_section(payload: dict[str, Any] | None, role: ChampionRole) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    key = ROLE_TO_MANIFEST_KEY[role]
    section = payload.get(key)
    return section if isinstance(section, dict) else None


def legacy_default_role(payload: dict[str, Any] | None) -> ChampionRole:
    if isinstance(payload, dict) and str(payload.get("legacy_default_role") or "").strip() == "future_shortlist":
        return "future_shortlist"
    return "valuation"


def normalized_path_str(path: Path | str) -> str:
    return os.path.normcase(str(Path(path)))


def manifest_targets_active_artifacts(
    payload: dict[str, Any],
    *,
    test_path: Path,
    val_path: Path,
    metrics_path: Path,
) -> bool:
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        return False

    expectations = {
        "test_predictions": test_path,
        "val_predictions": val_path,
        "metrics": metrics_path,
    }
    for key, expected in expectations.items():
        section = artifacts.get(key)
        if not isinstance(section, dict):
            return False
        raw_path = section.get("path")
        if not raw_path:
            return False
        if normalized_path_str(raw_path) != normalized_path_str(expected):
            return False
    return True


def resolve_role_artifact_paths(role: ChampionRole) -> dict[str, Path]:
    manifest_payload = load_manifest_payload()
    manifest_env_raw = os.getenv(MODEL_MANIFEST_ENV, "").strip()
    role_envs = ROLE_ENV_NAMES[role]
    test_env = os.getenv(role_envs["test"], "").strip()
    val_env = os.getenv(role_envs["val"], "").strip()
    metrics_env = os.getenv(role_envs["metrics"], "").strip()
    legacy_paths = {
        "test_predictions": resolve_path(*SPLIT_TO_PATH["test"]),
        "val_predictions": resolve_path(*SPLIT_TO_PATH["val"]),
        "metrics": resolve_path(METRICS_ENV, DEFAULT_METRICS),
    }
    if test_env or val_env or metrics_env:
        return {
            "test_predictions": Path(test_env) if test_env else legacy_paths["test_predictions"],
            "val_predictions": Path(val_env) if val_env else legacy_paths["val_predictions"],
            "metrics": Path(metrics_env) if metrics_env else legacy_paths["metrics"],
        }

    legacy_env_explicit = any(
        os.getenv(env_name, "").strip() for env_name in (TEST_PRED_ENV, VAL_PRED_ENV, METRICS_ENV)
    )
    manifest_matches_legacy = bool(
        manifest_payload
        and manifest_targets_active_artifacts(
            manifest_payload,
            test_path=legacy_paths["test_predictions"],
            val_path=legacy_paths["val_predictions"],
            metrics_path=legacy_paths["metrics"],
        )
    )
    can_use_manifest = bool(manifest_payload) and (
        bool(manifest_env_raw) or not legacy_env_explicit or manifest_matches_legacy
    )
    if can_use_manifest:
        section = manifest_role_section(manifest_payload, role)
        if isinstance(section, dict):
            artifacts = section.get("artifacts")
            if isinstance(artifacts, dict):
                test_meta = artifacts.get("test_predictions")
                val_meta = artifacts.get("val_predictions")
                metrics_meta = artifacts.get("metrics")
                if all(isinstance(meta, dict) and meta.get("path") for meta in (test_meta, val_meta, metrics_meta)):
                    return {
                        "test_predictions": Path(str(test_meta["path"])),
                        "val_predictions": Path(str(val_meta["path"])),
                        "metrics": Path(str(metrics_meta["path"])),
                    }

        if manifest_payload is not None and legacy_default_role(manifest_payload) == role:
            artifacts = manifest_payload.get("artifacts")
            if isinstance(artifacts, dict):
                test_meta = artifacts.get("test_predictions")
                val_meta = artifacts.get("val_predictions")
                metrics_meta = artifacts.get("metrics")
                if all(isinstance(meta, dict) and meta.get("path") for meta in (test_meta, val_meta, metrics_meta)):
                    return {
                        "test_predictions": Path(str(test_meta["path"])),
                        "val_predictions": Path(str(val_meta["path"])),
                        "metrics": Path(str(metrics_meta["path"])),
                    }

    return legacy_paths


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def watchlist_path() -> Path:
    return resolve_path(WATCHLIST_PATH_ENV, DEFAULT_WATCHLIST_PATH)


def file_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "size_bytes": None,
            "mtime_utc": None,
            "mtime_epoch": None,
        }
    stat = path.stat()
    dt = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    return {
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime_utc": dt,
        "mtime_epoch": float(stat.st_mtime),
    }


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def get_resolved_artifact_paths() -> dict[str, str]:
    test_path = resolve_path(*SPLIT_TO_PATH["test"])
    val_path = resolve_path(*SPLIT_TO_PATH["val"])
    metrics_path = resolve_path(METRICS_ENV, DEFAULT_METRICS)
    return {
        "test_predictions_path": str(test_path),
        "val_predictions_path": str(val_path),
        "metrics_path": str(metrics_path),
    }


def get_active_artifacts() -> dict[str, Any]:
    paths = get_resolved_artifact_paths()
    test_path = Path(paths["test_predictions_path"])
    val_path = Path(paths["val_predictions_path"])
    metrics_path = Path(paths["metrics_path"])
    valuation_paths = resolve_role_artifact_paths("valuation")
    future_paths = resolve_role_artifact_paths("future_shortlist")
    return {
        "test_predictions_path": str(test_path),
        "val_predictions_path": str(val_path),
        "metrics_path": str(metrics_path),
        "test_predictions_sha256": sha256_file(test_path),
        "val_predictions_sha256": sha256_file(val_path),
        "metrics_sha256": sha256_file(metrics_path),
        "prediction_service_base_role": "valuation",
        "shortlist_overlay_role": "future_shortlist",
        "valuation": {
            "test_predictions_path": str(valuation_paths["test_predictions"]),
            "val_predictions_path": str(valuation_paths["val_predictions"]),
            "metrics_path": str(valuation_paths["metrics"]),
            "test_predictions_sha256": sha256_file(valuation_paths["test_predictions"]),
            "val_predictions_sha256": sha256_file(valuation_paths["val_predictions"]),
            "metrics_sha256": sha256_file(valuation_paths["metrics"]),
        },
        "future_shortlist": {
            "test_predictions_path": str(future_paths["test_predictions"]),
            "val_predictions_path": str(future_paths["val_predictions"]),
            "metrics_path": str(future_paths["metrics"]),
            "test_predictions_sha256": sha256_file(future_paths["test_predictions"]),
            "val_predictions_sha256": sha256_file(future_paths["val_predictions"]),
            "metrics_sha256": sha256_file(future_paths["metrics"]),
        },
    }


def validate_strict_artifact_env() -> None:
    missing_env = [env_name for env_name in REQUIRED_ARTIFACT_ENVS if not os.getenv(env_name, "").strip()]
    if missing_env:
        raise RuntimeError(
            "Strict artifacts mode is enabled, but required env vars are missing: "
            + ", ".join(missing_env)
        )

    missing_files: list[str] = []
    for env_name in REQUIRED_ARTIFACT_ENVS:
        raw = os.getenv(env_name, "").strip()
        path = Path(raw)
        if not path.exists():
            missing_files.append(f"{env_name}={path}")
    if missing_files:
        raise RuntimeError(
            "Strict artifacts mode is enabled, but artifact files do not exist: "
            + ", ".join(missing_files)
        )


__all__ = [
    "BENCHMARK_REPORT_ENV",
    "CALIBRATION_MIN_SAMPLES_ENV",
    "ChampionRole",
    "DEFAULT_BENCHMARK_REPORT",
    "DEFAULT_METRICS",
    "DEFAULT_TEST_PRED",
    "DEFAULT_VAL_PRED",
    "ENABLE_RESIDUAL_CALIBRATION_ENV",
    "METRICS_ENV",
    "MODEL_MANIFEST_ENV",
    "ROLE_TO_MANIFEST_KEY",
    "SPLIT_TO_PATH",
    "Split",
    "env_flag",
    "file_meta",
    "get_active_artifacts",
    "get_resolved_artifact_paths",
    "legacy_default_role",
    "load_manifest_payload",
    "manifest_path",
    "manifest_role_section",
    "manifest_targets_active_artifacts",
    "normalized_path_str",
    "resolve_path",
    "resolve_role_artifact_paths",
    "sha256_file",
    "validate_strict_artifact_env",
    "watchlist_path",
]
