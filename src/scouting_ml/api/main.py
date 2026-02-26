"""FastAPI application entrypoint for Scouting ML."""
from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from scouting_ml.api.routes_market_value import router as market_value_router
from scouting_ml.api.routes_players_nlp import router as players_nlp_router
from scouting_ml.services.market_value_service import (
    get_resolved_artifact_paths,
    health_payload,
    validate_strict_artifact_env,
)


def _cors_origins_from_env() -> list[str]:
    raw = os.getenv("SCOUTING_API_CORS_ORIGINS", "*").strip()
    if not raw or raw == "*":
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


logger = logging.getLogger(__name__)


app = FastAPI(title="Scouting ML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins_from_env(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(players_nlp_router)
app.include_router(market_value_router)


@app.on_event("startup")
def startup_checks() -> None:
    if _env_flag("SCOUTING_STRICT_ARTIFACTS", default=False):
        validate_strict_artifact_env()
    paths = get_resolved_artifact_paths()
    logger.info(
        "Active artifacts | test=%s | val=%s | metrics=%s",
        paths["test_predictions_path"],
        paths["val_predictions_path"],
        paths["metrics_path"],
    )


@app.get("/", summary="API index")
def root() -> dict:
    """Provide a lightweight index instead of a 404 on '/'."""
    return {
        "service": "scouting_ml_api",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "market_value_base": "/market-value",
    }


@app.get("/health", summary="Global API health")
def health() -> dict:
    """Return global API health plus market-value artifact readiness."""
    return health_payload()


__all__ = ["app"]
