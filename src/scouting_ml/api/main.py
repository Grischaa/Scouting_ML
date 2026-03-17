"""FastAPI application entrypoint for Scouting ML."""
from __future__ import annotations

import logging
from threading import Lock

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import Response

from scouting_ml.api.routes_market_value import router as market_value_router
from scouting_ml.api.routes_players_nlp import router as players_nlp_router
from scouting_ml.core.runtime_config import load_api_runtime_config
from scouting_ml.services.market_value_service import (
    get_resolved_artifact_paths,
    health_payload,
    validate_strict_artifact_env,
)


logger = logging.getLogger(__name__)
API_CONFIG = load_api_runtime_config()
_STARTUP_LOCK = Lock()
_STARTUP_CHECKS_DONE = False


def _startup_checks() -> None:
    if API_CONFIG.strict_artifacts:
        validate_strict_artifact_env()
    paths = get_resolved_artifact_paths()
    logger.info(
        "Active artifacts | test=%s | val=%s | metrics=%s",
        paths["test_predictions_path"],
        paths["val_predictions_path"],
        paths["metrics_path"],
    )


def _ensure_startup_checks() -> None:
    global _STARTUP_CHECKS_DONE
    if _STARTUP_CHECKS_DONE:
        return
    with _STARTUP_LOCK:
        if _STARTUP_CHECKS_DONE:
            return
        _startup_checks()
        _STARTUP_CHECKS_DONE = True


app = FastAPI(title="Scouting ML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(API_CONFIG.cors_origins),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def startup_checks_middleware(request: Request, call_next) -> Response:
    _ensure_startup_checks()
    return await call_next(request)

# Register routers
app.include_router(players_nlp_router)
app.include_router(market_value_router)


@app.get("/", summary="API index")
async def root() -> dict:
    """Provide a lightweight index instead of a 404 on '/'."""
    return {
        "service": "scouting_ml_api",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "market_value_base": "/market-value",
    }


@app.get("/health", summary="Global API health")
async def health() -> dict:
    """Return global API health plus market-value artifact readiness."""
    return health_payload()


__all__ = ["app"]
