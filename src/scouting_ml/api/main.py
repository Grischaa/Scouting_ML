"""FastAPI application entrypoint for Scouting ML."""
from __future__ import annotations

import logging
from threading import Lock
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.status import HTTP_503_SERVICE_UNAVAILABLE

from scouting_ml.api.routes_market_value import router as market_value_router
from scouting_ml.api.routes_players_nlp import router as players_nlp_router
from scouting_ml.api.routes_team import router as team_router
from scouting_ml.core.runtime_config import load_api_runtime_config
from scouting_ml.services.market_value_service import (
    get_resolved_artifact_paths,
    health_payload,
    validate_strict_artifact_env,
)
from scouting_ml.team.db import create_all_tables, team_mode_enabled


logger = logging.getLogger(__name__)
API_CONFIG = load_api_runtime_config()
_STARTUP_LOCK = Lock()
_STARTUP_CHECKS_DONE = False
_STARTUP_CHECKS_ERROR: Exception | None = None
_STARTUP_CHECK_EXEMPT_PATHS = {"/health", "/market-value/health"}
_STARTUP_CHECK_PROTECTED_PREFIXES = ("/market-value",)


def _strict_artifacts_enabled() -> bool:
    return load_api_runtime_config().strict_artifacts


def _startup_checks() -> None:
    if team_mode_enabled():
        create_all_tables()
    if _strict_artifacts_enabled():
        validate_strict_artifact_env()
    paths = get_resolved_artifact_paths()
    logger.info(
        "Active artifacts | test=%s | val=%s | metrics=%s",
        paths["test_predictions_path"],
        paths["val_predictions_path"],
        paths["metrics_path"],
    )


def _ensure_startup_checks(*, raise_on_failure: bool = True) -> None:
    global _STARTUP_CHECKS_DONE, _STARTUP_CHECKS_ERROR
    if _STARTUP_CHECKS_DONE:
        if _STARTUP_CHECKS_ERROR is not None and raise_on_failure:
            raise _STARTUP_CHECKS_ERROR
        return
    with _STARTUP_LOCK:
        if _STARTUP_CHECKS_DONE:
            if _STARTUP_CHECKS_ERROR is not None and raise_on_failure:
                raise _STARTUP_CHECKS_ERROR
            return
        try:
            _startup_checks()
        except Exception as exc:
            logger.exception("Startup checks failed")
            _STARTUP_CHECKS_ERROR = exc
        finally:
            _STARTUP_CHECKS_DONE = True

        if _STARTUP_CHECKS_ERROR is not None and raise_on_failure:
            raise _STARTUP_CHECKS_ERROR


app = FastAPI(title="Scouting ML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(API_CONFIG.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def startup_checks_middleware(request: Request, call_next) -> Response:
    path = request.url.path
    _ensure_startup_checks(raise_on_failure=False)
    if (
        _STARTUP_CHECKS_ERROR is not None
        and path not in _STARTUP_CHECK_EXEMPT_PATHS
        and path.startswith(_STARTUP_CHECK_PROTECTED_PREFIXES)
    ):
        payload = health_payload()
        payload["detail"] = (
            "Market-value artifacts are not ready. Check /market-value/health for readiness details."
        )
        return JSONResponse(status_code=HTTP_503_SERVICE_UNAVAILABLE, content=payload)
    return await call_next(request)

# Register routers
app.include_router(players_nlp_router)
app.include_router(market_value_router)
app.include_router(team_router)

_STATIC_APP_DIR = Path(__file__).resolve().parents[1] / "website" / "static"
if _STATIC_APP_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(_STATIC_APP_DIR), html=True), name="scoutml_frontend")


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
async def health(response: Response) -> dict:
    """Return global API health plus market-value artifact readiness."""
    payload = health_payload()
    if payload.get("status") != "ok":
        response.status_code = HTTP_503_SERVICE_UNAVAILABLE
    return payload


__all__ = ["app"]
