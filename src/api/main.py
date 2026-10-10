"""API FastAPI de prédiction du pourcentage de pourboire."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse

from src.api.schemas import TripPredictionRequest, TripPredictionResponse
from src.api.service import PredictionService, TripRejected, load_service

logger = logging.getLogger(__name__)


def _enable_prometheus(application: FastAPI) -> None:
    """Active les compteurs HTTP automatiques, en plus du résumé p50/p95."""
    try:
        from prometheus_fastapi_instrumentator import Instrumentator
    except ImportError:
        logger.info("prometheus-fastapi-instrumentator absent.")
        return
    Instrumentator(
        should_group_status_codes=False,
        excluded_handlers=["/metrics", "/docs", "/openapi.json", "/redoc", "/favicon.ico"],
    ).instrument(application)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle une seule fois. L'API démarre même s'il est absent."""
    app.state.service = load_service()
    app.state.request_count = 0
    app.state.latencies = []
    yield


app = FastAPI(title="NYC Taxi Tip Predictor", version="1.0.0", lifespan=lifespan)


class LatencyMiddleware:
    """Compte les requêtes et leur latence, hors exposition des métriques."""

    def __init__(self, asgi_app: Any) -> None:
        self.asgi_app = asgi_app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope.get("path") == "/metrics":
            await self.asgi_app(scope, receive, send)
            return
        started = time.perf_counter()
        try:
            await self.asgi_app(scope, receive, send)
        finally:
            application = scope["app"]
            application.state.request_count += 1
            application.state.latencies.append(time.perf_counter() - started)
            if len(application.state.latencies) > 5_000:
                application.state.latencies = application.state.latencies[-5_000:]


app.add_middleware(LatencyMiddleware)
_enable_prometheus(app)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """422 sans trace brute."""
    del request
    return JSONResponse(status_code=422, content={"detail": _validation_detail(exc)})


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Toute erreur imprévue reste dans les journaux."""
    del request
    logger.exception("Erreur interne : %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Erreur interne."})


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    """Disponibilité du service et version du modèle chargé."""
    service: PredictionService | None = request.app.state.service
    version = service.version if service is not None else "none"
    return {"status": "ok", "model_version": version}


@app.post("/predict", response_model=TripPredictionResponse)
def predict(request: Request, trip: TripPredictionRequest) -> TripPredictionResponse:
    """Prédit le pourcentage de pourboire d'une course."""
    service = _require_service(request)
    try:
        return service.predict_one(trip)
    except TripRejected as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Erreur interne.")
        raise HTTPException(status_code=500, detail="Erreur interne.") from exc


@app.post("/predict/batch", response_model=list[TripPredictionResponse])
def predict_batch(
    request: Request, trips: list[TripPredictionRequest]
) -> list[TripPredictionResponse]:
    """Prédit un tableau de courses."""
    service = _require_service(request)
    try:
        return service.predict_many(trips)
    except TripRejected as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Erreur interne.")
        raise HTTPException(status_code=500, detail="Erreur interne.") from exc


@app.get("/metrics")
def metrics(request: Request) -> PlainTextResponse:
    """Nombre de requêtes et latences p50 et p95, au format Prometheus."""
    latencies = np.asarray(request.app.state.latencies, dtype=float)
    if latencies.size == 0:
        p50 = p95 = 0.0
    else:
        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
    body = "\n".join(
        [
            "# HELP tip_api_requests_total Nombre de requêtes HTTP.",
            "# TYPE tip_api_requests_total counter",
            f"tip_api_requests_total {int(request.app.state.request_count)}",
            "# HELP tip_api_latency_seconds Latence des requêtes en secondes.",
            "# TYPE tip_api_latency_seconds summary",
            f'tip_api_latency_seconds{{quantile="0.5"}} {p50:.6f}',
            f'tip_api_latency_seconds{{quantile="0.95"}} {p95:.6f}',
            "",
        ]
    )
    try:
        from prometheus_client import generate_latest

        body += generate_latest().decode("utf-8")
    except ImportError:
        logger.info("prometheus_client absent.")
    return PlainTextResponse(body, media_type="text/plain; version=0.0.4")


def _require_service(request: Request) -> PredictionService:
    service = request.app.state.service
    if service is None:
        raise HTTPException(status_code=503, detail="Modèle indisponible.")
    return service


def _validation_detail(exc: RequestValidationError) -> list[dict[str, Any]]:
    """Ne conserve que l'emplacement et le message, pas l'objet d'entrée."""
    details = []
    for error in exc.errors():
        details.append(
            {
                "loc": [str(item) for item in error.get("loc", ())],
                "msg": str(error.get("msg", "Entrée invalide.")),
                "type": str(error.get("type", "value_error")),
            }
        )
    return details


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=False)
