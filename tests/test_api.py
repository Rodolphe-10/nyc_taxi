"""Tests de l'API sans serveur MLflow, sur un petit modèle simulé."""

from __future__ import annotations

import lightgbm as lgb
import pytest
from fastapi.testclient import TestClient

from src.models.preprocess import TipPreprocessor
from src.optim.benchmarks import generate_taxi_df

_VALID = {
    "passenger_count": 1,
    "trip_distance": 2.5,
    "pickup_datetime": "2023-01-15T08:30:00",
    "dropoff_datetime": "2023-01-15T08:48:00",
    "pu_location_id": 161,
    "do_location_id": 236,
    "payment_type": 1,
    "fare_amount": 12.5,
}


def _train_tiny_model(directory) -> None:
    """Ajuste un booster court et les deux quantiles sur des courses simulées."""
    frame = generate_taxi_df(400, seed=42)
    preprocessor = TipPreprocessor(n_splits=2, add_holidays=False)
    target = preprocessor.target_series(frame)
    features = preprocessor.fit_transform_train(frame).reset_index(drop=True)
    target = target.reset_index(drop=True)
    dataset = lgb.Dataset(features, label=target)
    common = {
        "verbosity": -1,
        "learning_rate": 0.1,
        "num_leaves": 7,
        "max_depth": 3,
    }
    point = lgb.train({**common, "objective": "regression"}, dataset, num_boost_round=10)
    low = lgb.train(
        {**common, "objective": "quantile", "alpha": 0.1}, dataset, num_boost_round=10
    )
    high = lgb.train(
        {**common, "objective": "quantile", "alpha": 0.9}, dataset, num_boost_round=10
    )
    preprocessor.save(directory / "preprocessor.joblib")
    point.save_model(str(directory / "model.txt"))
    low.save_model(str(directory / "quantile_10.txt"))
    high.save_model(str(directory / "quantile_90.txt"))
    (directory / "version.txt").write_text("test", encoding="utf-8")


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Client dont le cycle de vie charge le modèle du dossier temporaire."""
    _train_tiny_model(tmp_path)
    monkeypatch.setenv("TIP_MODEL_DIR", str(tmp_path))
    monkeypatch.delenv("TIP_API_DISABLE_MODEL", raising=False)
    from src.api.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client_without_model(monkeypatch):
    """Client qui démarre alors qu'aucun modèle n'est chargé."""
    monkeypatch.setenv("TIP_API_DISABLE_MODEL", "1")
    from src.api.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_version(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": "test"}


def test_predict_valid_trip(client):
    response = client.post("/predict", json=_VALID)
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "test"
    assert 0 <= body["tip_pct_predicted"] <= 1
    low, high = body["confidence_interval"]
    assert 0 <= low <= high <= 1


def test_predict_batch(client):
    response = client.post("/predict/batch", json=[_VALID, _VALID])
    assert response.status_code == 200
    assert len(response.json()) == 2


@pytest.mark.parametrize(
    "field, value",
    [
        ("passenger_count", 0),
        ("passenger_count", 9),
        ("trip_distance", 0),
        ("trip_distance", 101),
        ("trip_distance", 0.05),
        ("pu_location_id", 0),
        ("pu_location_id", 266),
        ("do_location_id", 300),
        ("payment_type", 5),
        ("fare_amount", 0),
        ("fare_amount", 1.0),
        ("fare_amount", 250),
    ],
)
def test_invalid_field_is_422(client, field, value):
    payload = dict(_VALID)
    payload[field] = value
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert response.json()["detail"]


def test_duration_over_twelve_hours_is_422(client):
    payload = dict(_VALID)
    payload["dropoff_datetime"] = "2023-01-16T09:00:00"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "12" in response.text


def test_dropoff_before_pickup_is_422(client):
    payload = dict(_VALID)
    payload["dropoff_datetime"] = "2023-01-15T08:00:00"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_passenger_outside_model_filter_is_422(client):
    payload = dict(_VALID)
    payload["passenger_count"] = 8
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_missing_model_returns_503(client_without_model):
    health = client_without_model.get("/health")
    assert health.status_code == 200
    assert health.json()["model_version"] == "none"
    response = client_without_model.post("/predict", json=_VALID)
    assert response.status_code == 503
    assert response.json()["detail"] == "Modèle indisponible."
    assert "Traceback" not in response.text


def test_unexpected_error_is_generic_500(client, monkeypatch):
    def boom(self, request):
        del self, request
        raise RuntimeError("secret-traceback")

    monkeypatch.setattr("src.api.service.PredictionService.predict_one", boom)
    response = client.post("/predict", json=_VALID)
    assert response.status_code == 500
    assert response.json()["detail"] == "Erreur interne."
    assert "secret-traceback" not in response.text


def test_metrics_prometheus_format(client):
    client.get("/health")
    response = client.get("/metrics")
    assert response.status_code == 200
    text = response.text
    assert "tip_api_requests_total" in text
    assert 'tip_api_latency_seconds{quantile="0.5"}' in text
    assert 'tip_api_latency_seconds{quantile="0.95"}' in text
    assert "text/plain" in response.headers["content-type"]
