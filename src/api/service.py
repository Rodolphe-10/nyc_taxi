"""Chargement unique du modèle et prédiction d'une course brute."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.api.schemas import TripPredictionRequest, TripPredictionResponse
from src.models.preprocess import TipPreprocessor

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent
_REPORTS = _ROOT / "reports"


class TripRejected(ValueError):
    """La course passe Pydantic mais pas les filtres du prétraitement."""


class PredictionService:
    """Modèle de point, prétraitement et quantiles 10 % / 90 %."""

    def __init__(
        self,
        model: object,
        preprocessor: TipPreprocessor,
        quantile_low: Optional[lgb.Booster],
        quantile_high: Optional[lgb.Booster],
        version: str,
    ) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.quantile_low = quantile_low
        self.quantile_high = quantile_high
        self.version = version

    def predict_one(self, request: TripPredictionRequest) -> TripPredictionResponse:
        """Transforme une course brute et renvoie le pourcentage prédit."""
        frame = pd.DataFrame(
            [
                {
                    "tpep_pickup_datetime": request.pickup_datetime,
                    "PULocationID": request.pu_location_id,
                    "DOLocationID": request.do_location_id,
                    "trip_distance": request.trip_distance,
                    "fare_amount": request.fare_amount,
                    "passenger_count": request.passenger_count,
                    "payment_type": request.payment_type,
                }
            ]
        )
        try:
            features = self.preprocessor.transform(frame)
        except ValueError as exc:
            raise TripRejected(
                "La course est rejetée par le nettoyage : tarif entre 2,5 et 200, "
                "distance entre 0,1 et 100 miles, 1 à 6 passagers."
            ) from exc
        if features.empty:
            raise TripRejected("La course ne reste pas après le nettoyage.")

        point = float(np.clip(_as_vector(self.model.predict(features))[0], 0.0, 1.0))
        low, high = _interval(self.quantile_low, self.quantile_high, features, point)
        return TripPredictionResponse(
            tip_pct_predicted=point,
            confidence_interval=[low, high],
            model_version=self.version,
        )

    def predict_many(
        self, requests: Sequence[TripPredictionRequest]
    ) -> list[TripPredictionResponse]:
        """Prédit un lot. L'index de la première course rejetée est dans le message."""
        outputs: list[TripPredictionResponse] = []
        for index, request in enumerate(requests):
            try:
                outputs.append(self.predict_one(request))
            except TripRejected as exc:
                raise TripRejected(f"Course {index} : {exc}") from exc
        return outputs


def load_service() -> Optional[PredictionService]:
    """Charge le modèle une fois.

    ``TIP_API_DISABLE_MODEL=1`` force l'absence de modèle.
    ``TIP_MODEL_DIR`` pointe vers un dossier local et interdit le repli.
    Sinon le registre Production est essayé, puis ``reports/``.
    """
    if os.environ.get("TIP_API_DISABLE_MODEL") == "1":
        logger.warning("Chargement du modèle désactivé.")
        return None

    model_dir = os.environ.get("TIP_MODEL_DIR")
    if model_dir:
        path = Path(model_dir)
        if not (path / "preprocessor.joblib").is_file() or not (path / "model.txt").is_file():
            logger.error("Dossier modèle incomplet : %s", path)
            return None
        return _load_dir(path)

    registered = _load_registry()
    if registered is not None:
        return registered
    if (_REPORTS / "model.txt").is_file() and (_REPORTS / "preprocessor.joblib").is_file():
        logger.info("Repli sur l'artefact local reports/.")
        return _load_dir(_REPORTS)
    logger.warning("Aucun modèle disponible.")
    return None


def _load_dir(path: Path) -> PredictionService:
    """Recharge un booster, le prétraitement et les quantiles d'un dossier."""
    version_file = path / "version.txt"
    if version_file.is_file():
        version = version_file.read_text(encoding="utf-8").strip()
    else:
        version = _registry_version() or "local"
    return PredictionService(
        lgb.Booster(model_file=str(path / "model.txt")),
        TipPreprocessor.load(path / "preprocessor.joblib"),
        _booster_or_none(path / "quantile_10.txt"),
        _booster_or_none(path / "quantile_90.txt"),
        version or "local",
    )


def _load_registry() -> Optional[PredictionService]:
    """Charge ``models:/TipPredictor/Production`` si le prétraitement local existe."""
    if not (_REPORTS / "preprocessor.joblib").is_file():
        return None
    try:
        import mlflow
        import mlflow.pyfunc

        from src.models.train import get_tracking_uri

        mlflow.set_tracking_uri(get_tracking_uri())
        model = mlflow.pyfunc.load_model("models:/TipPredictor/Production")
    except ImportError:
        logger.info("Client MLflow absent, repli sur l'artefact local.")
        return None
    except Exception:
        logger.warning("Registre Production indisponible.", exc_info=True)
        return None
    version = _registry_version() or "Production"
    logger.info("Modèle Production chargé, version %s.", version)
    return PredictionService(
        model,
        TipPreprocessor.load(_REPORTS / "preprocessor.joblib"),
        _booster_or_none(_REPORTS / "quantile_10.txt"),
        _booster_or_none(_REPORTS / "quantile_90.txt"),
        version,
    )


def _registry_version() -> Optional[str]:
    path = _REPORTS / "registry.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("version")
    return None if value is None else str(value)


def _booster_or_none(path: Path) -> Optional[lgb.Booster]:
    if not path.is_file():
        return None
    return lgb.Booster(model_file=str(path))


def _as_vector(prediction: object) -> np.ndarray:
    return np.asarray(prediction, dtype=float).reshape(-1)


def _interval(
    low_model: Optional[lgb.Booster],
    high_model: Optional[lgb.Booster],
    features: pd.DataFrame,
    point: float,
) -> tuple[float, float]:
    """Bornes 10 % et 90 %, ramenées à l'intervalle [0, 1]."""
    if low_model is None or high_model is None:
        return point, point
    low = float(_as_vector(low_model.predict(features))[0])
    high = float(_as_vector(high_model.predict(features))[0])
    low, high = min(low, high), max(low, high)
    return float(np.clip(low, 0.0, 1.0)), float(np.clip(high, 0.0, 1.0))
