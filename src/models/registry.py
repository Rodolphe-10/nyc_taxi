"""Enregistrement et chargement des modèles dans le Model Registry MLflow."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd

from src.models.train import (
    MLFLOW_TRACKING_URI,
    get_best_run,
    list_all_runs,
)

logger = logging.getLogger(__name__)


def _set_mlflow_uri() -> None:
    """Pointe le client MLflow vers le store local."""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    except FileNotFoundError:
        logger.warning("URI MLflow introuvable : %s", MLFLOW_TRACKING_URI)
        raise


def register_best_model(
    model_name: str = "TipPredictor",
    experiment_name: Optional[str] = None,
) -> str:
    """Enregistre le meilleur run (RMSE) dans le Model Registry et passe la version en Production."""
    _set_mlflow_uri()
    best = get_best_run(experiment_name)
    if not best or not best.get("run_id"):
        raise RuntimeError("Aucun run disponible — entraîne d’abord un modèle (train.py).")

    model_uri = f"runs:/{best['run_id']}/model"
    result = mlflow.register_model(model_uri, model_name)

    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Production",
    )
    rmse = best.get("rmse")
    rmse_txt = f"{float(rmse):.4f}" if rmse is not None else "n/a"
    logger.info(
        "Modèle '%s' v%s → Production | RMSE=%s",
        model_name,
        result.version,
        rmse_txt,
    )
    return str(result.version)


def load_production_model(model_name: str = "TipPredictor") -> Any:
    """Charge le modèle pyfunc enregistré en stage Production."""
    _set_mlflow_uri()
    model_uri = f"models:/{model_name}/Production"
    model = mlflow.pyfunc.load_model(model_uri)
    logger.info("Modèle '%s/Production' chargé", model_name)
    return model


def predict(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Applique le modèle pyfunc sur un DataFrame de features."""
    return model.predict(X)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        version = register_best_model()
        print(f"Modèle enregistré : version {version}")
        model = load_production_model()
        print(f"Modèle chargé : {type(model)}")
    except Exception as exc:
        print(f"Échec registry : {exc}")
        print("Résumé des runs :")
        print(list_all_runs())
