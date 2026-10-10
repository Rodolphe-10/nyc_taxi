"""Enregistrement et chargement des modèles dans le Model Registry MLflow."""

from __future__ import annotations

import json
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
from mlflow.tracking import MlflowClient

from src.models.train import (
    EXPERIMENT_NAME,
    get_tracking_uri,
    list_all_runs,
)

logger = logging.getLogger(__name__)


def _set_mlflow_uri() -> None:
    """Pointe le client MLflow vers l'URI configurée."""
    uri = get_tracking_uri()
    try:
        mlflow.set_tracking_uri(uri)
    except FileNotFoundError:
        logger.warning("URI MLflow introuvable : %s", uri)
        raise


def _reported_n_train() -> Optional[str]:
    """Effectif d'entraînement écrit par le dernier ``python -m src.models.train``."""
    path = _ROOT / "reports" / "metrics.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("n_train")
    return None if value is None else str(value)


def _has_logged_model(client: MlflowClient, experiment_id: str, run_id: str) -> bool:
    """Vrai si le run a journalisé un modèle sous le nom ``model``."""
    logged = client.search_logged_models(
        experiment_ids=[experiment_id],
        filter_string=f"source_run_id='{run_id}'",
    )
    return any(getattr(item, "name", None) == "model" for item in logged)


def select_production_run(experiment_name: Optional[str] = None) -> dict:
    """Choisit le run à promouvoir.

    Le RMSE le plus bas de tout l'historique peut venir d'un entraînement
    antérieur au filtre des dates 2023. On retient donc le meilleur run qui
    possède un modèle journalisé et le même effectif que ``reports/metrics.json``.
    """
    _set_mlflow_uri()
    client = MlflowClient()
    exp_name = experiment_name or EXPERIMENT_NAME
    experiment = client.get_experiment_by_name(exp_name)
    if experiment is None:
        raise RuntimeError(f"Expérience MLflow introuvable : {exp_name}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse_val ASC"],
        max_results=40,
    )
    cohort = _reported_n_train()
    ranked = []
    for run in runs:
        rmse = run.data.metrics.get("rmse_val")
        if rmse is None:
            continue
        if cohort is not None and str(run.data.params.get("n_train")) != cohort:
            continue
        if not _has_logged_model(client, experiment.experiment_id, run.info.run_id):
            continue
        ranked.append(run)

    if not ranked and cohort is not None:
        logger.warning(
            "Aucun modèle pour n_train=%s. Repli sur le meilleur RMSE disponible.",
            cohort,
        )
        for run in runs:
            if run.data.metrics.get("rmse_val") is None:
                continue
            if _has_logged_model(client, experiment.experiment_id, run.info.run_id):
                ranked.append(run)

    if not ranked:
        raise RuntimeError("Aucun run avec un modèle journalisé.")

    best = min(ranked, key=lambda run: (run.data.metrics["rmse_val"], -run.info.start_time))
    return {
        "run_id": best.info.run_id,
        "rmse": best.data.metrics.get("rmse_val"),
        "mae": best.data.metrics.get("mae_val"),
        "r2": best.data.metrics.get("r2_val"),
        "n_train": best.data.params.get("n_train"),
        "run_name": best.data.tags.get("mlflow.runName"),
    }


def register_best_model(
    model_name: str = "TipPredictor",
    experiment_name: Optional[str] = None,
) -> str:
    """Enregistre le meilleur run et le passe au stage Production.

    L'énoncé appelle ``transition_model_version_stage``. MLflow 3 conserve
    cette méthode, marquée dépréciée au profit des alias.
    """
    _set_mlflow_uri()
    best = select_production_run(experiment_name)
    model_uri = f"runs:/{best['run_id']}/model"
    result = mlflow.register_model(model_uri, model_name)

    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Production",
        archive_existing_versions=True,
    )
    rmse = best.get("rmse")
    rmse_txt = f"{float(rmse):.4f}" if rmse is not None else "n/a"
    logger.info(
        "Modèle '%s' v%s → Production | run=%s | RMSE=%s",
        model_name,
        result.version,
        best["run_id"],
        rmse_txt,
    )
    report = _ROOT / "reports" / "registry.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "version": str(result.version),
                "stage": "Production",
                "model_uri": f"models:/{model_name}/Production",
                "source_run_id": best["run_id"],
                "run_name": best.get("run_name"),
                "rmse_val": best.get("rmse"),
                "n_train": best.get("n_train"),
            },
            indent=2,
        ),
        encoding="utf-8",
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
