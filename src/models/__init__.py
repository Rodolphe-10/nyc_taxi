"""Modèles : entraînement LightGBM / MLflow et registre.

Les imports lourds restent différés : l'API charge le prétraitement sans
installer le client MLflow.
"""

from __future__ import annotations

__all__ = [
    "get_best_run",
    "list_all_runs",
    "load_production_model",
    "predict",
    "prepare_dataset",
    "register_best_model",
    "run_5_experiments",
    "run_grid_experiments",
    "run_optuna_search",
    "train_lightgbm",
]

_REGISTRY = {"load_production_model", "predict", "register_best_model"}
_TRAIN = {
    "get_best_run",
    "list_all_runs",
    "prepare_dataset",
    "run_5_experiments",
    "run_grid_experiments",
    "run_optuna_search",
    "train_lightgbm",
}


def __getattr__(name: str):
    """Charge le registre ou l'entraînement seulement si le symbole est demandé."""
    if name in _REGISTRY:
        from src.models import registry

        return getattr(registry, name)
    if name in _TRAIN:
        from src.models import train

        return getattr(train, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
