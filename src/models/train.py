"""Entraînement LightGBM + traçage MLflow pour la prédiction de tip_pct (NYC Taxi)."""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import tempfile
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Exécution : « python src/models/train.py » — la racine du package est tp_taxi/.
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import optuna
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.features.encoding import BayesianTargetEncoder
from src.features.temporal import TemporalFeaturizer
from src.optim.benchmarks import generate_taxi_df
from src.optim.memory import optimize_dtypes

logger = logging.getLogger(__name__)

# Toujours sous la racine du projet (évite mlruns « ailleurs » si cwd ≠ tp_taxi).
_MLRUNS_DIR = (_ROOT / "mlruns").resolve()
# URI « file:///… » : sous Windows un chemin « C:\… » est parsé comme schéma « c » (invalide pour MLflow).
MLFLOW_TRACKING_URI = _MLRUNS_DIR.as_uri()
EXPERIMENT_NAME = "NYC_Taxi_TipPredictor"
TARGET = "tip_pct"
RANDOM_STATE = 42

FEATURES = [
    "pickup_hour",
    "pickup_weekday",
    "pickup_month",
    "pickup_quarter",
    "is_weekend",
    "is_rush_hour",
    "is_late_night",
    "pickup_hour_sin",
    "pickup_hour_cos",
    "pickup_weekday_sin",
    "pickup_weekday_cos",
    "PULocationID_target_enc",
    "DOLocationID_target_enc",
    "trip_distance",
    "fare_amount",
    "passenger_count",
    "payment_type",
]


def _set_mlflow_uri() -> None:
    """Configure l’URI de tracking ; crée le dossier si besoin."""
    _MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def prepare_dataset(
    n_rows: int = 500_000,
    use_real_data: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """Prépare le jeu de données (temporel + target encoding + dtypes) pour l’entraînement."""
    raw_dir = _ROOT / "data" / "raw"
    parquet_files = sorted(raw_dir.glob("*.parquet")) if raw_dir.is_dir() else []

    if use_real_data and parquet_files:
        dfs: List[pd.DataFrame] = []
        total = 0
        for p in parquet_files:
            if total >= n_rows:
                break
            part = pd.read_parquet(p)
            room = n_rows - total
            if len(part) <= room:
                dfs.append(part)
                total += len(part)
            else:
                dfs.append(
                    part.sample(n=room, random_state=RANDOM_STATE + total).reset_index(
                        drop=True
                    )
                )
                total = n_rows
                break
        df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        logger.info(
            "%s lignes lues depuis data/raw/ (plafond n_rows=%s)",
            f"{len(df):,}",
            f"{n_rows:,}",
        )
    else:
        df = generate_taxi_df(n_rows, seed=RANDOM_STATE)
        logger.info("Dataset simulé : %s lignes", f"{n_rows:,}")

    df = df[df["fare_amount"].between(2.5, 200)]
    df = df[df["trip_distance"].between(0.1, 100)]
    df = df[df["passenger_count"].between(1, 6)]
    df[TARGET] = (df["tip_amount"] / df["fare_amount"].replace(0, np.nan)).clip(0, 1)
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET] >= 0]

    featurizer = TemporalFeaturizer(
        cyclic_encoding=True,
        add_holidays=False,
        add_rush_hours=True,
    )
    df = featurizer.run(df)

    encoder = BayesianTargetEncoder(
        columns=["PULocationID", "DOLocationID"],
        target=TARGET,
        smoothing=10.0,
    )
    df = encoder.fit_transform(df)

    df = optimize_dtypes(df, verbose=False)

    available = [f for f in FEATURES if f in df.columns]
    X = df[available].copy()
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.fillna(0)
    y = df[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    logger.info(
        "Train : %s | Val : %s | Features : %s",
        f"{len(X_train):,}",
        f"{len(X_val):,}",
        len(available),
    )
    return X_train, X_val, y_train, y_val, available


def train_lightgbm(
    params: Dict[str, Any],
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    run_name: str = "lgbm_run",
    experiment_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Entraîne un modèle LightGBM et journalise l’expérience dans MLflow."""
    _set_mlflow_uri()
    exp_name = experiment_name or EXPERIMENT_NAME
    try:
        mlflow.set_experiment(exp_name)
    except FileNotFoundError as exc:
        logger.error("Impossible de créer ou d’accéder à l’expérience MLflow : %s", exc)
        raise

    train_params = {k: v for k, v in params.items() if k != "n_estimators"}
    n_boost = int(params.get("n_estimators", 500))

    run_id: Optional[str] = None
    rmse = mae = r2 = 0.0
    model: Optional[lgb.Booster] = None
    fi = pd.DataFrame()

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_val", len(X_val))
        mlflow.log_param("n_features", X_train.shape[1])

        dtrain = lgb.Dataset(
            X_train, label=y_train, feature_name=list(X_train.columns)
        )
        dval = lgb.Dataset(X_val, label=y_val, feature_name=list(X_val.columns))

        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ]

        model = lgb.train(
            train_params,
            dtrain,
            num_boost_round=n_boost,
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=callbacks,
        )

        best_iter = getattr(model, "best_iteration", None)
        if best_iter is None:
            best_iter = model.current_iteration()

        y_pred = model.predict(X_val, num_iteration=best_iter)
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        mae = float(mean_absolute_error(y_val, y_pred))
        y_mean = float(y_val.mean())
        ss_tot = float(np.sum((y_val - y_mean) ** 2))
        r2 = (
            1.0 - float(np.sum((y_val - y_pred) ** 2)) / ss_tot
            if ss_tot > 0
            else float("nan")
        )

        mlflow.log_metrics(
            {
                "rmse_val": rmse,
                "mae_val": mae,
                "r2_val": r2,
                "best_iter": float(best_iter),
            }
        )

        names = model.feature_name() or list(X_train.columns)
        fi = (
            pd.DataFrame(
                {
                    "feature": names,
                    "importance": model.feature_importance(importance_type="gain"),
                }
            )
            .sort_values("importance", ascending=False)
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as tmp:
            fi_path = tmp.name
        fi.to_csv(fi_path, index=False)
        mlflow.log_artifact(fi_path)
        try:
            os.remove(fi_path)
        except OSError:
            pass

        signature = infer_signature(X_train, y_pred)
        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=X_val.head(5),
        )

        run_id = run.info.run_id
        logger.info(
            "Run MLflow : %s | RMSE=%.4f | MAE=%.4f | R²=%.4f",
            run_id,
            rmse,
            mae,
            r2,
        )

    assert model is not None and run_id is not None
    return {
        "run_id": run_id,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "model": model,
        "features": list(X_train.columns),
        "fi": fi,
        "params": params,
    }


def _generate_grid_configs(n: int, seed: int = RANDOM_STATE) -> List[Dict[str, Any]]:
    """Construit jusqu’à n configurations distinctes (sous-ensemble aléatoire d’une grille)."""
    lrs = [0.01, 0.02, 0.05, 0.08, 0.1, 0.12]
    n_ests = [180, 250, 320, 400, 500, 650]
    depths = [4, 5, 6, 7, 8, 9]
    leaves = [31, 50, 63, 95, 127, 180]
    subsamples = [0.65, 0.75, 0.85, 0.9, 1.0]
    colsample = [0.75, 0.9, 1.0]
    product = list(
        itertools.product(lrs, n_ests, depths, leaves, subsamples, colsample)
    )
    rng = np.random.default_rng(seed)
    rng.shuffle(product)
    configs: List[Dict[str, Any]] = []
    for lr, n_est, depth, num_leaves, subs, colbt in product:
        cap = max(2, (1 << min(depth, 20)) - 1)
        nl = int(min(num_leaves, cap))
        configs.append(
            {
                "objective": "regression",
                "metric": "rmse",
                "verbosity": -1,
                "boosting_type": "gbdt",
                "learning_rate": float(lr),
                "n_estimators": int(n_est),
                "max_depth": int(depth),
                "num_leaves": nl,
                "subsample": float(subs),
                "colsample_bytree": float(colbt),
            }
        )
        if len(configs) >= n:
            break
    return configs


def run_optuna_search(
    n_trials: int = 10,
    n_rows: int = 100_000,
    experiment_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Recherche d’hyperparamètres Optuna (objectif : minimiser le RMSE de validation)."""
    X_train, X_val, y_train, y_val, _ = prepare_dataset(
        n_rows=n_rows, use_real_data=True
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 20, 200),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        result = train_lightgbm(
            params,
            X_train,
            X_val,
            y_train,
            y_val,
            run_name=f"optuna_trial_{trial.number}",
            experiment_name=experiment_name,
        )
        return float(result["rmse"])

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = dict(study.best_params)
    best_params["objective"] = "regression"
    best_params["metric"] = "rmse"
    best_params["verbosity"] = -1
    best_params["boosting_type"] = "gbdt"

    logger.info("Meilleur RMSE Optuna : %.4f", study.best_value)
    logger.info("Meilleurs params : %s", best_params)

    return {
        "best_params": best_params,
        "best_value": study.best_value,
        "study": study,
    }


def run_grid_experiments(
    n_rows: int = 100_000,
    n_configs: int = 15,
    *,
    configs: Optional[List[Dict[str, Any]]] = None,
    experiment_name: Optional[str] = None,
    seed: int = RANDOM_STATE,
) -> List[Dict[str, Any]]:
    """Lance plusieurs runs MLflow (grille hyperparamètres ou liste fournie)."""
    if configs is None:
        grid = _generate_grid_configs(n_configs, seed=seed)
    else:
        grid = list(configs)

    X_train, X_val, y_train, y_val, _ = prepare_dataset(
        n_rows=n_rows, use_real_data=True
    )

    results: List[Dict[str, Any]] = []
    exp_label = experiment_name or EXPERIMENT_NAME
    for i, params in enumerate(grid):
        result = train_lightgbm(
            params,
            X_train,
            X_val,
            y_train,
            y_val,
            run_name=f"grid_{i + 1:03d}",
            experiment_name=experiment_name,
        )
        results.append(result)
        print(
            f"Run {i + 1}/{len(grid)} | RMSE={result['rmse']:.4f} "
            f"(expérience MLflow « {exp_label} »)"
        )

    df_results = pd.DataFrame(
        [
            {
                "run": i + 1,
                "rmse": r["rmse"],
                "mae": r["mae"],
                "r2": r["r2"],
                "lr": r["params"]["learning_rate"],
                "depth": r["params"]["max_depth"],
            }
            for i, r in enumerate(results)
        ]
    ).sort_values("rmse")
    print(f"\n=== RÉSUMÉ DES {len(results)} RUNS ===")
    print(df_results.to_string(index=False))
    return results


def run_5_experiments(
    n_rows: int = 100_000,
    experiment_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Lance les 5 configurations historiques du TP (rétrocompatibilité)."""
    legacy: List[Dict[str, Any]] = [
        {
            "learning_rate": 0.05,
            "n_estimators": 300,
            "max_depth": 6,
            "num_leaves": 63,
            "subsample": 0.8,
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "boosting_type": "gbdt",
        },
        {
            "learning_rate": 0.1,
            "n_estimators": 200,
            "max_depth": 4,
            "num_leaves": 31,
            "subsample": 0.9,
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "boosting_type": "gbdt",
        },
        {
            "learning_rate": 0.01,
            "n_estimators": 500,
            "max_depth": 8,
            "num_leaves": 127,
            "subsample": 0.7,
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "boosting_type": "gbdt",
        },
        {
            "learning_rate": 0.05,
            "n_estimators": 400,
            "max_depth": 5,
            "num_leaves": 50,
            "subsample": 0.85,
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "boosting_type": "gbdt",
        },
        {
            "learning_rate": 0.08,
            "n_estimators": 250,
            "max_depth": 7,
            "num_leaves": 80,
            "subsample": 0.75,
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "boosting_type": "gbdt",
        },
    ]
    return run_grid_experiments(
        n_rows,
        n_configs=len(legacy),
        configs=legacy,
        experiment_name=experiment_name,
    )


def get_best_run(experiment_name: Optional[str] = None) -> Dict[str, Any]:
    """Retourne le meilleur run MLflow (tri sur rmse_val croissant)."""
    _set_mlflow_uri()
    exp_name = experiment_name or EXPERIMENT_NAME
    try:
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(exp_name)
    except FileNotFoundError:
        logger.warning("Dossier MLflow introuvable.")
        return {}

    if exp is None:
        return {}

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.rmse_val ASC"],
        max_results=1,
    )
    if not runs:
        return {}

    best = runs[0]
    return {
        "run_id": best.info.run_id,
        "rmse": best.data.metrics.get("rmse_val"),
        "mae": best.data.metrics.get("mae_val"),
        "r2": best.data.metrics.get("r2_val"),
        "params": dict(best.data.params),
        "status": best.info.status,
    }


def list_all_runs(experiment_name: Optional[str] = None) -> pd.DataFrame:
    """Liste tous les runs de l’expérience NYC Taxi (tri RMSE)."""
    _set_mlflow_uri()
    exp_name = experiment_name or EXPERIMENT_NAME
    try:
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(exp_name)
    except FileNotFoundError:
        logger.warning("Dossier MLflow introuvable.")
        return pd.DataFrame()

    if exp is None:
        return pd.DataFrame()

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.rmse_val ASC"],
    )
    rows = []
    for run in runs:
        rows.append(
            {
                "run_id": run.info.run_id[:8],
                "run_name": run.data.tags.get("mlflow.runName", ""),
                "rmse": run.data.metrics.get("rmse_val"),
                "mae": run.data.metrics.get("mae_val"),
                "r2": run.data.metrics.get("r2_val"),
                "lr": run.data.params.get("learning_rate"),
                "max_depth": run.data.params.get("max_depth"),
                "n_est": run.data.params.get("n_estimators"),
                "status": run.info.status,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Entraînement LightGBM + MLflow")
    parser.add_argument(
        "--mode",
        choices=["quick", "full", "optuna"],
        default="quick",
    )
    parser.add_argument("--n_rows", type=int, default=100_000)
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument(
        "--n_experiments",
        type=int,
        default=15,
        help="Mode full : nombre de runs (sous-ensemble aléatoire de la grille).",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Nom de l’expérience MLflow (onglet dans l’UI). Défaut : NYC_Taxi_TipPredictor.",
    )
    parser.add_argument(
        "--legacy_five",
        action="store_true",
        help="Mode full : n’exécuter que les 5 configurations historiques du TP.",
    )
    parser.add_argument(
        "--grid_seed",
        type=int,
        default=RANDOM_STATE,
        help="Mode full : graine numpy pour tirer la sous-grille (reproductibilité).",
    )
    args = parser.parse_args()

    if args.mode == "quick":
        print("=== MODE QUICK : 1 run MLflow ===")
        X_train, X_val, y_train, y_val, _ = prepare_dataset(
            args.n_rows, use_real_data=True
        )
        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "n_estimators": 200,
            "max_depth": 6,
            "num_leaves": 63,
            "verbosity": -1,
            "boosting_type": "gbdt",
        }
        result = train_lightgbm(
            params,
            X_train,
            X_val,
            y_train,
            y_val,
            run_name="quick_run",
            experiment_name=args.experiment,
        )
        print(f"RMSE={result['rmse']:.4f} | R²={result['r2']:.4f}")
        print(f"Run ID : {result['run_id']}")
        ui_uri = _MLRUNS_DIR.as_uri()
        print(
            "MLflow UI — lance exactement (depuis n’importe quel répertoire) :\n"
            f'  mlflow ui --port 5000 --backend-store-uri "{ui_uri}"\n'
            "Puis ouvre dans le navigateur : http://127.0.0.1:5000\n"
            "(Sous Windows le navigateur ne s’ouvre pas toujours automatiquement ; "
            "le port 5000 peut être pris par une autre app — change --port si besoin.)"
        )

    elif args.mode == "full":
        exp = args.experiment or EXPERIMENT_NAME
        if args.legacy_five:
            print(f"=== MODE FULL : 5 runs (grille historique) — « {exp} » ===")
            run_5_experiments(args.n_rows, experiment_name=args.experiment)
        else:
            print(
                f"=== MODE FULL : {args.n_experiments} runs (grille étendue) — « {exp} » ==="
            )
            run_grid_experiments(
                args.n_rows,
                n_configs=args.n_experiments,
                experiment_name=args.experiment,
                seed=args.grid_seed,
            )

    elif args.mode == "optuna":
        print(f"=== MODE OPTUNA : {args.n_trials} trials — expérience MLflow « {args.experiment or EXPERIMENT_NAME} » ===")
        run_optuna_search(
            args.n_trials, args.n_rows, experiment_name=args.experiment
        )
