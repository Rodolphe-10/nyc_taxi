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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
import mlflow
import mlflow.lightgbm

# Les versions récentes de MLflow refusent le dépôt fichier tant que cet
# opt-in n'est pas posé. Le suivi local reste dans mlruns/.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
import numpy as np
import optuna
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from src.models.preprocess import (
    FEATURES,
    RANDOM_STATE,
    TARGET,
    TipPreprocessor,
    temporal_split,
)
from src.optim.benchmarks import generate_taxi_df

logger = logging.getLogger(__name__)

# Toujours sous la racine du projet (évite mlruns « ailleurs » si cwd ≠ tp_taxi).
_MLRUNS_DIR = (_ROOT / "mlruns").resolve()
_REPORTS_DIR = _ROOT / "reports"
EXPERIMENT_NAME = "NYC_Taxi_TipPredictor"


def get_tracking_uri() -> str:
    """URI MLflow : variable d'environnement, sinon le dossier local mlruns/."""
    configured = os.environ.get("MLFLOW_TRACKING_URI")
    if configured:
        return configured
    _MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    return _MLRUNS_DIR.as_uri()


def _set_mlflow_uri() -> None:
    """Configure l’URI de tracking ; crée le dossier local si besoin."""
    mlflow.set_tracking_uri(get_tracking_uri())


def _load_raw_frame(n_rows: int, use_real_data: bool) -> pd.DataFrame:
    """Charge un préfixe chronologique des Parquet, ou un jeu simulé."""
    raw_dir = _ROOT / "data" / "raw"
    parquet_files = sorted(raw_dir.glob("*.parquet")) if raw_dir.is_dir() else []
    if use_real_data and parquet_files:
        parts = [pd.read_parquet(path) for path in parquet_files]
        df = pd.concat(parts, ignore_index=True)
        df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
        in_2023 = df["tpep_pickup_datetime"].between(
            "2023-01-01", "2024-01-01", inclusive="left"
        )
        logger.info(
            "%s lignes hors 2023 écartées avant le plafond chronologique",
            f"{int((~in_2023).sum()):,}",
        )
        df = df.loc[in_2023].sort_values("tpep_pickup_datetime").reset_index(drop=True)
        if len(df) > n_rows:
            df = df.iloc[:n_rows].copy()
        logger.info(
            "%s lignes lues depuis data/raw/ (plafond n_rows=%s, ordre chronologique)",
            f"{len(df):,}",
            f"{n_rows:,}",
        )
        return df
    logger.info("Dataset simulé : %s lignes", f"{n_rows:,}")
    return generate_taxi_df(n_rows, seed=RANDOM_STATE)


def _regression_scores(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """RMSE, MAE et R². Le R² vaut NaN si la cible est constante."""
    y_values = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_values, pred)))
    mae = float(mean_absolute_error(y_values, pred))
    ss_tot = float(np.sum((y_values - float(y_values.mean())) ** 2))
    r2 = 1.0 - float(np.sum((y_values - pred) ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def _card_mask(X: pd.DataFrame) -> pd.Series:
    """Courses payées par carte (payment_type == 1)."""
    return X["payment_type"].astype(float) == 1.0


def prepare_dataset(
    n_rows: int = 500_000,
    use_real_data: bool = True,
    split: str = "temporal",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """Prépare le jeu sans fuite : séparation d'abord, encodeur ajusté sur l'entraînement.

    ``split="temporal"`` (défaut) coupe le jeu dans l'ordre chronologique.
    ``split="random"`` retrouve l'ancienne séparation aléatoire.
    Le prétraitement ajusté est conservé dans ``prepare_dataset.preprocessor``.
    """
    if split not in {"temporal", "random"}:
        raise ValueError("split doit valoir 'temporal' ou 'random'.")

    df = _load_raw_frame(n_rows, use_real_data)
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    period_min = str(df["tpep_pickup_datetime"].min())
    period_max = str(df["tpep_pickup_datetime"].max())

    if split == "random":
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
    else:
        train_df, val_df = temporal_split(df, test_size=0.2)

    preprocessor = TipPreprocessor(smoothing=10.0, n_splits=5, add_holidays=False)
    y_train = preprocessor.target_series(train_df)
    y_val = preprocessor.target_series(val_df)
    X_train = preprocessor.fit_transform_train(train_df)
    X_val = preprocessor.transform(val_df)
    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    if len(X_train) != len(y_train) or len(X_val) != len(y_val):
        raise RuntimeError("Les variables et la cible n'ont pas le même nombre de lignes.")

    used_real = use_real_data and any((_ROOT / "data" / "raw").glob("*.parquet"))
    prepare_dataset.preprocessor = preprocessor  # type: ignore[attr-defined]
    prepare_dataset.train_zones = preprocessor.zone_ids(train_df)  # type: ignore[attr-defined]
    prepare_dataset.val_zones = preprocessor.zone_ids(val_df)  # type: ignore[attr-defined]
    if len(prepare_dataset.train_zones) != len(X_train) or len(prepare_dataset.val_zones) != len(X_val):  # type: ignore[attr-defined]
        raise RuntimeError("Les identifiants de zones ne sont pas alignés sur les variables.")
    prepare_dataset.metadata = {  # type: ignore[attr-defined]
        "n_rows_loaded": int(len(df)),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "period_min": period_min,
        "period_max": period_max,
        "split": split,
        "features": list(X_train.columns),
        "use_real_data": used_real,
    }
    logger.info(
        "Train : %s | Val : %s | Features : %s | split=%s",
        f"{len(X_train):,}",
        f"{len(X_val):,}",
        X_train.shape[1],
        split,
    )
    return X_train, X_val, y_train, y_val, list(X_train.columns)


def _save_importance_png(fi: pd.DataFrame, path: str) -> None:
    """Enregistre l'importance des variables en barres horizontales."""
    top = fi.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance"], color="#1f4e79")
    ax.set_xlabel("Gain")
    ax.set_title("Importance des variables")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _log_card_metrics(X_val: pd.DataFrame, y_val: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Scores sur les paiements par carte, si le sous-ensemble est assez grand."""
    if "payment_type" not in X_val.columns:
        return {}
    mask = _card_mask(X_val)
    if int(mask.sum()) < 2:
        return {}
    scores = _regression_scores(y_val.loc[mask], y_pred[mask.to_numpy()])
    return {
        "rmse_val_card": scores["rmse"],
        "mae_val_card": scores["mae"],
        "r2_val_card": scores["r2"],
    }


def train_lightgbm(
    params: Dict[str, Any],
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    run_name: str = "lgbm_run",
    experiment_name: Optional[str] = None,
    preprocessor: Optional[TipPreprocessor] = None,
) -> Dict[str, Any]:
    """Entraîne un modèle LightGBM et journalise l’expérience dans MLflow."""
    _set_mlflow_uri()
    exp_name = experiment_name or EXPERIMENT_NAME
    try:
        mlflow.set_experiment(exp_name)
    except FileNotFoundError as exc:
        logger.error("Impossible de créer ou d’accéder à l’expérience MLflow : %s", exc)
        raise

    X_train = X_train.astype(float)
    X_val = X_val.astype(float)
    train_params = {k: v for k, v in params.items() if k != "n_estimators"}
    n_boost = int(params.get("n_estimators", 500))

    run_id: Optional[str] = None
    rmse = mae = r2 = 0.0
    model: Optional[lgb.Booster] = None
    fi = pd.DataFrame()
    y_pred = np.array([])
    card_metrics: Dict[str, float] = {}

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
        scores = _regression_scores(y_val, y_pred)
        rmse, mae, r2 = scores["rmse"], scores["mae"], scores["r2"]
        card_metrics = _log_card_metrics(X_val, y_val, y_pred)

        mlflow.log_metrics(
            {
                "rmse_val": rmse,
                "mae_val": mae,
                "r2_val": r2,
                "best_iter": float(best_iter),
                **{k: v for k, v in card_metrics.items() if v == v},
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

        artifact_dir = Path(tempfile.mkdtemp(prefix="mlflow_fi_"))
        fi_path = artifact_dir / "feature_importance.csv"
        png_path = artifact_dir / "feature_importance.png"
        fi.to_csv(fi_path, index=False)
        _save_importance_png(fi, str(png_path))
        mlflow.log_artifact(str(fi_path))
        mlflow.log_artifact(str(png_path))
        if preprocessor is not None:
            pre_path = artifact_dir / "preprocessor.joblib"
            preprocessor.save(pre_path)
            mlflow.log_artifact(str(pre_path))
        for leftover in artifact_dir.iterdir():
            try:
                leftover.unlink()
            except OSError:
                pass
        try:
            artifact_dir.rmdir()
        except OSError:
            pass

        signature = infer_signature(X_train, y_pred)
        mlflow.lightgbm.log_model(
            model,
            name="model",
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
        "y_pred": y_pred,
        "card": card_metrics,
        "run_name": run_name,
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


def _predict_linear(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
) -> np.ndarray:
    """Moindres carrés. Les colonnes constantes sur l'entraînement sont retirées.

    Sur un court préfixe de janvier, le mois, le trimestre et l'heure de pointe
    ne varient pas : les laisser dans la matrice rend les coefficients instables.
    """
    keep = X_train.columns[X_train.std(ddof=0) > 1e-8]
    X_train_kept = X_train.loc[:, keep].to_numpy(dtype=float)
    X_val_kept = X_val.loc[:, keep].to_numpy(dtype=float)
    center = X_train_kept.mean(axis=0)
    scale = X_train_kept.std(axis=0)
    scale[scale < 1e-8] = 1.0
    train_scaled = (X_train_kept - center) / scale
    val_scaled = (X_val_kept - center) / scale
    design = np.column_stack([np.ones(len(train_scaled)), train_scaled])
    coefficients, *_ = np.linalg.lstsq(
        design, y_train.to_numpy(dtype=float), rcond=1e-8
    )
    return coefficients[0] + val_scaled @ coefficients[1:]


def run_baselines(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    experiment_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Journalise une moyenne d'entraînement puis une régression linéaire."""
    _set_mlflow_uri()
    exp_name = experiment_name or EXPERIMENT_NAME
    mlflow.set_experiment(exp_name)
    results: List[Dict[str, Any]] = []

    predictions = {
        "baseline_mean": np.full(len(y_val), float(y_train.mean())),
        "baseline_linear": _predict_linear(X_train, y_train, X_val),
    }

    for name, y_pred in predictions.items():
        scores = _regression_scores(y_val, y_pred)
        card = _log_card_metrics(X_val, y_val, y_pred)
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model", name)
            mlflow.log_metrics(
                {
                    "rmse_val": scores["rmse"],
                    "mae_val": scores["mae"],
                    "r2_val": scores["r2"],
                    **{k: v for k, v in card.items() if v == v},
                }
            )
        results.append({"name": name, **scores, "card": card, "y_pred": y_pred})
        logger.info(
            "%s | RMSE=%.4f | MAE=%.4f | R²=%.4f",
            name,
            scores["rmse"],
            scores["mae"],
            scores["r2"],
        )
    return results


def run_optuna_search(
    n_trials: int = 10,
    n_rows: int = 100_000,
    experiment_name: Optional[str] = None,
    split: str = "temporal",
) -> Dict[str, Any]:
    """Recherche d’hyperparamètres Optuna (objectif : minimiser le RMSE de validation)."""
    X_train, X_val, y_train, y_val, _ = prepare_dataset(
        n_rows=n_rows, use_real_data=True, split=split
    )
    preprocessor = getattr(prepare_dataset, "preprocessor", None)

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
            preprocessor=preprocessor,
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
    n_configs: int = 5,
    *,
    configs: Optional[List[Dict[str, Any]]] = None,
    experiment_name: Optional[str] = None,
    seed: int = RANDOM_STATE,
    split: str = "temporal",
    prepared: Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]] = None,
) -> List[Dict[str, Any]]:
    """Lance plusieurs runs MLflow (grille hyperparamètres ou liste fournie)."""
    if configs is None:
        grid = _generate_grid_configs(n_configs, seed=seed)
    else:
        grid = list(configs)

    if prepared is None:
        X_train, X_val, y_train, y_val, _ = prepare_dataset(
            n_rows=n_rows, use_real_data=True, split=split
        )
    else:
        X_train, X_val, y_train, y_val, _ = prepared
    preprocessor = getattr(prepare_dataset, "preprocessor", None)

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
            preprocessor=preprocessor,
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
    split: str = "temporal",
    prepared: Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]] = None,
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
        split=split,
        prepared=prepared,
    )


def _fit_booster(
    params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Tuple[lgb.Booster, np.ndarray]:
    """Entraîne un booster avec arrêt anticipé et renvoie les prédictions de validation."""
    train_params = {k: v for k, v in params.items() if k != "n_estimators"}
    n_boost = int(params.get("n_estimators", 300))
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    model = lgb.train(
        train_params,
        dtrain,
        num_boost_round=n_boost,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    best_iter = model.best_iteration or model.current_iteration()
    return model, model.predict(X_val, num_iteration=best_iter)


def train_quantile_interval(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    base_params: Dict[str, Any],
    experiment_name: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Deux LightGBM quantiles (10 % et 90 %), ajustés sur l'entraînement seul.

    La couverture est la part des courses de validation dont le pourboire
    observé tombe entre les deux prédictions. Elle n'est pas calculée
    sur les résidus qui ont servi à construire l'intervalle.
    """
    _set_mlflow_uri()
    mlflow.set_experiment(experiment_name or EXPERIMENT_NAME)
    fitted: Dict[str, lgb.Booster] = {}
    predictions: Dict[str, np.ndarray] = {}
    for alpha in (0.1, 0.9):
        params = dict(base_params)
        params.update({"objective": "quantile", "alpha": alpha, "metric": "quantile"})
        model, y_pred = _fit_booster(params, X_train, y_train, X_val, y_val)
        key = f"q{int(alpha * 100):02d}"
        fitted[key] = model
        predictions[key] = y_pred

    low = np.minimum(predictions["q10"], predictions["q90"])
    high = np.maximum(predictions["q10"], predictions["q90"])
    observed = y_val.to_numpy(dtype=float)
    covered = (observed >= low) & (observed <= high)
    coverage = float(np.mean(covered))
    card_mask = _card_mask(X_val).to_numpy()
    coverage_card = (
        float(np.mean(covered[card_mask])) if int(card_mask.sum()) >= 2 else float("nan")
    )
    width = float(np.mean(high - low))

    destination = output_dir or _REPORTS_DIR
    destination.mkdir(parents=True, exist_ok=True)
    low_path = destination / "quantile_10.txt"
    high_path = destination / "quantile_90.txt"
    fitted["q10"].save_model(str(low_path))
    fitted["q90"].save_model(str(high_path))

    with mlflow.start_run(run_name="quantile_10_90"):
        mlflow.log_params({"objective": "quantile", "alpha_low": 0.1, "alpha_high": 0.9})
        mlflow.log_metrics(
            {
                "interval_coverage": coverage,
                "interval_width": width,
                **(
                    {"interval_coverage_card": coverage_card}
                    if coverage_card == coverage_card
                    else {}
                ),
            }
        )
        mlflow.log_artifact(str(low_path))
        mlflow.log_artifact(str(high_path))

    logger.info(
        "Intervalle 10-90 | couverture validation=%.3f | largeur moyenne=%.3f",
        coverage,
        width,
    )
    return {
        "method": "lightgbm_quantile",
        "alphas": [0.1, 0.9],
        "coverage_val": coverage,
        "coverage_val_card": None if coverage_card != coverage_card else coverage_card,
        "mean_width": width,
    }


def compare_zone_embeddings(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    train_zones: pd.DataFrame,
    val_zones: pd.DataFrame,
    base_params: Dict[str, Any],
    experiment_name: Optional[str] = None,
    epochs: int = 10,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Remplace le target encoding par des embeddings de zones, sur le même découpage.

    L'embedding est appris sur les zones de départ de l'entraînement, puis
    appliqué aux zones de départ et d'arrivée. La validation ne sert pas
    à l'ajustement.
    """
    from src.features.encoding import TORCH_AVAILABLE, train_zone_embeddings

    if not TORCH_AVAILABLE:
        logger.warning("PyTorch absent : comparaison d'embeddings non exécutée.")
        return {"name": "embedding_lightgbm", "available": False}

    import torch

    torch.manual_seed(RANDOM_STATE)
    train_frame = train_zones.copy()
    train_frame["PULocationID"] = np.clip(
        train_frame["PULocationID"].fillna(0).to_numpy(dtype=int), 0, 265
    )
    train_frame["tip_pct"] = y_train.to_numpy(dtype=float)
    embedding_model = train_zone_embeddings(
        train_frame, zone_col="PULocationID", target="tip_pct", epochs=epochs
    )
    if embedding_model is None:
        return {"name": "embedding_lightgbm", "available": False}

    embedding_model.eval()
    table = embedding_model.get_embeddings()
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "zone_embeddings.npy", table)

    def _replace(features: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
        base = features.drop(
            columns=["PULocationID_target_enc", "DOLocationID_target_enc"],
            errors="ignore",
        ).reset_index(drop=True)
        blocks = [base]
        for column, prefix in (("PULocationID", "pu_emb"), ("DOLocationID", "do_emb")):
            index = np.clip(zones[column].fillna(0).to_numpy(dtype=int), 0, len(table) - 1)
            block = pd.DataFrame(
                table[index],
                columns=[f"{prefix}_{i}" for i in range(table.shape[1])],
            )
            blocks.append(block)
        return pd.concat(blocks, axis=1)

    encoded_train = _replace(X_train, train_zones)
    encoded_val = _replace(X_val, val_zones)
    result = train_lightgbm(
        dict(base_params),
        encoded_train,
        encoded_val,
        y_train,
        y_val,
        run_name="embedding_lightgbm",
        experiment_name=experiment_name,
    )
    result["name"] = "embedding_lightgbm"
    result["available"] = True
    logger.info(
        "Embeddings | RMSE=%.4f | RMSE carte=%.4f",
        result["rmse"],
        (result.get("card") or {}).get("rmse_val_card", float("nan")),
    )
    return result


def save_sorted_runs_figure(models: List[Dict[str, Any]], path: Path) -> None:
    """Enregistre le classement des runs par RMSE, à partir des scores mesurés."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(models, key=lambda item: float(item.get("rmse") or 1e9))
    cell_text = [
        [
            str(item.get("name", "")),
            f"{float(item['rmse']):.6f}",
            f"{float(item['mae']):.6f}",
            f"{float(item['r2']):.4f}",
        ]
        for item in rows
    ]
    fig, ax = plt.subplots(figsize=(8, 0.38 * len(rows) + 1.4))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=["run", "rmse_val", "mae_val", "r2_val"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)
    ax.set_title("Runs triés par RMSE de validation", pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _model_record(name: str, scores: Dict[str, float], card: Dict[str, float], params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Une ligne de reports/metrics.json."""
    record: Dict[str, Any] = {
        "name": name,
        "rmse": scores.get("rmse"),
        "mae": scores.get("mae"),
        "r2": scores.get("r2"),
        "rmse_card": card.get("rmse_val_card"),
        "mae_card": card.get("mae_val_card"),
        "r2_card": card.get("r2_val_card"),
    }
    if params is not None:
        record["params"] = {k: v for k, v in params.items() if k != "verbose"}
    return record


def write_reports(
    lgbm_results: List[Dict[str, Any]],
    baselines: List[Dict[str, Any]],
    interval: Optional[Dict[str, Any]] = None,
    extra_models: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    """Écrit metrics.json, l'importance des variables et le prétraitement."""
    import json
    from datetime import datetime, timezone

    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    meta = dict(getattr(prepare_dataset, "metadata", {}))
    preprocessor = getattr(prepare_dataset, "preprocessor", None)
    if preprocessor is not None:
        preprocessor.save(_REPORTS_DIR / "preprocessor.joblib")

    best = min(lgbm_results, key=lambda item: item["rmse"])
    booster = best.get("model")
    if booster is not None and hasattr(booster, "save_model"):
        booster.save_model(str(_REPORTS_DIR / "model.txt"))
    fi = best.get("fi")
    if isinstance(fi, pd.DataFrame) and not fi.empty:
        fi.to_csv(_REPORTS_DIR / "feature_importance.csv", index=False)
        _save_importance_png(fi, str(_REPORTS_DIR / "feature_importance.png"))

    models = [
        _model_record(item["name"], item, item.get("card") or {})
        for item in baselines
    ]
    for item in sorted(lgbm_results, key=lambda row: row["rmse"]):
        models.append(
            _model_record(
                item.get("run_name", "lightgbm"),
                item,
                item.get("card") or {},
                params=item.get("params"),
            )
        )
    for item in extra_models or []:
        if "rmse" not in item:
            continue
        models.append(
            _model_record(
                item.get("name") or item.get("run_name", "extra"),
                item,
                item.get("card") or {},
                params=item.get("params"),
            )
        )

    payload = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_rows_loaded": meta.get("n_rows_loaded"),
        "n_train": meta.get("n_train"),
        "n_val": meta.get("n_val"),
        "period_min": meta.get("period_min"),
        "period_max": meta.get("period_max"),
        "split": meta.get("split"),
        "use_real_data": meta.get("use_real_data"),
        "features": meta.get("features"),
        "payment_type_kept": True,
        "interval_coverage": None if interval is None else interval.get("coverage_val"),
        "interval": interval,
        "models": models,
    }
    save_sorted_runs_figure(models, _ROOT / "docs" / "img" / "mlflow_runs_rmse.png")

    def _clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: _clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_clean(v) for v in value]
        if isinstance(value, float) and value != value:
            return None
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        return value

    pd.DataFrame(models).sort_values("rmse").to_csv(
        _REPORTS_DIR / "mlflow_runs.csv", index=False
    )
    metrics_path = _REPORTS_DIR / "metrics.json"
    metrics_path.write_text(
        json.dumps(_clean(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Rapport écrit : %s", metrics_path)
    return metrics_path


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
        default=5,
        help="Mode full : nombre de runs (sous-ensemble aléatoire de la grille).",
    )
    parser.add_argument(
        "--split",
        choices=["temporal", "random"],
        default="temporal",
        help="Séparation validation : chronologique (défaut) ou aléatoire.",
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

    prepared = prepare_dataset(args.n_rows, use_real_data=True, split=args.split)
    X_train, X_val, y_train, y_val, _ = prepared
    baselines = run_baselines(
        X_train, X_val, y_train, y_val, experiment_name=args.experiment
    )

    if args.mode == "quick":
        print("=== MODE QUICK : 1 run MLflow ===")
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
            preprocessor=getattr(prepare_dataset, "preprocessor", None),
        )
        print(f"RMSE={result['rmse']:.4f} | R²={result['r2']:.4f}")
        print(f"Run ID : {result['run_id']}")
        write_reports([result], baselines)
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
            lgbm_results = run_5_experiments(
                args.n_rows,
                experiment_name=args.experiment,
                split=args.split,
                prepared=prepared,
            )
        else:
            print(
                f"=== MODE FULL : {args.n_experiments} runs (grille étendue) — « {exp} » ==="
            )
            lgbm_results = run_grid_experiments(
                args.n_rows,
                n_configs=args.n_experiments,
                experiment_name=args.experiment,
                seed=args.grid_seed,
                split=args.split,
                prepared=prepared,
            )
        best_params = min(lgbm_results, key=lambda item: item["rmse"])["params"]
        interval = train_quantile_interval(
            X_train,
            X_val,
            y_train,
            y_val,
            best_params,
            experiment_name=args.experiment,
            output_dir=_REPORTS_DIR,
        )
        embedding = compare_zone_embeddings(
            X_train,
            X_val,
            y_train,
            y_val,
            prepare_dataset.train_zones,  # type: ignore[attr-defined]
            prepare_dataset.val_zones,  # type: ignore[attr-defined]
            best_params,
            experiment_name=args.experiment,
            output_dir=_REPORTS_DIR,
        )
        write_reports(
            lgbm_results,
            baselines,
            interval=interval,
            extra_models=[embedding],
        )

    elif args.mode == "optuna":
        print(f"=== MODE OPTUNA : {args.n_trials} trials — expérience MLflow « {args.experiment or EXPERIMENT_NAME} » ===")
        run_optuna_search(
            args.n_trials,
            args.n_rows,
            experiment_name=args.experiment,
            split=args.split,
        )
