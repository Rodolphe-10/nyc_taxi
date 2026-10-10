"""Tests du prétraitement et d'un entraînement court sur données simulées."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.preprocess import TipPreprocessor, temporal_split
from src.models.train import (
    compare_zone_embeddings,
    prepare_dataset,
    train_lightgbm,
    train_quantile_interval,
)
from src.optim.benchmarks import generate_taxi_df


def test_temporal_split_is_ordered():
    df = generate_taxi_df(100, seed=42)
    train_df, val_df = temporal_split(df, test_size=0.2)
    assert train_df["tpep_pickup_datetime"].max() <= val_df["tpep_pickup_datetime"].min()
    assert len(train_df) == 80
    assert len(val_df) == 20


def test_encoder_is_fit_on_train_only():
    df = generate_taxi_df(300, seed=42)
    train_df, val_df = temporal_split(df, test_size=0.25)
    val_df = val_df.copy()
    val_df["tip_amount"] = 0.0

    preprocessor = TipPreprocessor(n_splits=3)
    preprocessor.fit(train_df)
    train_only = preprocessor.encoder._global_means["PULocationID"]

    both = TipPreprocessor(n_splits=3)
    both.fit(pd.concat([train_df, val_df], ignore_index=True))
    assert train_only != both.encoder._global_means["PULocationID"]
    assert abs(train_only - preprocessor.target_series(train_df).mean()) < 1e-6


def test_unknown_zone_uses_global_mean():
    df = generate_taxi_df(200, seed=7)
    df["PULocationID"] = 10
    df["DOLocationID"] = 12
    preprocessor = TipPreprocessor(n_splits=2)
    preprocessor.fit(df)
    expected = preprocessor.encoder._global_means["PULocationID"]

    raw = df.iloc[[0]].drop(columns=["tip_amount"])
    raw["PULocationID"] = 300
    raw["DOLocationID"] = 301
    transformed = preprocessor.transform(raw)
    assert transformed.loc[0, "PULocationID_target_enc"] == expected
    assert transformed.loc[0, "DOLocationID_target_enc"] == expected


def test_preprocessor_roundtrip(tmp_path):
    df = generate_taxi_df(120, seed=3)
    preprocessor = TipPreprocessor(n_splits=2)
    preprocessor.fit(df)
    path = tmp_path / "preprocessor.joblib"
    preprocessor.save(path)
    loaded = TipPreprocessor.load(path)
    original = preprocessor.transform(df.drop(columns=["tip_amount"]))
    restored = loaded.transform(df.drop(columns=["tip_amount"]))
    pd.testing.assert_frame_equal(original, restored)


def test_prepare_dataset_temporal_and_quick_train(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())
    X_train, X_val, y_train, y_val, features = prepare_dataset(
        n_rows=400,
        use_real_data=False,
        split="temporal",
    )
    assert len(X_train) > len(X_val)
    assert "payment_type" in features
    assert len(X_train) == len(y_train)
    result = train_lightgbm(
        {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.1,
            "n_estimators": 15,
            "max_depth": 3,
            "num_leaves": 7,
            "verbosity": -1,
            "boosting_type": "gbdt",
        },
        X_train,
        X_val,
        y_train,
        y_val,
        run_name="test_quick",
        experiment_name="test_tip_preprocessor",
        preprocessor=prepare_dataset.preprocessor,
    )
    assert result["rmse"] >= 0
    assert np.isfinite(result["rmse"])
    assert len(prepare_dataset.train_zones) == len(X_train)
    assert len(prepare_dataset.val_zones) == len(X_val)


def test_quantile_coverage_is_measured_on_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())
    X_train, X_val, y_train, y_val, _ = prepare_dataset(
        n_rows=400,
        use_real_data=False,
        split="temporal",
    )
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.1,
        "n_estimators": 20,
        "max_depth": 3,
        "num_leaves": 7,
        "verbosity": -1,
        "boosting_type": "gbdt",
    }
    interval = train_quantile_interval(
        X_train,
        X_val,
        y_train,
        y_val,
        params,
        experiment_name="test_quantile",
        output_dir=tmp_path,
    )
    assert 0.0 <= interval["coverage_val"] <= 1.0
    assert interval["mean_width"] >= 0.0
    assert (tmp_path / "quantile_10.txt").is_file()
    assert (tmp_path / "quantile_90.txt").is_file()

    embedding = compare_zone_embeddings(
        X_train,
        X_val,
        y_train,
        y_val,
        prepare_dataset.train_zones,
        prepare_dataset.val_zones,
        params,
        experiment_name="test_embeddings",
        epochs=2,
    )
    assert embedding["available"] is True
    assert np.isfinite(embedding["rmse"])
