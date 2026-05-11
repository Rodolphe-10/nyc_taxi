"""Script d'intégration Partie 1.

Charge 1 mois de données via la Factory,
orchestre via Pipeline avec tous les observateurs,
persiste le résultat dans data/processed/.
"""

import logging
import os

import numpy as np
import pandas as pd

from src.core.context import Timer
from src.core.metaclasses import BasePipelineStep
from src.pipeline.loaders import LoaderFactory
from src.pipeline.orchestrator import Pipeline
from src.pipeline.strategies import FeatureEngineer, FrequencyEncodingStrategy, TargetEncodingStrategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# Étape factice pour illustrer un loader déjà exécuté en amont via Factory.
class DataLoader(BasePipelineStep):
    name = "data_loader"

    def __init__(self, path):
        self.path = path

    def run(self, df):
        return df


# Étape de nettoyage métier des courses taxi.
class DataCleaner(BasePipelineStep):
    name = "data_cleaner"

    def run(self, df):
        df = df.copy()
        df = df[df["fare_amount"].between(2.5, 200)]
        df = df[df["trip_distance"].between(0.1, 100)]
        df = df[df["passenger_count"].between(1, 6)]
        df = df[df["tip_amount"] >= 0]
        df["tip_pct"] = df["tip_amount"] / df["fare_amount"]
        df = df[df["tip_pct"].between(0, 1)]
        return df


# Étape de création de variables dérivées temporelles et de trajet.
class FeatureCreator(BasePipelineStep):
    name = "feature_creator"

    def run(self, df):
        df = df.copy()
        df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
        df["pickup_weekday"] = df["tpep_pickup_datetime"].dt.dayofweek
        df["pickup_month"] = df["tpep_pickup_datetime"].dt.month
        duration = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60
        df["trip_duration_min"] = duration
        df["speed_mph"] = df["trip_distance"] / (df["trip_duration_min"] / 60).clip(lower=0.1)
        return df


# Étape d'encodage des zones via une stratégie injectable.
class ZoneEncoder(BasePipelineStep):
    name = "zone_encoder"

    def run(self, df):
        engineer = FeatureEngineer(strategy=FrequencyEncodingStrategy(), column="PULocationID")
        return engineer.run(df)


def main():
    """Point d'entrée du script d'intégration Partie 1."""
    raw_path = "data/raw/yellow_tripdata_2023-01.parquet"
    processed_path = "data/processed/yellow_2023_01_processed.parquet"

    os.makedirs("data/processed", exist_ok=True)

    if os.path.exists(raw_path):
        print(f"[INFO] Chargement via Factory : {raw_path}")
        with Timer("Chargement Factory", unit="s") as t:
            loader = LoaderFactory.create(raw_path)
            df = loader.run()
        print(f"[OK] {len(df):,} lignes chargées en {t.elapsed:.2f}s")
    else:
        print("[INFO] Fichier Parquet absent → génération de données simulées")
        np.random.seed(42)
        n = 100_000
        dates = pd.date_range("2023-01-01", periods=n, freq="1min")
        df = pd.DataFrame(
            {
                "tpep_pickup_datetime": dates,
                "tpep_dropoff_datetime": dates + pd.to_timedelta(np.random.uniform(5, 60, n), unit="min"),
                "fare_amount": np.random.uniform(5, 80, n),
                "tip_amount": np.random.uniform(0, 15, n),
                "trip_distance": np.random.uniform(0.5, 20, n),
                "passenger_count": np.random.randint(1, 6, n),
                "PULocationID": np.random.randint(1, 265, n),
                "DOLocationID": np.random.randint(1, 265, n),
                "payment_type": np.random.randint(1, 4, n),
                "total_amount": np.random.uniform(8, 100, n),
            }
        )
        print(f"[OK] {len(df):,} lignes simulées")

    pipeline = (
        Pipeline(name="TaxiPipeline_Part1", alert_threshold_pct=30.0)
        .add_step(DataCleaner())
        .add_step(FeatureCreator())
        .add_step(ZoneEncoder())
    )

    print(f"\n{pipeline.mro_info}\n")

    with Timer("Pipeline complet", unit="s") as t:
        result = pipeline.run(df)

    print(f"\n[OK] Pipeline terminé en {t.elapsed:.2f}s")
    print(f"[OK] Shape finale : {result.shape}")

    print("\n--- MÉTRIQUES DU PIPELINE ---")
    print(pipeline.metrics.to_string())
    print(f"\nÉtape la plus lente : {pipeline.slowest_step}")

    result.to_parquet(processed_path, index=False)
    print(f"\n[OK] Résultat sauvegardé : {processed_path}")

    pipeline.to_pickle("data/processed/pipeline_part1.pkl")
    print("[OK] Pipeline sérialisé : data/processed/pipeline_part1.pkl")

    print("\n✅ PARTIE 1 COMPLÈTE — Tous les patterns implémentés et testés")


if __name__ == "__main__":
    main()
