"""Orchestrateur principal combinant mixins et composition d'étapes."""

import logging
import os
import pickle
import time
from typing import List, Optional

import pandas as pd

from src.core.context import MemoryGuard, PipelineContext, Timer
from src.core.metaclasses import BasePipelineStep
from src.pipeline.observers import AlertObserver, LoggingObserver, MetricsObserver, PipelineSubject


# Mixin de logging : rend n'importe quelle classe "loggable" sans changer son coeur métier.
class LoggableMixin:
    """Ajoute des méthodes de logs structurés."""

    def _get_logger(self) -> logging.Logger:
        return logging.getLogger(self.__class__.__name__)

    def log_info(self, message: str) -> None:
        self._get_logger().info(f"[{self.__class__.__name__}] {message}")

    def log_warning(self, message: str) -> None:
        self._get_logger().warning(f"[{self.__class__.__name__}] ⚠ {message}")

    def log_error(self, message: str) -> None:
        self._get_logger().error(f"[{self.__class__.__name__}] ✗ {message}")

    def log_success(self, message: str) -> None:
        self._get_logger().info(f"[{self.__class__.__name__}] ✓ {message}")


# Mixin de persistance : sérialise/désérialise l'état du pipeline en pickle.
class SerializableMixin:
    """Ajoute les capacités de sérialisation."""

    def to_pickle(self, path: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "wb") as file:
            pickle.dump(self, file)
        logging.getLogger(self.__class__.__name__).info("Sérialisé dans '%s'", path)

    @classmethod
    def from_pickle(cls, path: str) -> "SerializableMixin":
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fichier introuvable : {path}")
        with open(path, "rb") as file:
            obj = pickle.load(file)
        if not isinstance(obj, cls):
            raise TypeError(f"L'objet désérialisé n'est pas une instance de {cls.__name__}")
        return obj

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


# Mixin de validation : protège le pipeline contre des entrées/sorties incohérentes.
class ValidatableMixin:
    """Ajoute la validation des DataFrames d'entrée et de sortie."""

    REQUIRED_COLUMNS: List[str] = []
    MIN_ROWS: int = 1

    def validate(self, df: pd.DataFrame) -> bool:
        if df is None or not isinstance(df, pd.DataFrame):
            raise TypeError("L'entrée doit être un pd.DataFrame")
        if len(df) < self.MIN_ROWS:
            raise ValueError(f"DataFrame trop petit : {len(df)} lignes < minimum {self.MIN_ROWS}")
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"Colonne requise absente : '{col}'")
        return True

    def validate_output(self, df_in: pd.DataFrame, df_out: pd.DataFrame) -> None:
        if df_out.empty:
            raise ValueError("La sortie d'étape est vide")
        pct_lost = (len(df_in) - len(df_out)) / len(df_in) * 100
        if pct_lost > 50:
            logging.getLogger(self.__class__.__name__).warning(
                "Attention : %.1f%% des lignes supprimées dans cette étape",
                pct_lost,
            )


# Le MRO est important ici :
# Pipeline(LoggableMixin, SerializableMixin, ValidatableMixin) garantit
# l'ordre de résolution des méthodes selon la hiérarchie Python.
class Pipeline(LoggableMixin, SerializableMixin, ValidatableMixin):
    """Chef d'orchestre qui compose une suite d'étapes BasePipelineStep."""

    REQUIRED_COLUMNS: List[str] = []
    MIN_ROWS: int = 1

    def __init__(
        self,
        steps: Optional[List[BasePipelineStep]] = None,
        name: str = "Pipeline",
        memory_threshold_mb: float = 8000.0,
        alert_threshold_pct: float = 30.0,
    ):
        self.name = name
        self.steps = steps or []
        self.memory_threshold_mb = memory_threshold_mb
        self._results = []
        self._is_fitted = False

        self._subject = PipelineSubject()
        self._logging_obs = LoggingObserver()
        self._metrics_obs = MetricsObserver()
        self._alert_obs = AlertObserver(threshold_pct=alert_threshold_pct)

        self._subject.attach(self._logging_obs)
        self._subject.attach(self._metrics_obs)
        self._subject.attach(self._alert_obs)

        BasePipelineStep.set_subject(self._subject)

        self.log_info(f"Pipeline '{self.name}' initialisé avec {len(self.steps)} étapes")

    def add_step(self, step: BasePipelineStep) -> "Pipeline":
        if not isinstance(step, BasePipelineStep):
            raise TypeError("Une étape doit être une instance de BasePipelineStep")
        self.steps.append(step)
        self.log_info(f"Étape ajoutée : '{step.name}'")
        return self

    def remove_step(self, step_name: str) -> "Pipeline":
        initial_len = len(self.steps)
        self.steps = [step for step in self.steps if step.name != step_name]
        if len(self.steps) == initial_len:
            self.log_warning(f"Étape absente : '{step_name}'")
        else:
            self.log_info(f"Étape supprimée : '{step_name}'")
        return self

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Exécute toutes les étapes en séquence avec monitoring complet."""
        self.validate(df)

        self.log_info(
            f"Démarrage — {len(self.steps)} étapes | {len(df):,} lignes | {df.shape[1]} colonnes"
        )
        self._subject.notify_start(len(self.steps))

        t_global = time.perf_counter()
        result = df.copy()

        with MemoryGuard(self.memory_threshold_mb, self.name):
            for i, step in enumerate(self.steps):
                self.log_info(f"Étape {i + 1}/{len(self.steps)} : '{step.name}'")
                with Timer(step.name, unit="ms") as timer:
                    result = step.execute(result)
                self.validate_output(df, result)
                self.log_success(
                    f"'{step.name}' terminé en {timer.elapsed:.1f}ms | {len(result):,} lignes restantes"
                )

        total_ms = (time.perf_counter() - t_global) * 1000
        self._subject.notify_end(total_ms)
        self._is_fitted = True

        self._results.append(
            {
                "n_steps": len(self.steps),
                "input_rows": len(df),
                "output_rows": len(result),
                "total_ms": total_ms,
            }
        )

        self.log_success(f"Pipeline terminé en {total_ms:.1f}ms | {len(result):,} lignes en sortie")
        return result

    @property
    def metrics(self) -> pd.DataFrame:
        return self._metrics_obs.summary()

    @property
    def alerts(self) -> list:
        return self._alert_obs.alerts()

    @property
    def slowest_step(self) -> str:
        return self._metrics_obs.slowest_step()

    @property
    def mro_info(self) -> str:
        return "MRO de Pipeline : " + " → ".join(cls.__name__ for cls in Pipeline.__mro__)

    def __repr__(self) -> str:
        return f"<Pipeline name='{self.name}' steps={[s.name for s in self.steps]} fitted={self._is_fitted}>"

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)


if __name__ == "__main__":
    import numpy as np

    # Étape factice de chargement.
    class LoadStep(BasePipelineStep):
        name = "chargement"

        def run(self, df):
            self_log = logging.getLogger("LoadStep")
            return df

    # Étape factice de nettoyage.
    class CleanStep(BasePipelineStep):
        name = "nettoyage"

        def run(self, df):
            return df[df["fare_amount"] > 0]

    # Étape factice de feature engineering.
    class FeatureStep(BasePipelineStep):
        name = "feature_engineering"

        def run(self, df):
            df = df.copy()
            df["trip_duration"] = np.random.uniform(5, 60, len(df))
            df["speed_mph"] = df["trip_distance"] / (df["trip_duration"] / 60)
            return df

    # Étape factice de filtrage d'aberrants.
    class FilterStep(BasePipelineStep):
        name = "filtrage_aberrants"

        def run(self, df):
            return df.sample(frac=0.8, random_state=42)

    np.random.seed(42)
    n = 50_000
    df = pd.DataFrame(
        {
            "fare_amount": np.random.uniform(-5, 50, n),
            "trip_distance": np.random.uniform(0.1, 20, n),
            "tip_amount": np.random.uniform(0, 10, n),
            "passenger_count": np.random.randint(1, 6, n),
        }
    )

    pipeline = (
        Pipeline(name="TaxiPipeline_v1")
        .add_step(LoadStep())
        .add_step(CleanStep())
        .add_step(FeatureStep())
        .add_step(FilterStep())
    )

    print(repr(pipeline))
    print(f"\nNombre d'étapes : {len(pipeline)}")
    print(f"\n{pipeline.mro_info}")

    result = pipeline.run(df)

    print("\n--- MÉTRIQUES ---")
    print(pipeline.metrics.to_string())
    print(f"\nÉtape la plus lente : {pipeline.slowest_step}")
    print(f"Alertes : {len(pipeline.alerts)}")

    pipeline.to_pickle("data/processed/pipeline_v1.pkl")
    print("\nPipeline sérialisé.")

    pipeline2 = Pipeline.from_pickle("data/processed/pipeline_v1.pkl")
    print(f"Pipeline restauré : {repr(pipeline2)}")
