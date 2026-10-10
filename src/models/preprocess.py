"""Prétraitement ajustable et sérialisable pour la prédiction de tip_pct."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Union

import joblib
import numpy as np
import pandas as pd

from src.features.encoding import BayesianTargetEncoder, CrossFittedTargetEncoder
from src.features.temporal import TemporalFeaturizer

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TARGET = "tip_pct"

FEATURES: List[str] = [
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

_ZONE_COLUMNS = ["PULocationID", "DOLocationID"]
_COLUMN_ALIASES = {
    "pu_location_id": "PULocationID",
    "do_location_id": "DOLocationID",
    "pickup_datetime": "tpep_pickup_datetime",
}


class TipPreprocessor:
    """Nettoyage, variables temporelles et encodage des zones.

    L'encodeur de production est ajusté sur l'ensemble d'entraînement seulement.
    Les lignes d'entraînement passent par un encodage croisé ; la validation et
    les courses nouvelles utilisent l'encodeur ajusté sur tout l'entraînement.
    Une zone inconnue reçoit la moyenne globale de l'entraînement.
    """

    def __init__(
        self,
        smoothing: float = 10.0,
        n_splits: int = 5,
        add_holidays: bool = False,
    ) -> None:
        self.smoothing = smoothing
        self.n_splits = n_splits
        self.add_holidays = add_holidays
        self.encoder = BayesianTargetEncoder(
            columns=list(_ZONE_COLUMNS),
            target=TARGET,
            smoothing=smoothing,
        )
        self.feature_columns: List[str] = list(FEATURES)
        self._fitted = False
        self._logger = logging.getLogger(self.__class__.__name__)

    def fit(self, df: pd.DataFrame) -> "TipPreprocessor":
        """Ajuste l'encodeur de zones sur les courses d'entraînement."""
        prepared = self._prepare(df, with_target=True)
        self.encoder.fit(prepared)
        self._fitted = True
        self._logger.info("TipPreprocessor ajusté sur %s lignes", f"{len(prepared):,}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforme des courses déjà nettoyées ou brutes en variables du modèle.

        Si ``tip_amount`` est présent, les mêmes filtres que l'entraînement
        sont appliqués afin d'aligner les lignes avec ``target_series``.
        """
        if not self._fitted:
            raise RuntimeError("Appelez fit() avant transform().")
        prepared = self._prepare(df, with_target="tip_amount" in df.columns)
        encoded = self.encoder.transform(prepared)
        return self._select(encoded)

    def target_series(self, df: pd.DataFrame) -> pd.Series:
        """Cible tip_pct, filtrée comme les variables du modèle."""
        prepared = self._prepare(df, with_target=True)
        return prepared[TARGET].reset_index(drop=True).rename(TARGET)

    def zone_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identifiants de zones alignés sur transform() ou target_series()."""
        prepared = self._prepare(df, with_target="tip_amount" in df.columns)
        return prepared[_ZONE_COLUMNS].reset_index(drop=True)

    def fit_transform_train(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode l'entraînement par validation croisée, sans fuite de cible."""
        prepared = self._prepare(df, with_target=True)
        self.encoder.fit(prepared)
        n_splits = min(self.n_splits, len(prepared))
        if n_splits < 2:
            encoded = self.encoder.transform(prepared)
        else:
            crossed = CrossFittedTargetEncoder(
                columns=list(_ZONE_COLUMNS),
                target=TARGET,
                smoothing=self.smoothing,
                n_splits=n_splits,
                random_state=RANDOM_STATE,
            ).fit_transform(prepared)
            for col in _ZONE_COLUMNS:
                crossed[f"{col}_target_enc"] = crossed[f"{col}_cross_enc"]
            encoded = crossed
        self._fitted = True
        return self._select(encoded)

    def save(self, path: Union[str, Path]) -> None:
        """Sérialise le prétraitement ajusté."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)
        self._logger.info("Prétraitement enregistré : %s", destination)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TipPreprocessor":
        """Recharge un prétraitement ajusté."""
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError(f"Objet inattendu dans {path} : {type(loaded).__name__}")
        return loaded

    def _prepare(self, df: pd.DataFrame, with_target: bool) -> pd.DataFrame:
        frame = _rename_aliases(df)
        if "tpep_pickup_datetime" in frame.columns:
            frame["tpep_pickup_datetime"] = pd.to_datetime(frame["tpep_pickup_datetime"])
        frame = _filter_trips(frame, with_target=with_target)
        featurizer = TemporalFeaturizer(
            cyclic_encoding=True,
            add_holidays=self.add_holidays,
            add_rush_hours=True,
        )
        return featurizer.run(frame)

    def _select(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [col for col in self.feature_columns if col not in df.columns]
        if missing:
            raise KeyError(f"Colonnes absentes après prétraitement : {missing}")
        out = df[self.feature_columns].copy()
        for col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        return out.fillna(0.0)


def _rename_aliases(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    for alias, canonical in _COLUMN_ALIASES.items():
        if alias in frame.columns and canonical not in frame.columns:
            frame[canonical] = frame[alias]
    return frame


def _filter_trips(df: pd.DataFrame, with_target: bool) -> pd.DataFrame:
    """Applique les filtres métier. La cible n'est calculée qu'à l'entraînement."""
    frame = df.copy()
    required = ["fare_amount", "trip_distance", "passenger_count", "tpep_pickup_datetime"]
    required += _ZONE_COLUMNS
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise KeyError(f"Colonnes requises absentes : {missing}")

    frame = frame[frame["fare_amount"].between(2.5, 200)]
    frame = frame[frame["trip_distance"].between(0.1, 100)]
    frame = frame[frame["passenger_count"].between(1, 6)]
    if with_target:
        if "tip_amount" not in frame.columns:
            raise KeyError("tip_amount est requis pour ajuster le prétraitement.")
        frame = frame[frame["tip_amount"] >= 0]
        frame[TARGET] = (
            frame["tip_amount"] / frame["fare_amount"].replace(0, np.nan)
        ).clip(0, 1)
        frame = frame.dropna(subset=[TARGET])
        frame = frame[frame[TARGET] >= 0]
    if frame.empty:
        raise ValueError("Aucune course ne reste après le nettoyage.")
    return frame.reset_index(drop=True)


def temporal_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    time_column: str = "tpep_pickup_datetime",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sépare le jeu en conservant l'ordre chronologique."""
    ordered = df.sort_values(time_column).reset_index(drop=True)
    cut = int(len(ordered) * (1.0 - test_size))
    cut = max(1, min(cut, len(ordered) - 1))
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()
