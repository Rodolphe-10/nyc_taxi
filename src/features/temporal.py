"""Feature engineering temporel pour NYC Taxi.

Ce module ajoute des variables calendrier, des indicateurs métier (rush hour),
et un encodage cyclique sin/cos pour les variables périodiques.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.core.context import Timer
from src.core.metaclasses import BasePipelineStep
from src.optim.benchmarks import generate_taxi_df

try:
    import holidays

    HOLIDAYS_AVAILABLE = True
except ImportError:
    HOLIDAYS_AVAILABLE = False
    logging.warning("holidays absent — pip install holidays")

try:
    from statsmodels.tsa.seasonal import STL

    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logging.warning("statsmodels absent — pip install statsmodels")


def encode_cyclic(series: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
    """
    Encodage cyclique pour variables périodiques.

    POURQUOI sin/cos plutôt que valeur brute ?
    ─────────────────────────────────────────────
    Problème avec valeur brute :
      heure=0 et heure=23 semblent distants de 23 unités pour le modèle.
      Mais minuit et 23h sont en réalité adjacents !

    Solution sin/cos :
      encode_cyclic(23, period=24) ≈ encode_cyclic(0, period=24)
      → Le modèle comprend la continuité circulaire.

    Applications :
      Heure  : period=24
      Jour   : period=7
      Mois   : period=12
      Minute : period=60
    """
    # Projection angle -> cercle unité : theta = 2*pi*x/period.
    sin_enc = np.sin(2 * np.pi * series / period)
    cos_enc = np.cos(2 * np.pi * series / period)
    return (
        pd.Series(sin_enc, index=series.index, dtype="float32"),
        pd.Series(cos_enc, index=series.index, dtype="float32"),
    )


class TemporalFeaturizer(BasePipelineStep):
    """Étape pipeline qui enrichit les données avec des features temporelles."""

    name = "temporal_featurizer"
    version = "2.0.0"

    def __init__(
        self,
        cyclic_encoding: bool = True,
        add_holidays: bool = True,
        add_rush_hours: bool = True,
        country: str = "US",
        state: str = "NY",
    ) -> None:
        self.cyclic_encoding = cyclic_encoding
        self.add_holidays = add_holidays
        self.add_rush_hours = add_rush_hours
        self.country = country
        self.state = state
        self._logger = logging.getLogger(self.__class__.__name__)
        self._us_holidays = None

        if add_holidays and HOLIDAYS_AVAILABLE:
            self._us_holidays = holidays.country_holidays(country, subdiv=state)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applique toutes les transformations temporelles."""
        df = df.copy()
        with Timer("TemporalFeaturizer", unit="ms") as t:
            df = self._add_calendar_features(df)
            if self.cyclic_encoding:
                df = self._add_cyclic_features(df)
            if self.add_rush_hours:
                df = self._add_proxy_indicators(df)
            if self.add_holidays and HOLIDAYS_AVAILABLE:
                df = self._add_holiday_features(df)

        self._logger.info(
            "TemporalFeaturizer terminé en %.1fms | %s colonnes",
            t.elapsed,
            df.shape[1],
        )
        return df

    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les features calendaires de base."""
        dt = df["tpep_pickup_datetime"]
        df["pickup_hour"] = dt.dt.hour.astype("int8")
        df["pickup_weekday"] = dt.dt.dayofweek.astype("int8")
        df["pickup_day"] = dt.dt.day.astype("int8")
        df["pickup_month"] = dt.dt.month.astype("int8")
        df["pickup_quarter"] = dt.dt.quarter.astype("int8")
        df["pickup_week_iso"] = dt.dt.isocalendar().week.astype("int8")
        df["is_weekend"] = (df["pickup_weekday"] >= 5).astype("int8")
        df["pickup_year"] = dt.dt.year.astype("int16")
        return df

    def _add_cyclic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode les variables cycliques avec sin/cos."""
        sin_h, cos_h = encode_cyclic(df["pickup_hour"], 24)
        df["pickup_hour_sin"] = sin_h.astype("float32")
        df["pickup_hour_cos"] = cos_h.astype("float32")

        sin_d, cos_d = encode_cyclic(df["pickup_weekday"], 7)
        df["pickup_weekday_sin"] = sin_d.astype("float32")
        df["pickup_weekday_cos"] = cos_d.astype("float32")

        sin_m, cos_m = encode_cyclic(df["pickup_month"], 12)
        df["pickup_month_sin"] = sin_m.astype("float32")
        df["pickup_month_cos"] = cos_m.astype("float32")
        return df

    def _add_proxy_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les indicateurs proxy métier."""
        hour = df["pickup_hour"]
        weekday = df["pickup_weekday"]

        is_morning_rush = (hour.between(7, 9)) & (weekday < 5)
        is_evening_rush = (hour.between(17, 19)) & (weekday < 5)
        df["is_rush_hour"] = (is_morning_rush | is_evening_rush).astype("int8")

        df["is_late_night"] = hour.between(0, 5).astype("int8")

        is_lunch = hour.between(11, 13)
        is_dinner = hour.between(18, 20)
        df["is_meal_time"] = (is_lunch | is_dinner).astype("int8")

        df["is_night"] = ((hour >= 22) | (hour <= 4)).astype("int8")
        return df

    def _add_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les indicateurs de jours fériés US/NY."""
        if not HOLIDAYS_AVAILABLE or self._us_holidays is None:
            return df

        import datetime

        dates = df["tpep_pickup_datetime"].dt.date
        df["is_holiday"] = dates.apply(lambda d: int(d in self._us_holidays)).astype("int8")
        df["is_holiday_eve"] = dates.apply(
            lambda d: int((d + datetime.timedelta(days=1)) in self._us_holidays)
        ).astype("int8")
        return df


def decompose_seasonality(
    df: pd.DataFrame,
    freq: str = "h",
    period: int = 24,
) -> Optional[pd.DataFrame]:
    """Décompose la saisonnalité (STL) du nombre de courses."""
    logger = logging.getLogger(__name__)
    if not STATSMODELS_AVAILABLE:
        logger.warning("statsmodels absent — décomposition ignorée")
        return None

    ts = (
        df.set_index("tpep_pickup_datetime")
        .resample(freq)
        .size()
        .rename("trip_count")
    )
    ts = ts.asfreq(freq, fill_value=0)

    stl = STL(ts, period=period, robust=True)
    result = stl.fit()
    decomp = pd.DataFrame(
        {
            "observed": result.observed,
            "trend": result.trend,
            "seasonal": result.seasonal,
            "residual": result.resid,
        }
    )
    logger.info("STL décomposé : %s périodes | period=%s", len(decomp), period)
    return decomp


if __name__ == "__main__":
    print("=== EXERCICE 3.5 — FEATURE ENGINEERING TEMPOREL ===\n")

    df_raw = generate_taxi_df(n_rows=100_000)

    print("--- A. TemporalFeaturizer ---")
    featurizer = TemporalFeaturizer(
        cyclic_encoding=True,
        add_holidays=True,
        add_rush_hours=True,
    )
    with Timer("TemporalFeaturizer.execute()", unit="ms"):
        df_features = featurizer.execute(df_raw.copy())
    print(f"Shape : {df_raw.shape} → {df_features.shape}")
    temporal_cols = [c for c in df_features.columns if c not in df_raw.columns]
    print(f"Nouvelles features ({len(temporal_cols)}) :")
    for col in temporal_cols:
        print(f"  • {col} : {df_features[col].dtype}")

    print("\n--- B. Encodage Cyclique ---")
    hours = pd.Series(range(24))
    sin_h, cos_h = encode_cyclic(hours, 24)
    print("Encodage heure (sin, cos) :")
    for h, s, c in zip([0, 6, 12, 18, 23], sin_h[[0, 6, 12, 18, 23]], cos_h[[0, 6, 12, 18, 23]]):
        print(f"  h={h:2d} → sin={s:+.3f}, cos={c:+.3f}")

    print("\n  Distance 0h ↔ 23h (brut)  : 23 unités")
    sin0, cos0 = encode_cyclic(pd.Series([0]), 24)
    sin23, cos23 = encode_cyclic(pd.Series([23]), 24)
    dist = np.sqrt((sin0.values[0] - sin23.values[0]) ** 2 + (cos0.values[0] - cos23.values[0]) ** 2)
    print(f"  Distance 0h ↔ 23h (sin/cos): {dist:.3f} unités")
    sin6, cos6 = encode_cyclic(pd.Series([6]), 24)
    dist6 = np.sqrt((sin0.values[0] - sin6.values[0]) ** 2 + (cos0.values[0] - cos6.values[0]) ** 2)
    print(f"  Distance 0h ↔  6h (sin/cos): {dist6:.3f} unités")

    print("\n--- C. Indicateurs Rush Hour ---")
    rush_stats = (
        df_features.groupby("is_rush_hour")["tip_pct"].agg(["mean", "count"])
        if "tip_pct" in df_features.columns
        else None
    )
    if rush_stats is not None:
        print(rush_stats.to_string())

    print("\n--- D. Décomposition STL ---")
    if STATSMODELS_AVAILABLE:
        decomp = decompose_seasonality(df_raw, freq="h", period=24)
        if decomp is not None:
            print(f"Composantes STL : {decomp.shape}")
            print(decomp.describe().to_string())
    else:
        print("[SKIP] statsmodels absent")
