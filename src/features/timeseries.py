"""Features de séries temporelles agrégées par zone et par heure (NYC Taxi)."""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from src.core.context import Timer
from src.optim.benchmarks import generate_taxi_df

try:
    from statsmodels.tsa.seasonal import STL  # noqa: F401 — STL disponible si besoin
    from statsmodels.tsa.stattools import adfuller

    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logging.warning("statsmodels absent — tests ADF indisponibles")

logger = logging.getLogger(__name__)


def aggregate_hourly(df: pd.DataFrame, zone_col: str = "PULocationID") -> pd.DataFrame:
    """Agrège le dataset à la maille horaire par zone."""
    df = df.copy()
    df["pickup_hour_ts"] = df["tpep_pickup_datetime"].dt.floor("h")

    hourly = (
        df.groupby([zone_col, "pickup_hour_ts"])
        .agg(
            trip_count=("fare_amount", "count"),
            tip_mean=("tip_amount", "mean"),
            fare_mean=("fare_amount", "mean"),
            distance_mean=("trip_distance", "mean"),
        )
        .reset_index()
    )
    hourly = hourly.sort_values([zone_col, "pickup_hour_ts"])
    logger.info(
        "Agrégé : %s lignes horaires | %s zones",
        f"{len(hourly):,}",
        hourly[zone_col].nunique(),
    )
    return hourly


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "trip_count",
    group_col: str = "PULocationID",
    lags: List[int] = None,
) -> pd.DataFrame:
    """Lags par zone : décalage temporel sans regarder le futur (shift positif = passé).

    - lag_1h   : valeur de l'heure précédente (inertie locale).
    - lag_24h  : même heure la veille (saisonnalité journalière, rush récurrents).
    - lag_168h : même heure une semaine avant (saisonnalité hebdomadaire).
    """
    if lags is None:
        lags = [1, 24, 168]

    df = df.copy().sort_values([group_col, "pickup_hour_ts"])
    for lag in lags:
        # shift(lag) : à l'instant t on n'utilise que les valeurs d'au moins lag pas en arrière
        df[f"{target_col}_lag_{lag}h"] = df.groupby(group_col)[target_col].shift(lag)

    logger.info("Lag features : %s", [f"lag_{l}h" for l in lags])
    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "trip_count",
    group_col: str = "PULocationID",
    windows: List[int] = None,
) -> pd.DataFrame:
    """Rolling stats par zone avec shift(1) avant fenêtre → pas de fuite de la valeur courante."""
    if windows is None:
        windows = [3, 24, 168]

    df = df.copy().sort_values([group_col, "pickup_hour_ts"])
    grp = df.groupby(group_col)[target_col]

    for w in windows:
        df[f"{target_col}_roll_mean_{w}h"] = grp.transform(
            lambda x, ww=w: x.shift(1).rolling(ww, min_periods=1).mean()
        )
        df[f"{target_col}_roll_std_{w}h"] = grp.transform(
            lambda x, ww=w: x.shift(1).rolling(ww, min_periods=2).std()
        )
        df[f"{target_col}_roll_min_{w}h"] = grp.transform(
            lambda x, ww=w: x.shift(1).rolling(ww, min_periods=1).min()
        )
        df[f"{target_col}_roll_max_{w}h"] = grp.transform(
            lambda x, ww=w: x.shift(1).rolling(ww, min_periods=1).max()
        )

    logger.info("Rolling features : windows=%s", windows)
    return df


def add_differencing(
    df: pd.DataFrame,
    target_col: str = "trip_count",
    group_col: str = "PULocationID",
) -> pd.DataFrame:
    """Différenciation d'ordre 1 et saisonnière 24h par zone.

    - diff_1  : enlève une tendance locale / marche lente.
    - diff_24 : enlève une composante journalière répétée (même heure veille).
    Si la série devient stationnaire après diff, modèles type ARIMA/SARIMA sont plus adaptés.
    """
    df = df.copy().sort_values([group_col, "pickup_hour_ts"])
    df[f"{target_col}_diff_1"] = df.groupby(group_col)[target_col].diff(1)
    df[f"{target_col}_diff_24"] = df.groupby(group_col)[target_col].diff(24)
    logger.info("Différenciation d'ordre 1 et saisonnière (24h) calculée")
    return df


def add_fourier_features(
    df: pd.DataFrame,
    datetime_col: str = "pickup_hour_ts",
    n_harmonics: int = 5,
    period: int = 24,
) -> pd.DataFrame:
    """Harmoniques de Fourier sur l'heure dans la période (saisonnalité intra-journalière).

    k=1 : fondamentale (un cycle par jour) ; k>1 : harmoniques (ex. double rush).
    Avantage vs one-hot heure : nombre fixe de colonnes (2 * n_harmonics), continu et lisse.
    """
    df = df.copy()
    df["hour_in_period"] = df[datetime_col].dt.hour.astype(np.int16)

    for k in range(1, n_harmonics + 1):
        df[f"fourier_sin_{k}"] = (
            np.sin(2 * np.pi * k * df["hour_in_period"] / period).astype("float32")
        )
        df[f"fourier_cos_{k}"] = (
            np.cos(2 * np.pi * k * df["hour_in_period"] / period).astype("float32")
        )

    logger.info("Fourier features : %s harmoniques", n_harmonics)
    return df


def adf_test(series: pd.Series, label: str = "") -> Dict:
    """Test ADF (Augmented Dickey-Fuller) pour la stationnarité."""
    if not STATSMODELS_AVAILABLE:
        logger.warning("statsmodels absent — ADF ignoré")
        return {"available": False}

    series_clean = series.dropna()
    if len(series_clean) < 20:
        return {"error": "Série trop courte"}

    result = adfuller(series_clean, autolag="AIC")
    adf_stat = result[0]
    p_value = result[1]
    n_lags = result[2]
    n_obs = result[3]
    crit_vals = result[4]

    is_stationary = bool(p_value < 0.05)

    print("┌──────────────────────────────────────────────────┐")
    print(f"│  ADF Test : {label:<36}│")
    print("├──────────────────────────────────────────────────┤")
    print(f"│  Statistique ADF : {adf_stat:<28.4f}│")
    print(f"│  p-value         : {p_value:<28.4f}│")
    print(f"│  Lags utilisés   : {n_lags:<28}│")
    print("│  Seuils critiques:                               │")
    print(f"│    1%  : {crit_vals['1%']:<34.4f}│")
    print(f"│    5%  : {crit_vals['5%']:<34.4f}│")
    print(f"│    10% : {crit_vals['10%']:<33.4f}│")
    conclusion = "STATIONNAIRE" if is_stationary else "NON STATIONNAIRE"
    print(f"│  Conclusion : {conclusion} (p={p_value:.3f})")
    print("└──────────────────────────────────────────────────┘")

    return {
        "label": label,
        "adf_stat": adf_stat,
        "p_value": p_value,
        "is_stationary": is_stationary,
        "n_lags": n_lags,
        "n_obs": n_obs,
        "critical_values": crit_vals,
    }


class TimeSeriesFeaturizer:
    """Pipeline agrégation horaire + lags, rolling, différences, Fourier."""

    def __init__(
        self,
        lags: List[int] = None,
        windows: List[int] = None,
        n_harmonics: int = 5,
    ) -> None:
        self.lags = lags if lags is not None else [1, 24, 168]
        self.windows = windows if windows is not None else [3, 24, 168]
        self.n_harmonics = n_harmonics
        self._logger = logging.getLogger(self.__class__.__name__)

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str = "trip_count",
        group_col: str = "PULocationID",
    ) -> pd.DataFrame:
        """Pipeline complet : agrégation + toutes les features TS."""
        with Timer("Agrégation horaire", unit="ms") as t:
            df_hourly = aggregate_hourly(df, group_col)
        self._logger.info("Agrégation : %.1fms", t.elapsed)

        with Timer("Lag features", unit="ms") as t:
            df_hourly = add_lag_features(df_hourly, target_col, group_col, self.lags)
        self._logger.info("Lags : %.1fms", t.elapsed)

        with Timer("Rolling features", unit="ms") as t:
            df_hourly = add_rolling_features(df_hourly, target_col, group_col, self.windows)
        self._logger.info("Rolling : %.1fms", t.elapsed)

        with Timer("Différenciation", unit="ms") as t:
            df_hourly = add_differencing(df_hourly, target_col, group_col)
        self._logger.info("Diff : %.1fms", t.elapsed)

        with Timer("Fourier", unit="ms") as t:
            df_hourly = add_fourier_features(df_hourly, n_harmonics=self.n_harmonics)
        self._logger.info("Fourier : %.1fms", t.elapsed)

        n_features = df_hourly.shape[1]
        self._logger.info(
            "TimeSeriesFeaturizer terminé | %s features | %s lignes",
            n_features,
            f"{len(df_hourly):,}",
        )
        return df_hourly


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== EXERCICE 3.8 — TIME SERIES ENGINEERING ===\n")

    df_raw = generate_taxi_df(n_rows=200_000)

    print("--- A. Agrégation Horaire ---")
    with Timer("aggregate_hourly", unit="ms") as t:
        df_hourly = aggregate_hourly(df_raw)
    print(f"Shape : {df_hourly.shape}")
    print(f"Zones : {df_hourly['PULocationID'].nunique()}")
    print(df_hourly.head(5).to_string())

    print("\n--- B. Lag Features ---")
    df_lags = add_lag_features(df_hourly.copy(), lags=[1, 24, 168])
    lag_cols = [c for c in df_lags.columns if "lag" in c]
    print(f"Colonnes lag : {lag_cols}")
    print(df_lags[["pickup_hour_ts", "trip_count"] + lag_cols[:3]].head(10).to_string())

    print("\n--- C. Rolling Statistics ---")
    df_roll = add_rolling_features(df_hourly.copy(), windows=[3, 24])
    roll_cols = [c for c in df_roll.columns if "roll" in c]
    print(f"Colonnes rolling ({len(roll_cols)}) : {roll_cols[:4]}...")

    print("\n--- D. Différenciation ---")
    df_diff = add_differencing(df_hourly.copy())
    diff_cols = [c for c in df_diff.columns if "diff" in c]
    print(f"Colonnes diff : {diff_cols}")

    print("\n--- E. Test ADF (Stationnarité) ---")
    if STATSMODELS_AVAILABLE:
        zone_data = df_hourly[df_hourly["PULocationID"] == 132]
        if len(zone_data) >= 20:
            ts_raw = zone_data["trip_count"]
            ts_diff = zone_data["trip_count"].diff(1).dropna()

            print("\nSérie ORIGINALE :")
            adf_raw = adf_test(ts_raw, label="trip_count zone 132")

            print("\nSérie DIFFÉRENCIÉE (ordre 1) :")
            adf_diff = adf_test(ts_diff, label="trip_count diff(1) zone 132")

            if adf_raw.get("available") is not False:
                print(f"\nOriginal  stationnaire : {adf_raw['is_stationary']}")
                print(f"Différencié stationnaire: {adf_diff['is_stationary']}")
    else:
        print("[SKIP] statsmodels absent")

    print("\n--- F. Fourier Features ---")
    df_fourier = add_fourier_features(df_hourly.copy(), n_harmonics=5)
    fourier_cols = [c for c in df_fourier.columns if "fourier" in c]
    print(f"Harmoniques créées ({len(fourier_cols)}) : {fourier_cols}")
    print(df_fourier[["pickup_hour_ts"] + fourier_cols[:4]].head(5).to_string())

    print("\n--- G. TimeSeriesFeaturizer Complet ---")
    featurizer = TimeSeriesFeaturizer(
        lags=[1, 24, 168],
        windows=[3, 24],
        n_harmonics=5,
    )
    with Timer("TimeSeriesFeaturizer.fit_transform()", unit="s") as t:
        df_ts = featurizer.fit_transform(df_raw.copy())
    print(f"Terminé en {t.elapsed:.2f}s")
    print(f"Shape finale : {df_ts.shape}")
    ts_cols = [c for c in df_ts.columns if any(x in c for x in ["lag", "roll", "diff", "fourier"])]
    print(f"Features TS ({len(ts_cols)}) : {ts_cols[:8]}...")
