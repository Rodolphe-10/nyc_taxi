"""Fenêtrage temporel, MultiIndex et pivot pour tableau de bord taxi par zone / heure."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

from src.core.context import Timer
from src.optim.benchmarks import generate_taxi_df
from src.optim.memory import optimize_dtypes

logger = logging.getLogger(__name__)


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prépare les colonnes temporelles, la cible et les filtres métier (courses plausibles)."""
    df = df.copy()

    # Features temporelles : croiser zone × heure × jour (MultiIndex / pivot).
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_weekday"] = df["tpep_pickup_datetime"].dt.dayofweek
    df["pickup_date"] = df["tpep_pickup_datetime"].dt.date
    df["pickup_month"] = df["tpep_pickup_datetime"].dt.month

    df["tip_pct"] = (df["tip_amount"] / df["fare_amount"].clip(lower=0.01)).clip(0, 2)

    df["duration_min"] = (
        (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60
    ).clip(lower=0.1)

    df = df[df["fare_amount"].between(2.5, 200)]
    df = df[df["trip_distance"].between(0.1, 100)]
    df = df[df["passenger_count"].between(1, 6)]

    return optimize_dtypes(df, verbose=False)


def build_activity_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège l'activité par (zone de prise, heure, jour de la semaine).

    MultiIndex : lecture rapide par sous-clés (ex. toutes les heures d'une zone)
    sans jointure supplémentaire — adapté aux heatmaps et séries temporelles par PULocationID.
    """
    activity = df.groupby(["PULocationID", "pickup_hour", "pickup_weekday"], observed=False).agg(
        trip_count=("fare_amount", "count"),
        revenue=("fare_amount", "sum"),
        tip_mean=("tip_amount", "mean"),
        tip_pct_mean=("tip_pct", "mean"),
        distance_mean=("trip_distance", "mean"),
        duration_mean=("duration_min", "mean"),
    )

    print(f"Activity DataFrame : {activity.shape}")
    print(f"Index levels : {activity.index.names}")
    print(f"Zones uniques : {activity.index.get_level_values(0).nunique()}")

    return activity


def build_pivot_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Matrice zone × heure pour visualisation (heatmap) ou features tabulaires.

    Colonnes multiniveaux : une métrique par heure — prêt pour ``stack``/export ML.
    """
    d = df.copy()
    d["trip_count_proxy"] = 1

    pivot = pd.pivot_table(
        d,
        values=["trip_count_proxy", "tip_pct"],
        index="PULocationID",
        columns="pickup_hour",
        aggfunc={"trip_count_proxy": "count", "tip_pct": "mean"},
        fill_value=0,
    )

    print(f"Pivot shape : {pivot.shape}")
    print(f"Colonnes MultiLevel : {pivot.columns[:5].tolist()}")

    return pivot


def stack_unstack_demo(activity: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Démontre ``unstack`` / ``swaplevel`` : passer d'un index 3D à des matrices 2D
    (ex. heures en colonnes) pour comparer visuellement les profils horaires par zone.
    """
    out: Dict[str, pd.DataFrame] = {}

    # Heures en colonnes : (zone, weekday) × heure — profil horaire par jour-type.
    unstacked_hour = activity["tip_pct_mean"].unstack(level="pickup_hour")
    print(f"unstacked_hour shape : {unstacked_hour.shape}")
    print(unstacked_hour.head(3).to_string())
    out["unstacked_hour"] = unstacked_hour

    # Jour en colonnes : (zone, heure) × weekday — comparer lundi vs vendredi à 8h.
    unstacked_weekday = activity["trip_count"].unstack(level="pickup_weekday")
    print(f"unstacked_weekday shape : {unstacked_weekday.shape}")
    print(unstacked_weekday.head(3).to_string())
    out["unstacked_weekday"] = unstacked_weekday

    swapped = activity.swaplevel("PULocationID", "pickup_hour").sort_index()
    print(f"swaplevel shape : {swapped.shape}")
    print(swapped.head(3).to_string())
    out["swaplevel"] = swapped

    return out


def top_zones_by_tip(activity: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Classement des couples (zone, heure, jour) les plus « généreux » en pourboire moyen.

    ``pd.IndexSlice`` permet de requêter la grille (zone, heure, jour) sans ``merge`` :
    ici on l'utilise sur la vue complète avant tri (cas d'usage drill-down dashboard).
    """
    idx = pd.IndexSlice
    ranked = (
        activity.loc[idx[:, :, :], :]
        .reset_index()
        .sort_values("tip_pct_mean", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    print("╔═══╦══════════════╦══════╦═════════╦══════════════╗")
    print("║ # ║ Zone Pickup  ║ Heure║ Weekday ║ Tip moyen %  ║")
    print("╠═══╬══════════════╬══════╬═════════╬══════════════╣")
    for i, row in ranked.iterrows():
        print(
            f"║ {i + 1:>2} ║ {int(row['PULocationID']):>12} ║ {int(row['pickup_hour']):>4} ║ "
            f"{int(row['pickup_weekday']):>7} ║ {row['tip_pct_mean']:>12.3f} ║"
        )
    print("╚═══╩══════════════╩══════╩═════════╩══════════════╝")

    return ranked


def trimmed_mean(x: pd.Series, pct: float = 0.1) -> float:
    """
    Moyenne tronquée : robuste aux outliers sur ``tip_pct`` / montants.

    # La moyenne tronquée est robuste aux outliers
    # Elle ignore les pct% valeurs extrêmes de chaque côté
    # Utile pour tip_pct qui a des valeurs aberrantes
    """
    s = pd.Series(x).dropna()
    n = len(s)
    if n == 0:
        return float("nan")
    k = int(n * pct)
    if k == 0:
        if n < 5:
            return float(s.mean())
        k = 1
    x_sorted = s.sort_values()
    if 2 * k >= n:
        return float(s.mean())
    x_trimmed = x_sorted.iloc[k:-k]
    return float(x_trimmed.mean())


def agg_with_custom_functions(df: pd.DataFrame) -> pd.DataFrame:
    """Compare moyenne simple vs moyenne tronquée par zone (agrégations nommées)."""
    result = df.groupby("PULocationID", observed=False).agg(
        tip_pct_mean=("tip_pct", "mean"),
        tip_pct_trimmed=("tip_pct", trimmed_mean),
        tip_pct_trimmed_20pct=("tip_pct", lambda x: trimmed_mean(x, pct=0.2)),
        fare_median=("fare_amount", "median"),
        fare_trimmed=("fare_amount", trimmed_mean),
        trip_count=("fare_amount", "count"),
    )

    diff = (result["tip_pct_mean"] - result["tip_pct_trimmed"]).abs().mean()
    print(f"Différence moyenne : {diff:.4f}")

    return result


def compute_rolling_tip(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Moyenne du tip sur les N courses **précédentes** dans la même zone (PULocationID).

    # shift(1) : exclut la course courante (évite la fuite de données)
    # rolling(window) : fenêtre glissante de N courses précédentes
    # groupby() : calcul indépendant par zone
    # min_periods=1 : calcule même si moins de N courses disponibles
    """
    df = df.copy()
    df = df.sort_values(["PULocationID", "tpep_pickup_datetime"])

    df["tip_rolling_mean"] = df.groupby("PULocationID", observed=False)["tip_amount"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )

    cap = float(df["tip_amount"].max() * 2)
    if df["tip_rolling_mean"].gt(cap).any():
        logger.warning("tip_rolling_mean dépasse le plafond attendu — clipping appliqué")
    df["tip_rolling_mean"] = df["tip_rolling_mean"].clip(upper=cap)

    logger.info("Rolling tip calculé | window=%s", window)
    return df


def compute_fare_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rang percentile du ``fare_amount`` parmi les courses du même jour et de la même zone.

    # rank(pct=True) : retourne le rang en pourcentage [0, 1]
    # method="average" : moyenne des rangs en cas d'égalité
    # Interprétation : 0.9 = cette course coûte plus que 90% des courses
    #                  de la même zone ce jour-là
    """
    df = df.copy()
    df["pickup_date"] = df["tpep_pickup_datetime"].dt.date

    df["fare_rank_pct"] = df.groupby(["PULocationID", "pickup_date"], observed=False)[
        "fare_amount"
    ].rank(pct=True, method="average")

    logger.info("Fare rank calculé")
    return df


def compute_expanding_tip_pct(df: pd.DataFrame) -> pd.DataFrame:
    """
    Moyenne cumulative du ``tip_pct`` par zone (réputation « tip » historique).

    # expanding() : fenêtre qui s'agrandit depuis le début
    # Contrairement à rolling(), utilise TOUTES les observations passées
    # Converge vers la vraie moyenne de la zone au fil du temps
    # Utile comme estimateur de la "réputation tip" d'une zone
    """
    df = df.copy()
    df = df.sort_values(["PULocationID", "tpep_pickup_datetime"])

    df["tip_pct_expanding"] = df.groupby("PULocationID", observed=False)["tip_pct"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )

    logger.info("Expanding tip_pct calculé")
    return df


def compute_lag_features(df: pd.DataFrame, lags: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Nombre de courses la veille / il y a N jours dans la zone (activité passée).

    # lag_1d  : nombre de courses hier dans cette zone
    # lag_3d  : il y a 3 jours
    # lag_7d  : même jour la semaine dernière (saisonnalité hebdo)
    # Ces features capturent les tendances d'activité par zone
    """
    lags = lags or [1, 3, 7]
    df = df.copy()
    df["pickup_date"] = df["tpep_pickup_datetime"].dt.date

    daily_counts = (
        df.groupby(["PULocationID", "pickup_date"], observed=False).size().reset_index(name="daily_trip_count")
    )
    daily_counts = daily_counts.sort_values(["PULocationID", "pickup_date"])

    for lag in lags:
        daily_counts[f"trip_count_lag_{lag}d"] = daily_counts.groupby("PULocationID")[
            "daily_trip_count"
        ].shift(lag)

    merge_cols = daily_counts.drop(columns=["daily_trip_count"])
    df = df.merge(merge_cols, on=["PULocationID", "pickup_date"], how="left")

    logger.info("Lag features calculées : %s", [f"lag_{l}d" for l in lags])
    return df


def compute_all_window_features(
    df: pd.DataFrame,
    rolling_window: int = 5,
    lags: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Pipeline : préparation puis rolling, rang fare, expanding tip%, lags — chronométré."""
    lags = lags or [1, 3, 7]

    with Timer("prepare_dataframe", unit="ms") as t0:
        out = prepare_dataframe(df)
    print(f"[prepare_dataframe] {t0.elapsed:.1f}ms | shape={out.shape}")

    with Timer("compute_rolling_tip", unit="ms") as t1:
        out = compute_rolling_tip(out, window=rolling_window)
    print(f"[compute_rolling_tip] {t1.elapsed:.1f}ms | shape={out.shape}")

    with Timer("compute_fare_rank", unit="ms") as t2:
        out = compute_fare_rank(out)
    print(f"[compute_fare_rank] {t2.elapsed:.1f}ms | shape={out.shape}")

    with Timer("compute_expanding_tip_pct", unit="ms") as t3:
        out = compute_expanding_tip_pct(out)
    print(f"[compute_expanding_tip_pct] {t3.elapsed:.1f}ms | shape={out.shape}")

    with Timer("compute_lag_features", unit="ms") as t4:
        out = compute_lag_features(out, lags=lags)
    print(f"[compute_lag_features] {t4.elapsed:.1f}ms | shape={out.shape}")

    print(
        """
Nouvelles features créées :
- tip_rolling_mean   : moyenne tip des N dernières courses (zone)
- fare_rank_pct      : rang percentile fare dans (zone, jour)
- tip_pct_expanding  : moyenne cumulative tip_pct (zone)
- trip_count_lag_Nd  : nb courses il y a N jours (zone)
"""
    )

    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== EXERCICE 3.2 — MULTIINDEX & PIVOTING ===\n")

    df_raw = generate_taxi_df(n_rows=200_000)
    df = prepare_dataframe(df_raw)
    print(f"Dataset préparé : {df.shape}\n")

    print("--- A. MultiIndex Activity ---")
    activity = build_activity_multiindex(df)
    print(activity.head(10).to_string())

    print("\n--- B. Stack / Unstack ---")
    stack_unstack_demo(activity)

    print("\n--- C. Pivot Table zone × heure ---")
    pivot = build_pivot_table(df)
    print(f"Pivot shape : {pivot.shape}")
    print(pivot.iloc[:5, :5].to_string())

    print("\n--- D. Top 10 zones par tip_pct ---")
    top10 = top_zones_by_tip(activity, top_n=10)
    print(top10.to_string())

    print("\n--- E. Agrégation avec trimmed_mean ---")
    agg_result = agg_with_custom_functions(df)
    print(agg_result.head(10).to_string())
    diff = (agg_result["tip_pct_mean"] - agg_result["tip_pct_trimmed"]).abs().mean()
    print(f"\nDifférence mean vs trimmed_mean : {diff:.4f}")

    print("\n=== EXERCICE 3.3 — WINDOW FUNCTIONS ===\n")

    df_raw = generate_taxi_df(n_rows=100_000)
    df = prepare_dataframe(df_raw)

    print("--- A. Rolling Tip Mean (window=5) ---")
    with Timer("rolling_tip", unit="ms") as t:
        df_rolling = compute_rolling_tip(df.copy(), window=5)
    print(f"Calculé en {t.elapsed:.1f}ms")
    print(df_rolling[["PULocationID", "tip_amount", "tip_rolling_mean"]].head(10).to_string())
    null_pct = df_rolling["tip_rolling_mean"].isnull().mean() * 100
    print(f"Valeurs nulles : {null_pct:.1f}%")

    print("\n--- B. Fare Rank Percentile ---")
    with Timer("fare_rank", unit="ms") as t:
        df_rank = compute_fare_rank(df.copy())
    print(f"Calculé en {t.elapsed:.1f}ms")
    print(df_rank[["PULocationID", "fare_amount", "fare_rank_pct"]].describe().to_string())

    print("\n--- C. Expanding Tip Pct Mean ---")
    with Timer("expanding", unit="ms") as t:
        df_exp = compute_expanding_tip_pct(df.copy())
    print(f"Calculé en {t.elapsed:.1f}ms")
    z = int(df_exp["PULocationID"].iloc[0])
    sample = df_exp[df_exp["PULocationID"] == z].head(10)
    print(sample[["tpep_pickup_datetime", "tip_pct", "tip_pct_expanding"]].to_string())

    print("\n--- D. Lag Features ---")
    with Timer("lags", unit="ms") as t:
        df_lag = compute_lag_features(df.copy(), lags=[1, 3, 7])
    print(f"Calculé en {t.elapsed:.1f}ms")
    lag_cols = [c for c in df_lag.columns if "lag" in c]
    print(f"Colonnes lag créées : {lag_cols}")
    print(df_lag[lag_cols].describe().to_string())

    print("\n--- E. Pipeline Window Features Complet ---")
    df_features = compute_all_window_features(df_raw.copy(), rolling_window=5, lags=[1, 3, 7])
    new_cols = ["tip_rolling_mean", "fare_rank_pct", "tip_pct_expanding"] + [
        f"trip_count_lag_{l}d" for l in [1, 3, 7]
    ]
    print(f"\nFeatures créées : {new_cols}")
    print(f"Shape finale : {df_features.shape}")
    print(df_features[new_cols].describe().to_string())
