"""Benchmarks et profilage des transformations de features."""

import cProfile
import io
import logging
import math
import time
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd
import psutil
import pstats
import tracemalloc

from src.core.context import Timer


# Génère un dataset synthétique proche des colonnes NYC Taxi.
def generate_taxi_df(n_rows=100_000, seed=42):
    """Construit un DataFrame réaliste pour les benchmarks."""
    np.random.seed(seed)
    pickup = pd.date_range("2023-01-01", periods=n_rows, freq="1min")
    duration_min = np.random.uniform(5, 60, n_rows)
    dropoff = pickup + pd.to_timedelta(duration_min, unit="min")

    fare_amount = np.random.uniform(2.5, 80.0, n_rows)
    tip_amount = np.random.uniform(0.0, 15.0, n_rows)
    trip_distance = np.random.uniform(0.1, 25.0, n_rows)
    surcharge = np.random.uniform(0.5, 5.0, n_rows)

    df = pd.DataFrame(
        {
            "tpep_pickup_datetime": pickup,
            "tpep_dropoff_datetime": dropoff,
            "fare_amount": fare_amount,
            "tip_amount": tip_amount,
            "trip_distance": trip_distance,
            "passenger_count": np.random.randint(1, 6, n_rows),
            "PULocationID": np.random.randint(1, 265, n_rows),
            "DOLocationID": np.random.randint(1, 265, n_rows),
            "payment_type": np.random.randint(1, 4, n_rows),
            "total_amount": fare_amount + tip_amount + surcharge,
        }
    )
    return df


# Implémentation volontairement lente pour montrer les limites de iterrows().
def compute_features_naive(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule des features ligne par ligne avec un surcoût Python élevé."""
    result = df.copy()
    durations = []
    speeds = []
    penalties = []

    # iterrows() crée un objet Python Series pour chaque ligne
    # → overhead interpréteur × N lignes
    # → pas de vectorisation SIMD
    # → pas de parallélisme
    # GIL actif sur chaque opération Python
    for _, row in result.iterrows():
        duration_min = (row["tpep_dropoff_datetime"] - row["tpep_pickup_datetime"]).total_seconds() / 60
        speed_mph = row["trip_distance"] / max(duration_min / 60, 0.001)
        penalty = 0.5 * row["trip_distance"] ** 2 + math.log(row["fare_amount"] + 1)
        durations.append(duration_min)
        speeds.append(speed_mph)
        penalties.append(penalty)

    result["duration_min"] = durations
    result["speed_mph"] = speeds
    result["penalty"] = penalties
    return result


# cProfile mesure le coût CPU par fonction (temps cumulé et appels).
def profile_with_cprofile(func, df, n_rows=10_000) -> str:
    """Profile une fonction avec cProfile et retourne le rapport texte."""
    profiler = cProfile.Profile()
    subset = df.head(n_rows).copy()
    profiler.enable()
    func(subset)
    profiler.disable()

    output = io.StringIO()
    stats = pstats.Stats(profiler, stream=output).sort_stats("cumulative")
    stats.print_stats(10)
    report = output.getvalue()

    with open("profil_naive_cprofile.txt", "w", encoding="utf-8") as file:
        file.write(report)

    return report


# tracemalloc mesure les allocations Python (mémoire courante + pic).
def profile_with_memory(func, df, n_rows=10_000) -> Dict:
    """Profile la mémoire consommée par une fonction."""
    subset = df.head(n_rows).copy()
    process = psutil.Process()
    _before_mb = process.memory_info().rss / 1e6

    tracemalloc.start()
    func(subset)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "peak_mb": peak / 1e6,
        "current_mb": current / 1e6,
        "n_rows": n_rows,
    }


# Benchmark temporel répétable (plusieurs runs) pour limiter la variance.
def benchmark_function(func: Callable, df: pd.DataFrame, label: str, n_runs=3) -> Dict:
    """Exécute une fonction plusieurs fois et retourne des statistiques robustes."""
    if n_runs < 3:
        n_runs = 3

    durations_s = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func(df.copy())
        durations_s.append(time.perf_counter() - start)

    min_s = min(durations_s)
    max_s = max(durations_s)
    mean_s = float(np.mean(durations_s))
    std_s = float(np.std(durations_s))

    stats = {
        "label": label,
        "n_rows": len(df),
        "n_runs": n_runs,
        "min_ms": min_s * 1000,
        "max_ms": max_s * 1000,
        "mean_ms": mean_s * 1000,
        "std_ms": std_s * 1000,
        "rows_per_second": len(df) / mean_s if mean_s > 0 else float("inf"),
    }

    print("┌─────────────────────────────────────────────────┐")
    print(f"│ Benchmark : {label}")
    print(f"│ Lignes    : {len(df):,}")
    print(f"│ Runs      : {n_runs}")
    print(f"│ Min       : {stats['min_ms']:.1f} ms")
    print(f"│ Max       : {stats['max_ms']:.1f} ms")
    print(f"│ Moyenne   : {stats['mean_ms']:.1f} ms")
    print(f"│ Débit     : {stats['rows_per_second']:,.0f} lignes/s")
    print("└─────────────────────────────────────────────────┘")

    return stats


# Comparaison multi-fonctions sur le même dataset avec calcul de speedup.
def compare_functions(funcs: Dict[str, Callable], df: pd.DataFrame) -> pd.DataFrame:
    """Compare plusieurs fonctions et retourne un tableau synthétique."""
    rows = []
    for label, func in funcs.items():
        stats = benchmark_function(func, df, label=label, n_runs=3)
        rows.append(
            {
                "label": label,
                "mean_ms": stats["mean_ms"],
                "rows_per_second": stats["rows_per_second"],
            }
        )

    comp = pd.DataFrame(rows)
    if comp.empty:
        return comp

    baseline = comp.iloc[0]["mean_ms"]
    comp["speedup"] = baseline / comp["mean_ms"]
    comp = comp.sort_values(by="mean_ms").reset_index(drop=True)
    print(comp.to_string(index=False))
    return comp


if __name__ == "__main__":
    print("Génération du dataset...")
    df_small = generate_taxi_df(n_rows=10_000)
    df_large = generate_taxi_df(n_rows=100_000)
    print(f"Dataset small : {df_small.shape}")
    print(f"Dataset large : {df_large.shape}")

    print("\n--- BENCHMARK NAÏF ---")
    stats = benchmark_function(
        compute_features_naive,
        df_small,
        label="compute_features_naive (iterrows)",
        n_runs=3,
    )

    print("\n--- PROFILAGE cProfile ---")
    rapport = profile_with_cprofile(compute_features_naive, df_small)
    print("Top 10 fonctions les plus coûteuses :")
    print(rapport[:2000])

    print("\n--- PROFILAGE MÉMOIRE ---")
    mem_stats = profile_with_memory(compute_features_naive, df_small)
    print(f"Pic mémoire : {mem_stats['peak_mb']:.2f} Mo")
    print(f"Mémoire courante : {mem_stats['current_mb']:.2f} Mo")

    print(
        """
CONCLUSIONS DU PROFILAGE :
─────────────────────────────────────────────────
cProfile révèle :
→ iterrows() représente ~80% du temps d'exécution
→ Chaque appel crée un objet Series Python coûteux
→ math.log() et l'arithmétique Python ajoutent du surcoût

memory_profiler révèle :
→ Pic mémoire proportionnel à n_rows
→ Chaque Series créée par iterrows() consomme de la mémoire
→ Pas de réutilisation du buffer mémoire (contrairement à NumPy)

SOLUTION → Exercice 2.2 : vectorisation NumPy/Pandas
Attendu : speedup ≥ ×100
"""
    )
