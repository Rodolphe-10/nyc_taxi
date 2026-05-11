"""Vectorisation NumPy/Pandas et benchmarks de performance."""

import logging

import numpy as np
import pandas as pd

from src.core.context import Timer
from src.optim.benchmarks import (
    benchmark_function,
    compare_functions,
    compute_features_naive,
    generate_taxi_df,
)


# Version vectorisée des calculs de features (sans boucle Python explicite).
def compute_features_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule duration/speed/penalty en opérations vectorisées."""
    df = df.copy()

    # Équivalent naïf : (dropoff - pickup).total_seconds()/60 ligne par ligne.
    # Ici, Pandas applique l'opération à toute la colonne datetime d'un coup.
    duration = df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    df["duration_min"] = duration.dt.total_seconds() / 60

    # Équivalent naïf : trip_distance / max(duration_h, 0.001) par ligne.
    # clip() applique la borne en C sur toute la série.
    duration_h = (df["duration_min"] / 60).clip(lower=0.001)
    df["speed_mph"] = df["trip_distance"] / duration_h

    # Équivalent naïf : 0.5*d^2 + log(fare+1) par ligne.
    # np.log1p est une ufunc C rapide et stable numériquement.
    df["penalty"] = 0.5 * df["trip_distance"] ** 2 + np.log1p(df["fare_amount"])

    # Pas de boucle Python → pas d'overhead interpréteur par ligne
    # NumPy opère sur des buffers C contigus en mémoire
    # Instructions SIMD (AVX2) : traitement de 4-8 float64 par cycle CPU
    # dt.total_seconds() : opération vectorisée sur DatetimeIndex
    # np.log1p : ufunc C optimisée (plus précise que log(x+1) pour x petit)
    # .clip() : évite la division par zéro sans branchement Python
    return df


# Optimise les types pour réduire l'empreinte mémoire sans perte majeure.
def optimize_dtypes(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Délègue à ``src.optim.memory.optimize_dtypes`` (carte NYC + passes auto)."""
    from src.optim.memory import optimize_dtypes as _memory_optimize_dtypes

    return _memory_optimize_dtypes(df, custom_map=None, verbose=verbose)


# Matrice de distances euclidiennes via broadcasting NumPy pur.
def compute_zone_distance_matrix(zone_coords: np.ndarray) -> np.ndarray:
    """Calcule la distance euclidienne entre toutes les paires de zones."""
    # Broadcasting évite de créer des boucles imbriquées
    # NumPy étend les dimensions de taille 1 automatiquement
    # Résultat : même calcul qu'une double boucle but en C compilé
    # Mémoire : (265, 265, 2) × 8 octets = ~1.1 Mo — acceptable
    diff = zone_coords[:, np.newaxis, :] - zone_coords[np.newaxis, :, :]
    distances = np.sqrt((diff ** 2).sum(axis=2))
    return distances


# Matrice de distances haversine vectorisée (distance réelle sur sphère).
def compute_zone_distance_matrix_haversine(zone_coords: np.ndarray) -> np.ndarray:
    """Calcule la distance haversine (km) entre toutes les paires de zones."""
    coords_rad = np.radians(zone_coords)
    lat = coords_rad[:, 0]
    lon = coords_rad[:, 1]

    lat1 = lat[:, np.newaxis]
    lat2 = lat[np.newaxis, :]
    lon1 = lon[:, np.newaxis]
    lon2 = lon[np.newaxis, :]

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r_km = 6371.0
    return r_km * c


# Benchmark complet naïf vs vectorisé et contrôle de cohérence numérique.
def benchmark_vectorization(n_rows=100_000) -> pd.DataFrame:
    """Compare naïf vs vectorisé et renvoie un tableau comparatif."""
    df = generate_taxi_df(n_rows=n_rows)

    stats_naive = benchmark_function(
        compute_features_naive, df, label="compute_features_naive", n_runs=3
    )
    stats_vect = benchmark_function(
        compute_features_vectorized, df, label="compute_features_vectorized", n_runs=5
    )

    speedup = stats_naive["mean_ms"] / stats_vect["mean_ms"]
    print("╔══════════════════════════════════════════════════╗")
    print(f"║     BENCHMARK VECTORISATION — {n_rows:,} lignes  ║")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║ Naïf (iterrows)   : {stats_naive['mean_ms']:.1f} ms")
    print(f"║ Vectorisé (NumPy) : {stats_vect['mean_ms']:.1f} ms")
    print(f"║ Speedup           : {speedup:.1f}x")
    print(f"║ Débit naïf        : {stats_naive['rows_per_second']:,.0f} lignes/s")
    print(f"║ Débit vectorisé   : {stats_vect['rows_per_second']:,.0f} lignes/s")
    print("╚══════════════════════════════════════════════════╝")

    result_naive = compute_features_naive(df.copy())
    result_vect = compute_features_vectorized(df.copy())
    np.testing.assert_allclose(
        result_naive["duration_min"].values,
        result_vect["duration_min"].values,
        rtol=1e-4,
    )
    print("[OK] Résultats numériquement identiques")

    funcs = {
        "naive": compute_features_naive,
        "vectorized": compute_features_vectorized,
    }
    return compare_functions(funcs, df)


if __name__ == "__main__":
    print("=== EXERCICE 2.2 — VECTORISATION ===\n")
    df_cmp = benchmark_vectorization(n_rows=100_000)
    print(df_cmp.to_string())

    print("\n=== OPTIMISATION MÉMOIRE ===")
    df_raw = generate_taxi_df(n_rows=100_000)
    df_opt = optimize_dtypes(df_raw.copy())
    ram_avant = df_raw.memory_usage(deep=True).sum() / 1e6
    ram_apres = df_opt.memory_usage(deep=True).sum() / 1e6
    print(f"RAM avant  : {ram_avant:.1f} Mo")
    print(f"RAM après  : {ram_apres:.1f} Mo")
    print(f"Gain       : {(ram_avant - ram_apres) / ram_avant * 100:.1f}%")

    print("\n=== BROADCASTING — MATRICE DE DISTANCES ===")
    np.random.seed(42)
    coords = np.random.uniform(low=[40.5, -74.3], high=[40.9, -73.7], size=(265, 2))
    with Timer("Distance euclidienne (broadcasting)") as t:
        mat_eucl = compute_zone_distance_matrix(coords)
    print(f"Matrice euclidienne : {mat_eucl.shape} en {t.elapsed:.1f}ms")

    with Timer("Distance haversine (broadcasting)") as t:
        mat_harv = compute_zone_distance_matrix_haversine(coords)
    print(f"Matrice haversine   : {mat_harv.shape} en {t.elapsed:.1f}ms")
    print(f"Distance max        : {mat_harv.max():.1f} km")
    print(f"Distance moy        : {mat_harv.mean():.1f} km")
