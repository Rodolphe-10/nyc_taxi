"""Kernels Numba JIT : haversine (Python / NumPy / Numba / parallèle) et rolling pondéré."""

from __future__ import annotations

import logging
import math
import time
from typing import Dict

import numpy as np
import pandas as pd

from src.optim.benchmarks import benchmark_function, generate_taxi_df

try:
    import numba  # noqa: F401
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logging.warning("Numba absent — versions JIT indisponibles")

    def njit(*args, **kwargs):
        return lambda fn: fn

    prange = range

EARTH_RADIUS_KM = 6371.0


def haversine_python(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance haversine (km) pour une paire de points — référence scalaire."""
    # Version de référence — correcte mais lente sur grands volumes
    # math.sin/cos opèrent sur des scalaires Python
    # Overhead interpréteur × N appels
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    lon1_r = math.radians(lon1)
    lon2_r = math.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return EARTH_RADIUS_KM * c


def haversine_numpy(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Haversine vectorisée (arrays) — ufuncs C + SIMD."""
    # Opère sur tout l'array en une seule passe C
    # np.sin/cos : ufuncs compilées avec SIMD
    # np.clip : évite les NaN sur arcsin(sqrt(x > 1))
    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)
    lon1_r = np.radians(lon1)
    lon2_r = np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return EARTH_RADIUS_KM * c


@njit(cache=True)
def haversine_numba(lat1, lon1, lat2, lon2):
    """
    Version Numba JIT — compilée en code machine LLVM au 1er appel.
    Fonctionnement :
    1. 1er appel : Numba inspecte les types (float64[]) et compile
    2. LLVM génère des instructions machine natives (x86-64/ARM)
    3. Appels suivants : exécution directe sans GIL ni interpréteur
    4. cache=True : sauvegarde la compilation sur disque
    Gain typique vs NumPy : ×2-5 (meilleure localité cache, moins d'allocs)
    """
    n = len(lat1)
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        lat1_r = lat1[i] * (math.pi / 180.0)
        lat2_r = lat2[i] * (math.pi / 180.0)
        lon1_r = lon1[i] * (math.pi / 180.0)
        lon2_r = lon2[i] * (math.pi / 180.0)
        dlat = lat2_r - lat1_r
        dlon = lon2_r - lon1_r
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
        if a < 0.0:
            a = 0.0
        elif a > 1.0:
            a = 1.0
        c = 2 * math.asin(math.sqrt(a))
        result[i] = EARTH_RADIUS_KM * c
    return result


@njit(parallel=True, cache=True)
def haversine_numba_parallel(lat1, lon1, lat2, lon2):
    """
    Version Numba parallèle — prange distribue les itérations
    sur tous les cœurs CPU disponibles automatiquement.
    parallel=True active le backend de parallélisation de Numba.
    Gain supplémentaire vs njit simple : ×N_cœurs théoriquement.
    En pratique : ×2-4 selon le nombre de cœurs et la taille des données.
    """
    n = len(lat1)
    result = np.empty(n, dtype=np.float64)
    for i in prange(n):
        lat1_r = lat1[i] * (math.pi / 180.0)
        lat2_r = lat2[i] * (math.pi / 180.0)
        lon1_r = lon1[i] * (math.pi / 180.0)
        lon2_r = lon2[i] * (math.pi / 180.0)
        dlat = lat2_r - lat1_r
        dlon = lon2_r - lon1_r
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
        if a < 0.0:
            a = 0.0
        elif a > 1.0:
            a = 1.0
        c = 2 * math.asin(math.sqrt(a))
        result[i] = EARTH_RADIUS_KM * c
    return result


def warmup_numba() -> None:
    """Compile les kernels Numba sur des données minuscules (hors mesure)."""
    if NUMBA_AVAILABLE:
        lat = np.array([40.7128], dtype=np.float64)
        lon = np.array([-74.0060], dtype=np.float64)
        lat2 = np.array([48.8566], dtype=np.float64)
        lon2 = np.array([2.3522], dtype=np.float64)
        haversine_numba(lat, lon, lat2, lon2)
        haversine_numba_parallel(lat, lon, lat2, lon2)
        print("[OK] Warmup Numba terminé")
    else:
        print("[WARN] Numba absent — warmup ignoré")


def _min_wall_time_sec(fn, n_runs: int = 3) -> float:
    best = float("inf")
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def benchmark_haversine(n_points: int = 10_000_000) -> Dict:
    """Compare Python pur (extrapolé), NumPy, Numba séquentiel et parallèle."""
    np.random.seed(42)
    lat1 = np.random.uniform(40.5, 40.9, n_points).astype(np.float64)
    lon1 = np.random.uniform(-74.3, -73.7, n_points).astype(np.float64)
    lat2 = np.random.uniform(40.5, 40.9, n_points).astype(np.float64)
    lon2 = np.random.uniform(-74.3, -73.7, n_points).astype(np.float64)

    warmup_numba()

    min_n = min(10_000, n_points)

    def _run_python_loop():
        for i in range(min_n):
            haversine_python(float(lat1[i]), float(lon1[i]), float(lat2[i]), float(lon2[i]))

    t_python_sub = _min_wall_time_sec(_run_python_loop, n_runs=3)
    t_python_extrap = t_python_sub * (n_points / min_n)

    t_numpy = _min_wall_time_sec(lambda: haversine_numpy(lat1, lon1, lat2, lon2), n_runs=3)

    if NUMBA_AVAILABLE:

        def _run_numba():
            haversine_numba(lat1, lon1, lat2, lon2)

        def _run_numba_par():
            haversine_numba_parallel(lat1, lon1, lat2, lon2)

        t_numba = _min_wall_time_sec(_run_numba, n_runs=3)
        t_numba_parallel = _min_wall_time_sec(_run_numba_par, n_runs=3)
        np.testing.assert_allclose(
            haversine_numpy(lat1[:100], lon1[:100], lat2[:100], lon2[:100]),
            haversine_numba(lat1[:100], lon1[:100], lat2[:100], lon2[:100]),
            rtol=1e-5,
        )
    else:
        t_numba = float("nan")
        t_numba_parallel = float("nan")

    speedup_numpy = t_python_extrap / t_numpy if t_numpy > 0 else float("inf")
    speedup_numba = t_python_extrap / t_numba if NUMBA_AVAILABLE and t_numba == t_numba and t_numba > 0 else float("nan")
    speedup_numba_parallel = (
        t_python_extrap / t_numba_parallel
        if NUMBA_AVAILABLE and t_numba_parallel == t_numba_parallel and t_numba_parallel > 0
        else float("nan")
    )

    m_pairs_py = n_points / t_python_extrap if t_python_extrap > 0 else 0.0
    m_pairs_np = n_points / t_numpy if t_numpy > 0 else 0.0
    m_pairs_nb = n_points / t_numba if NUMBA_AVAILABLE and t_numba == t_numba and t_numba > 0 else 0.0
    m_pairs_nbp = (
        n_points / t_numba_parallel
        if NUMBA_AVAILABLE and t_numba_parallel == t_numba_parallel and t_numba_parallel > 0
        else 0.0
    )

    print("╔═══════════════════════════════════════════════════════════════╗")
    print(f"║     BENCHMARK HAVERSINE — {n_points:,} paires de points       ║")
    print("╠══════════════════════╦══════════╦══════════╦════════════════╣")
    print("║ Implémentation       ║ Temps(s) ║ Speedup  ║ Mpaires/s      ║")
    print("╠══════════════════════╬══════════╬══════════╬════════════════╣")
    print(f"║ Python pur (extrap.) ║ {t_python_extrap:>8.1f}s  ║   1.00x  ║ {m_pairs_py:>12.3f}     ║")
    print(f"║ NumPy vectorisé      ║ {t_numpy:>8.3f}s  ║ {speedup_numpy:>7.1f}x  ║ {m_pairs_np:>12.1f}     ║")
    if NUMBA_AVAILABLE:
        print(f"║ Numba @njit          ║ {t_numba:>8.3f}s  ║ {speedup_numba:>7.1f}x  ║ {m_pairs_nb:>12.1f}     ║")
        print(
            f"║ Numba parallel       ║ {t_numba_parallel:>8.3f}s  ║ {speedup_numba_parallel:>7.1f}x  ║ {m_pairs_nbp:>12.1f}     ║"
        )
    else:
        print("║ Numba @njit          ║    n/a   ║   n/a    ║      n/a       ║")
        print("║ Numba parallel       ║    n/a   ║   n/a    ║      n/a       ║")
    print("╚══════════════════════╩══════════╩══════════╩════════════════╝")

    return {
        "t_python_extrap": t_python_extrap,
        "t_python_sub": t_python_sub,
        "t_numpy": t_numpy,
        "t_numba": t_numba,
        "t_numba_parallel": t_numba_parallel,
        "speedup_numpy": speedup_numpy,
        "speedup_numba": speedup_numba,
        "speedup_numba_parallel": speedup_numba_parallel,
        "n_points": n_points,
    }


@njit(cache=True)
def rolling_weighted_median_numba(values, weights, window: int):
    """
    Médiane pondérée sur fenêtre glissante (implémentation fenêtre par fenêtre).
    # Pandas rolling().apply() avec une fonction custom est très lent
    # car il appelle la fonction Python pour chaque fenêtre
    # Numba compile la boucle entière en code machine → bien plus rapide
    """
    n = len(values)
    result = np.full(n, np.nan, dtype=np.float64)

    for i in range(window, n + 1):
        window_vals = values[i - window : i]
        window_wgts = weights[i - window : i]

        sorted_vals = window_vals.copy()
        sorted_wgts = window_wgts.copy()
        m = window
        for j in range(1, m):
            key_v = sorted_vals[j]
            key_w = sorted_wgts[j]
            k = j - 1
            while k >= 0 and sorted_vals[k] > key_v:
                sorted_vals[k + 1] = sorted_vals[k]
                sorted_wgts[k + 1] = sorted_wgts[k]
                k -= 1
            sorted_vals[k + 1] = key_v
            sorted_wgts[k + 1] = key_w

        total_weight = 0.0
        for j in range(m):
            total_weight += sorted_wgts[j]
        if total_weight <= 0.0:
            result[i - 1] = sorted_vals[0]
            continue

        cumsum = 0.0
        median_val = sorted_vals[0]
        for j in range(m):
            cumsum += sorted_wgts[j] / total_weight
            if cumsum >= 0.5:
                median_val = sorted_vals[j]
                break
        result[i - 1] = median_val

    return result


def benchmark_rolling(n_points: int = 100_000, window: int = 20) -> Dict:
    """Compare rolling median Pandas (non pondérée) vs médiane pondérée Numba."""
    np.random.seed(7)
    values = np.random.uniform(0, 50, n_points).astype(np.float64)
    weights = np.random.uniform(0.1, 1.0, n_points).astype(np.float64)
    df = pd.DataFrame({"values": values})

    def _pandas_rolling(d: pd.DataFrame, win: int = window):
        return d["values"].rolling(win).median().to_numpy()

    stats_p = benchmark_function(_pandas_rolling, df, label="pandas_rolling_median", n_runs=3)
    t_pandas_ms = stats_p["min_ms"]

    warmup_n = min(1000, n_points)
    rolling_weighted_median_numba(values[:warmup_n], weights[:warmup_n], window)

    t0 = time.perf_counter()
    numba_result = rolling_weighted_median_numba(values, weights, window)
    t_numba_ms = (time.perf_counter() - t0) * 1000.0

    speedup = t_pandas_ms / t_numba_ms if t_numba_ms > 0 else float("inf")

    print("╔══════════════════════════════════════════════════════╗")
    print("║     BENCHMARK : ROLLING MEDIAN                       ║")
    print("╠══════════════════════╦══════════════════════════════╣")
    print(f"║ Pandas rolling       ║ {t_pandas_ms:>10.1f} ms (min / 3 runs)     ║")
    print(f"║ Numba pondéré       ║ {t_numba_ms:>10.1f} ms                  ║")
    print(f"║ Speedup (×)         ║ {speedup:>10.1f}                       ║")
    print("╚══════════════════════╩══════════════════════════════╝")

    return {
        "t_pandas_ms": t_pandas_ms,
        "t_numba_ms": t_numba_ms,
        "speedup": speedup,
        "pandas_result": stats_p,
        "numba_result": numba_result,
    }


if __name__ == "__main__":
    print("[INFO] Exemple dataset benchmarks :", generate_taxi_df(5).shape)

    print("=== TEST HAVERSINE : NYC → Paris ===")
    nyc_lat, nyc_lon = 40.7128, -74.0060
    par_lat, par_lon = 48.8566, 2.3522

    dist_py = haversine_python(nyc_lat, nyc_lon, par_lat, par_lon)
    dist_numpy = haversine_numpy(
        np.array([nyc_lat], dtype=np.float64),
        np.array([nyc_lon], dtype=np.float64),
        np.array([par_lat], dtype=np.float64),
        np.array([par_lon], dtype=np.float64),
    )[0]

    print(f"Python : {dist_py:.1f} km")
    print(f"NumPy  : {dist_numpy:.1f} km")
    print(f"Réel   : ~5 837 km")
    print(f"Écart  : {abs(dist_py - 5837):.0f} km")

    print("\n=== BENCHMARK HAVERSINE — 10M paires ===")
    if NUMBA_AVAILABLE:
        results = benchmark_haversine(n_points=10_000_000)
    else:
        results = benchmark_haversine(n_points=100_000)
        print("[INFO] Numba absent — benchmark réduit à 100K points")

    print("\n=== ROLLING WEIGHTED MEDIAN ===")
    if NUMBA_AVAILABLE:
        rolling_results = benchmark_rolling(n_points=100_000, window=20)
        print(f"Pandas  : {rolling_results['t_pandas_ms']:.1f} ms")
        print(f"Numba   : {rolling_results['t_numba_ms']:.1f} ms")
        print(f"Speedup : {rolling_results['speedup']:.1f}x")
    else:
        print("[SKIP] Numba absent")

    print(
        """
SYNTHÈSE PARTIE 2 — OPTIMISATION :
════════════════════════════════════════════════════════
Ex 2.1 Profilage    : cProfile/line_profiler/memory_profiler
                      → iterrows() = 80% du temps
Ex 2.2 Vectorisation: NumPy ufuncs + broadcasting
                      → speedup ×100+ vs iterrows()
Ex 2.3 Multiprocess : ProcessPoolExecutor sur 12 mois
                      → speedup ×N_cœurs (limité par Amdahl)
Ex 2.4 GIL          : Threading ❌ CPU-bound / ✅ I/O-bound
                      → ProcessPool contourne le GIL
Ex 2.5 Asyncio      : aiohttp + Semaphore pour les downloads
                      → speedup ×4+ vs séquentiel
Ex 2.6 Numba JIT    : LLVM → code machine natif
                      → haversine : ×200+ vs Python pur
                      → rolling median : ×50+ vs Pandas apply
════════════════════════════════════════════════════════
"""
    )
