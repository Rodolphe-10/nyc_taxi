"""Dask : graphes lazy, partitions et parallélisme pour données volumineuses NYC Taxi.

Contrairement à Pandas (exécution eager, tout en RAM), Dask construit un graphe
de tâches : aucun calcul lourd n'a lieu tant qu'on n'appelle pas ``.compute()``.
L'ordonnanceur peut alors paralléliser sur plusieurs cœurs ou un cluster.

**Quand préférer Dask à Pandas ?**
- Données plus grandes que la RAM (spill disque, partitions).
- Lecture / agrégation de nombreux fichiers Parquet en parallèle.
- Cluster multi-nœuds (``distributed``).

**Quand Pandas suffit ?**
- Jeu entier tient en mémoire, latence faible souhaitée.
- Très petits jeux : l'overhead du graphe Dask peut dominer.
"""

from __future__ import annotations

import logging
import os
import time  # noqa: F401 — cahier des charges 3.4
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.core.context import Timer
from src.optim.benchmarks import generate_taxi_df
from src.optim.memory import optimize_dtypes

try:
    import dask
    import dask.dataframe as dd
    from dask.distributed import Client, LocalCluster

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    logging.warning("Dask absent — pip install dask[complete]")

logger = logging.getLogger(__name__)

_BENCHMARK_RESULT_KEYS: List[str] = [
    "t_pandas",
    "t_dask",
    "speedup_dask",
    "results_equal",
    "n_rows",
]


def create_dask_client(
    n_workers: Optional[int] = None,
    threads_per_worker: int = 2,
    memory_limit: str = "2GB",
) -> Optional[object]:
    """Crée un ``Client`` Dask local (dashboard) ; ``None`` si Dask indisponible."""
    if not DASK_AVAILABLE:
        logger.warning("Dask absent — client non créé")
        return None

    n_workers = n_workers or (os.cpu_count() or 1)
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
    )
    client = Client(cluster)
    logger.info("Dask client créé : %s workers", n_workers)
    logger.info("Dashboard : %s", getattr(client, "dashboard_link", None))
    return client


def load_parquets_dask(raw_dir: str = "data/raw", pattern: str = "*.parquet"):
    """Lit les Parquet sous ``raw_dir`` en Dask DataFrame lazy (glob)."""
    if not DASK_AVAILABLE:
        return None

    import dask.dataframe as dd

    path_pattern = str(Path(raw_dir) / pattern)
    ddf = dd.read_parquet(path_pattern, engine="pyarrow")
    logger.info("Dask DataFrame créé (lazy) : %s partitions", ddf.npartitions)
    return ddf


def generate_dask_dataframe(n_rows: int = 500_000, n_partitions: int = 10):
    """Dask DataFrame simulé (lazy) ou DataFrame Pandas si Dask absent."""
    df_pandas = optimize_dtypes(generate_taxi_df(n_rows=n_rows), verbose=False)
    if not DASK_AVAILABLE:
        return df_pandas

    import dask.dataframe as dd

    ddf = dd.from_pandas(df_pandas, npartitions=n_partitions)
    logger.info(
        "Dask DataFrame simulé : %s lignes | %s partitions",
        f"{n_rows:,}",
        n_partitions,
    )
    return ddf


def compute_activity_dask(ddf) -> pd.DataFrame:
    """Même logique d'agrégation qu'en 3.2 (groupby zone × heure × jour), via Dask."""
    if not DASK_AVAILABLE:
        raise ImportError("Dask n'est pas installé")

    ddf = ddf.copy()
    ddf["pickup_hour"] = ddf["tpep_pickup_datetime"].dt.hour
    ddf["pickup_weekday"] = ddf["tpep_pickup_datetime"].dt.dayofweek
    ddf["tip_pct"] = ddf["tip_amount"] / ddf["fare_amount"].clip(lower=0.01)

    activity_dask = ddf.groupby(["PULocationID", "pickup_hour", "pickup_weekday"]).agg(
        {
            "fare_amount": ["count", "sum", "mean"],
            "tip_amount": "mean",
            "tip_pct": "mean",
            "trip_distance": "mean",
        }
    )

    with Timer("Dask compute()", unit="s") as t:
        result = activity_dask.compute()

    logger.info("Dask compute() terminé en %.2fs | %s", t.elapsed, result.shape)
    return result


def _normalize_activity_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Aplatit les colonnes MultiIndex pour comparaison Pandas / Dask."""
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = ["_".join(str(c) for c in col if str(c) != "") for col in out.columns.values]
    return out.sort_index()


def benchmark_pandas_vs_dask(n_rows: int = 500_000) -> Dict:
    """Compare un groupby–agrégation identique en Pandas (eager) et Dask (lazy + compute)."""
    df_pandas = generate_taxi_df(n_rows=n_rows)

    with Timer("Pandas groupby", unit="s") as t_pd:
        df_pd = df_pandas.copy()
        df_pd["pickup_hour"] = df_pd["tpep_pickup_datetime"].dt.hour
        df_pd["pickup_weekday"] = df_pd["tpep_pickup_datetime"].dt.dayofweek
        df_pd["tip_pct"] = df_pd["tip_amount"] / df_pd["fare_amount"].clip(lower=0.01)
        result_pandas = df_pd.groupby(
            ["PULocationID", "pickup_hour", "pickup_weekday"]
        ).agg(
            {
                "fare_amount": ["count", "sum", "mean"],
                "tip_amount": "mean",
                "tip_pct": "mean",
                "trip_distance": "mean",
            }
        )
    t_pandas = t_pd.elapsed

    t_dask: Optional[float] = None
    result_dask = None
    if DASK_AVAILABLE:
        import dask.dataframe as dd

        ddf = dd.from_pandas(df_pandas, npartitions=max(1, os.cpu_count() or 1))
        with Timer("Dask groupby + compute()", unit="s") as t_dk:
            ddf_prep = ddf.copy()
            ddf_prep["pickup_hour"] = ddf_prep["tpep_pickup_datetime"].dt.hour
            ddf_prep["pickup_weekday"] = ddf_prep["tpep_pickup_datetime"].dt.dayofweek
            ddf_prep["tip_pct"] = (
                ddf_prep["tip_amount"] / ddf_prep["fare_amount"].clip(lower=0.01)
            )
            result_dask = (
                ddf_prep.groupby(["PULocationID", "pickup_hour", "pickup_weekday"])
                .agg(
                    {
                        "fare_amount": ["count", "sum", "mean"],
                        "tip_amount": "mean",
                        "tip_pct": "mean",
                        "trip_distance": "mean",
                    }
                )
                .compute()
            )
        t_dask = t_dk.elapsed

    results_equal: Optional[bool] = None
    if DASK_AVAILABLE and result_dask is not None:
        rp = _normalize_activity_columns(result_pandas)
        rd = _normalize_activity_columns(result_dask)
        rd = rd.reindex(columns=rp.columns)
        try:
            pd.testing.assert_frame_equal(
                rp,
                rd,
                check_dtype=False,
                atol=1e-3,
            )
            results_equal = True
        except AssertionError:
            results_equal = False

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     BENCHMARK : PANDAS vs DASK                               ║")
    print(f"║     Dataset : {n_rows:,} lignes                              ║")
    print("╠═══════════════╦══════════════╦═══════════════════════════════╣")
    print("║ Framework     ║ Temps (s)    ║ Commentaire                   ║")
    print("╠═══════════════╬══════════════╬═══════════════════════════════╣")
    print(f"║ Pandas        ║    {t_pandas:>5.2f}s   ║ Tout en mémoire               ║")
    if t_dask is not None:
        print(f"║ Dask          ║    {t_dask:>5.2f}s   ║ Lazy + parallèle              ║")
    else:
        print("║ Dask          ║     n/a      ║ (non installé)                ║")
    print("╚═══════════════╩══════════════╩═══════════════════════════════╝")
    print(f"Résultats identiques : {results_equal}")

    print(
        f"""
QUAND DASK EST PLUS RAPIDE QUE PANDAS ?
────────────────────────────────────────────────────────────
✅ Dask gagne quand :
   - Dataset > RAM disponible (spilling sur disque)
   - Calculs sur 12 fichiers Parquet en parallèle
   - Cluster multi-nœuds (Dask Distributed)
   - Opérations embarrassingly parallel (groupby indépendants)

❌ Pandas gagne quand :
   - Dataset tient en RAM
   - Opérations séquentielles complexes
   - Overhead du graphe de calcul Dask domine
   - Petits datasets (< 100K lignes)

Sur {n_rows:,} lignes : Pandas est probablement plus rapide
car l'overhead du graphe Dask dépasse le gain du parallélisme.
Dask devient avantageux à partir de ~10M lignes ou plusieurs fichiers.
"""
    )

    speedup = (t_pandas / t_dask) if (t_dask is not None and t_dask > 0) else None

    out: Dict[str, object] = {
        "t_pandas": t_pandas,
        "t_dask": t_dask,
        "speedup_dask": speedup,
        "results_equal": results_equal,
        "n_rows": n_rows,
    }
    assert set(out.keys()) == set(_BENCHMARK_RESULT_KEYS)
    return out


def optimize_dask_pipeline(ddf, output_dir: str = "data/features") -> None:
    """Pipeline lazy (filtres + features) puis écriture Parquet partitionnée."""
    if not DASK_AVAILABLE:
        return

    os.makedirs(output_dir, exist_ok=True)

    ddf_processed = (
        ddf.assign(
            pickup_hour=ddf["tpep_pickup_datetime"].dt.hour,
            pickup_weekday=ddf["tpep_pickup_datetime"].dt.dayofweek,
            tip_pct=ddf["tip_amount"] / ddf["fare_amount"].clip(lower=0.01),
            duration_min=(
                ddf["tpep_dropoff_datetime"] - ddf["tpep_pickup_datetime"]
            ).dt.total_seconds()
            / 60.0,
        )
        .query("fare_amount >= 2.5 and fare_amount <= 200")
        .query("trip_distance >= 0.1 and trip_distance <= 100")
    )

    output_path = str(Path(output_dir) / "processed_dask")
    with Timer("Dask write_parquet", unit="s") as t:
        ddf_processed.to_parquet(
            output_path,
            engine="pyarrow",
            write_index=False,
            overwrite=True,
        )

    logger.info("Pipeline Dask écrit en %.2fs → %s", t.elapsed, output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== EXERCICE 3.4 — DASK POUR DATASETS VOLUMINEUX ===\n")

    if not DASK_AVAILABLE:
        print("[WARN] Dask non installé. pip install dask[complete]")
        print("[INFO] Exécution du benchmark Pandas seul...")

    print("--- A. Benchmark Pandas vs Dask ---")
    results = benchmark_pandas_vs_dask(n_rows=500_000)
    print("\nRésumé :")
    print(f"  Pandas : {results['t_pandas']:.2f}s")
    if results["t_dask"] is not None:
        print(f"  Dask   : {results['t_dask']:.2f}s")
        if results["speedup_dask"] is not None:
            print(f"  Speedup: {results['speedup_dask']:.2f}x")
    print(f"  Égalité: {results['results_equal']}")

    if DASK_AVAILABLE:
        print("\n--- B. Pipeline Dask complet ---")
        ddf = generate_dask_dataframe(n_rows=200_000, n_partitions=4)
        print(f"Dask DataFrame : {ddf.npartitions} partitions")
        print(f"Colonnes : {list(ddf.columns)}")

        print("\nGraphe de tâches (lazy) :")
        activity = compute_activity_dask(ddf)
        print(f"Résultat compute() : {activity.shape}")

        print("\n--- C. Client Dask ---")
        client = create_dask_client(n_workers=2, memory_limit="1GB")
        if client is not None:
            print(f"Dashboard : {client.dashboard_link}")
            print("→ Ouvre http://localhost:8787 pour voir le dashboard")
            client.close()
