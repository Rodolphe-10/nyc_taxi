"""Traitement parallèle des mois NYC Taxi et benchmark de speedup."""

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, List

import numpy as np
import pandas as pd

from src.optim.benchmarks import generate_taxi_df
from src.optim.vectorize import compute_features_vectorized, optimize_dtypes


def cpu_bound_task(size: int) -> float:
    """Tâche CPU-bound : somme de sin(x) sur un vecteur NumPy (module-level, picklable)."""
    x = np.linspace(0, np.pi * 10, size)
    # NumPy relâche le GIL pendant ses opérations C internes
    # mais la coordination des threads reste coûteuse
    # Pour une tâche 100% Python (boucle), le GIL bloquerait totalement
    return float(np.sin(x).sum())


def cpu_bound_task_pure_python(size: int) -> float:
    """Tâche CPU-bound en Python pur (sans NumPy) — montre le vrai blocage GIL."""
    total = 0.0
    for i in range(size):
        total += math.sin(i * math.pi / size)
    # Chaque itération Python acquiert/relâche le GIL
    # ThreadPoolExecutor ne peut pas paralléliser cela
    # ProcessPoolExecutor contourne le problème via des processus séparés
    return total


def io_bound_task(path: str) -> Dict:
    """Lit un Parquet ou simule l'I/O ; retourne stats (module-level, picklable)."""
    t0 = time.perf_counter()
    if os.path.exists(path):
        df = pd.read_parquet(path)
    else:
        time.sleep(0.1)
        df = generate_taxi_df(n_rows=10_000)
    elapsed = time.perf_counter() - t0
    return {"path": path, "n_rows": len(df), "elapsed_ms": elapsed * 1000}


# Worker module-level : requis sur Windows pour que ProcessPool puisse pickler la fonction.
def process_one_month(args: tuple) -> Dict:
    """Traite un mois (chargement ou génération, features, optimisation, sauvegarde).

    Tuple accepté :
    ``(month, n_rows, seed)``,
    ``(month, n_rows, seed, persist_outputs)`` ou
    ``(month, n_rows, seed, persist_outputs, use_real)``.

    ``n_rows`` est toujours respecté. Le Parquet de ``data/raw`` n'est lu que si
    ``use_real`` est vrai et que le fichier existe ; sinon un jeu simulé de
    ``n_rows`` lignes est généré.
    """
    if len(args) >= 4:
        month, n_rows, seed, persist_outputs = args[:4]
    else:
        month, n_rows, seed = args[:3]
        persist_outputs = True
    use_real = bool(args[4]) if len(args) >= 5 else False
    try:
        t0 = time.perf_counter()
        raw_path = f"data/raw/yellow_tripdata_2023-{month:02d}.parquet"
        output_path = f"data/processed/yellow_2023_{month:02d}_processed.parquet"
        os.makedirs("data/processed", exist_ok=True)

        if use_real and os.path.exists(raw_path):
            df = pd.read_parquet(raw_path)
        else:
            df = generate_taxi_df(n_rows=n_rows, seed=seed)

        df = compute_features_vectorized(df)
        df = optimize_dtypes(df, verbose=False)
        if persist_outputs:
            df.to_parquet(output_path, index=False)

        duration_ms = (time.perf_counter() - t0) * 1000
        return {
            "month": month,
            "n_rows": len(df),
            "duration_ms": duration_ms,
            "output_path": output_path,
            "pid": os.getpid(),
            "success": True,
        }
    except Exception as e:  # pragma: no cover - chemin d'échec
        return {
            "month": month,
            "success": False,
            "error": str(e),
        }


# Référence séquentielle pour mesurer le gain du parallélisme.
def process_months_sequential(
    months: List[int],
    n_rows: int = 50_000,
    persist_outputs: bool = True,
    use_real: bool = False,
) -> tuple:
    """Exécute les mois l'un après l'autre dans un seul processus."""
    t0 = time.perf_counter()
    results = []
    for month in months:
        result = process_one_month(
            (month, n_rows, month * 42, persist_outputs, use_real)
        )
        results.append(result)
        print(f"[SEQ] Mois {month}/12 traité")
    elapsed_total_s = time.perf_counter() - t0
    return results, elapsed_total_s


# Exécution parallèle via ProcessPool : chaque mois est indépendant.
def process_months_parallel(
    months: List[int],
    n_rows: int = 50_000,
    max_workers: int = None,
    persist_outputs: bool = True,
    use_real: bool = False,
) -> tuple:
    """Distribue les mois sur plusieurs processus CPU."""
    max_workers = max_workers or os.cpu_count()
    print(f"[PAR] {len(months)} mois | {max_workers} workers | {os.cpu_count()} CPU")

    t0 = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_one_month,
                (month, n_rows, month * 42, persist_outputs, use_real),
            )
            for month in months
        ]
        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            if result.get("success"):
                print(
                    f"[PAR] Mois {result['month']} terminé ({result['duration_ms']:.0f}ms) — PID {result['pid']}"
                )
            else:
                print(f"[PAR] Mois {result.get('month')} en échec: {result.get('error')}")

    elapsed_total_s = time.perf_counter() - t0
    return results, elapsed_total_s


# Courbe de speedup : variation du nombre de workers et analyse de l'efficacité.
def speedup_benchmark(
    months: List[int] = None,
    n_rows: int = 20_000,
    worker_counts: List[int] = None,
    use_real: bool = False,
) -> pd.DataFrame:
    """Calcule speedup/efficacité selon le nombre de workers."""
    months = months or list(range(1, 13))
    cpu_count = os.cpu_count() or 1
    worker_counts = worker_counts or [1, 2, 4, cpu_count]
    worker_counts = sorted(set(w for w in worker_counts if w <= cpu_count))

    records = []
    for n_workers in worker_counts:
        _, elapsed_s = process_months_parallel(
            months, n_rows, n_workers, use_real=use_real
        )
        records.append({"n_workers": n_workers, "elapsed_s": elapsed_s})

    df = pd.DataFrame(records)
    ref = float(df.loc[df["n_workers"] == 1, "elapsed_s"].iloc[0]) if (df["n_workers"] == 1).any() else float(df["elapsed_s"].iloc[0])
    df["speedup"] = ref / df["elapsed_s"]
    df["efficiency_pct"] = df["speedup"] / df["n_workers"] * 100

    print("╔═══════════════════════════════════════════════════════╗")
    print("║      COURBE DE SPEEDUP — ProcessPoolExecutor          ║")
    print("╠══════════╦════════════╦═══════════╦════════════════╣")
    for _, row in df.iterrows():
        print(
            f"║ {int(row['n_workers']):>8} ║ {row['elapsed_s']:>8.2f}s ║ {row['speedup']:>7.2f}x ║ {row['efficiency_pct']:>12.1f}% ║"
        )
    print("╚══════════╩════════════╩═══════════╩════════════════╝")

    print(
        """
ANALYSE DE LA LOI D'AMDAHL :
L'efficacité décroît avec le nombre de workers car :
1. Overhead pickle : sérialisation des données entre processus
2. Overhead fork   : création des sous-processus OS
3. I/O contention  : plusieurs processus écrivent simultanément
4. Partie séquentielle : coordination, collecte des résultats

Formule d'Amdahl : Speedup = 1 / (S + (1-S)/N)
où S = fraction séquentielle, N = nombre de workers
"""
    )

    return df[["n_workers", "elapsed_s", "speedup", "efficiency_pct"]]


def benchmark_gil_cpu_bound(n_workers: int = 4, size: int = 2_000_000) -> Dict:
    """Compare séquentiel / threading / processus sur une tâche CPU Python pur.

    Sur Linux/macOS, ``fork`` hérite du processus parent (démarrage rapide).
    Sur Windows, ``spawn`` relance l'interpréteur dans chaque worker : overhead
    notable pour de petites tâches ; utiliser ``size`` assez grand pour que le
    calcul domine (voir tests).
    """
    tasks = [size] * n_workers

    t0 = time.perf_counter()
    results_seq = [cpu_bound_task_pure_python(s) for s in tasks]
    t_seq = time.perf_counter() - t0

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        results_thr = list(ex.map(cpu_bound_task_pure_python, tasks))
    t_thr = time.perf_counter() - t0

    # Même pool : warmup puis chrono — sur Windows (spawn), évite que le démarrage
    # des processus gonfle artificiellement t_multiprocessing.
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        list(ex.map(cpu_bound_task_pure_python, [max(1, size // 100)] * n_workers))
        t0 = time.perf_counter()
        results_pro = list(ex.map(cpu_bound_task_pure_python, tasks))
    t_pro = time.perf_counter() - t0

    assert len(results_seq) == len(results_thr) == len(results_pro) == n_workers

    speedup_thr = t_seq / t_thr if t_thr > 0 else float("inf")
    speedup_pro = t_seq / t_pro if t_pro > 0 else float("inf")
    gil_overhead_pct = (t_thr - t_seq) / t_seq * 100 if t_seq > 0 else 0.0

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         GIL — TÂCHE CPU-BOUND (Python pur)                   ║")
    _line_workers = f"         {n_workers} workers | {size:,} itérations/tâche      "
    print("║" + _line_workers[:62].ljust(62) + "║")
    print("╠══════════════╦══════════════╦═════════════╦════════════════╣")
    print("║   Mode       ║   Temps (s)  ║   Speedup   ║  Commentaire   ║")
    print("╠══════════════╬══════════════╬═════════════╬════════════════╣")
    print(f"║ Séquentiel   ║    {t_seq:>5.2f}s    ║    1.00x    ║  référence     ║")
    print(f"║ Threading    ║    {t_thr:>5.2f}s    ║    {speedup_thr:>4.2f}x    ║  GIL bloque !  ║")
    print(f"║ Processes    ║    {t_pro:>5.2f}s    ║    {speedup_pro:>4.2f}x    ║  vrai parallél ║")
    print("╚══════════════╩══════════════╩═════════════╩════════════════╝")

    return {
        "t_seq": t_seq,
        "t_threading": t_thr,
        "t_multiprocessing": t_pro,
        "speedup_threading": speedup_thr,
        "speedup_processes": speedup_pro,
        "gil_overhead_pct": gil_overhead_pct,
        "n_workers": n_workers,
    }


def benchmark_gil_io_bound(n_files: int = 8) -> Dict:
    """Compare séquentiel vs ThreadPool sur chemins Parquet (ou sleep simulé)."""
    paths = [
        f"data/raw/yellow_tripdata_2023-{m:02d}.parquet" for m in range(1, n_files + 1)
    ]

    t0 = time.perf_counter()
    results_seq = [io_bound_task(p) for p in paths]
    t_seq = time.perf_counter() - t0

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_files) as ex:
        results_thr = list(ex.map(io_bound_task, paths))
    t_thr = time.perf_counter() - t0

    speedup = t_seq / t_thr if t_thr > 0 else float("inf")

    print("╔══════════════════════════════════════════════════════╗")
    print("║         GIL — TÂCHE I/O-BOUND                        ║")
    print("╠══════════════╦══════════════╦════════════════════════╣")
    print("║   Mode       ║   Temps (s)  ║   Speedup              ║")
    print("╠══════════════╬══════════════╬════════════════════════╣")
    print(f"║ Séquentiel   ║    {t_seq:>5.2f}s    ║    1.00x               ║")
    print(f"║ Threading    ║    {t_thr:>5.2f}s    ║    {speedup:>5.2f}x  ← GIL relâché║")
    print("╚══════════════╩══════════════╩════════════════════════╝")

    return {
        "t_seq": t_seq,
        "t_threading": t_thr,
        "speedup_threading": speedup,
        "n_files": n_files,
    }


def generate_gil_summary_table() -> pd.DataFrame:
    """Tableau pédagogique : quand threading / multiprocessing / asyncio."""
    df = pd.DataFrame(
        {
            "Type de tâche": [
                "CPU-bound",
                "I/O bloquant",
                "I/O non bloquant",
            ],
            "Threading": [
                "❌ GIL bloque",
                "✅ GIL relâché",
                "⚠️ Possible",
            ],
            "Multiprocessing": [
                "✅ Vrai parallél",
                "✅ Fonctionne",
                "⚠️ Overhead",
            ],
            "Asyncio": [
                "❌ Inutile",
                "✅ Optimal",
                "✅ Optimal",
            ],
        }
    )
    print(df.to_string(index=False))
    return df


# Démonstration empirique GIL : threading vs multiprocessing sur CPU-bound.
def threading_vs_multiprocessing_cpu_bound(n: int = 5_000_000) -> Dict:
    """Compare séquentiel/threading/processes sur une charge CPU (Python pur, GIL)."""
    workers = 4
    chunk = max(1, n // workers)

    t0 = time.perf_counter()
    _ = [cpu_bound_task_pure_python(chunk) for _ in range(workers)]
    sequential_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(cpu_bound_task_pure_python, [chunk] * workers))
    threading_s = time.perf_counter() - t0

    # Warmup : amortit le coût spawn Windows (ProcessPoolExecutor) avant chronométrage.
    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(cpu_bound_task_pure_python, [max(1, chunk // 20)] * workers))
        t0 = time.perf_counter()
        list(ex.map(cpu_bound_task_pure_python, [chunk] * workers))
        processes_s = time.perf_counter() - t0

    return {
        "sequential_s": sequential_s,
        "threading_s": threading_s,
        "processes_s": processes_s,
        "speedup_threading": sequential_s / threading_s if threading_s > 0 else float("inf"),
        "speedup_processes": sequential_s / processes_s if processes_s > 0 else float("inf"),
    }


if __name__ == "__main__":
    MONTHS = list(range(1, 13))
    N_ROWS = 20_000

    print("=== TRAITEMENT SÉQUENTIEL ===")
    _, t_seq = process_months_sequential(MONTHS, N_ROWS, use_real=True)
    print(f"Temps séquentiel : {t_seq:.2f}s")

    print("\n=== TRAITEMENT PARALLÈLE ===")
    _, t_par = process_months_parallel(MONTHS, N_ROWS, use_real=True)
    print(f"Temps parallèle  : {t_par:.2f}s")
    print(f"Speedup global   : {t_seq / t_par:.2f}x")

    print("\n=== COURBE DE SPEEDUP ===")
    df_speedup = speedup_benchmark(
        months=MONTHS,
        n_rows=N_ROWS,
        worker_counts=[1, 2, 4, os.cpu_count()],
        use_real=True,
    )
    print(df_speedup.to_string())

    print("\n=== GIL : THREADING vs MULTIPROCESSING ===")
    results = threading_vs_multiprocessing_cpu_bound()
    print(f"Séquentiel : {results['sequential_s']:.2f}s")
    print(
        f"Threading  : {results['threading_s']:.2f}s  "
        f"(speedup {results['speedup_threading']:.2f}x)"
    )
    print(
        f"Processes  : {results['processes_s']:.2f}s  "
        f"(speedup {results['speedup_processes']:.2f}x)"
    )
    print("\nConclusion : le threading N'AMÉLIORE PAS les tâches CPU-bound")
    print("car le GIL empêche l'exécution simultanée du bytecode Python.")

    print("\n=== EXERCICE 2.4 — GIL : THREADING vs MULTIPROCESSING ===\n")

    print("--- Tâche CPU-bound (Python pur) ---")
    results_cpu = benchmark_gil_cpu_bound(n_workers=4, size=500_000)
    print("\nConclusion CPU-bound :")
    print(f"  Threading  speedup : {results_cpu['speedup_threading']:.2f}x")
    print(f"  Processes  speedup : {results_cpu['speedup_processes']:.2f}x")
    print(f"  Overhead GIL       : {results_cpu['gil_overhead_pct']:+.1f}%")
    if results_cpu["speedup_threading"] < 1.2:
        print("  → Le threading N'AMÉLIORE PAS les tâches CPU-bound (GIL)")
    if results_cpu["speedup_processes"] > 1.5:
        print("  → ProcessPoolExecutor CONTOURNE le GIL efficacement")

    print("\n--- Tâche I/O-bound ---")
    results_io = benchmark_gil_io_bound(n_files=8)
    print("\nConclusion I/O-bound :")
    print(f"  Threading speedup : {results_io['speedup_threading']:.2f}x")
    if results_io["speedup_threading"] > 1.5:
        print("  → Le threading AIDE sur les tâches I/O-bound")
        print("  → Le GIL est relâché pendant les opérations I/O")

    print("\n--- TABLEAU DE SYNTHÈSE ---")
    generate_gil_summary_table()

    print(
        """
EXPLICATION DU GIL :
────────────────────────────────────────────────────────────
Le GIL (Global Interpreter Lock) est un mutex dans CPython
qui protège les structures de données internes de l'interpréteur.

POURQUOI EXISTE-T-IL ?
→ Simplifier la gestion mémoire (comptage de références)
→ Éviter les race conditions sur les objets Python

CONSÉQUENCES :
→ CPU-bound + Threading : les threads se battent pour le GIL
  Résultat : même temps qu'en séquentiel, voire plus lent
  (overhead de changement de contexte)
→ I/O-bound + Threading : le GIL est relâché pendant l'attente I/O
  Résultat : les autres threads peuvent s'exécuter → gain réel
→ ProcessPoolExecutor : chaque processus a son PROPRE GIL
  Résultat : vrai parallélisme CPU, mais overhead mémoire/pickle

QUAND UTILISER QUOI ?
┌─────────────────┬──────────────────┬──────────────────┐
│ Tâche           │ Solution optimale│ Raison           │
├─────────────────┼──────────────────┼──────────────────┤
│ CPU-bound       │ ProcessPool      │ Contourne le GIL │
│ I/O bloquant    │ ThreadPool       │ GIL relâché      │
│ I/O réseau      │ asyncio          │ Non-bloquant     │
│ Mixed           │ ProcessPool+async│ Combinaison      │
└─────────────────┴──────────────────┴──────────────────┘
"""
    )
