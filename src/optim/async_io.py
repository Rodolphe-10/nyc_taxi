"""Téléchargements NYC Taxi : séquentiel (requests) vs asyncio + aiohttp.

Pour l'I/O réseau, threading et asyncio permettent d'occuper le CPU pendant
l'attente : le GIL est relâché pendant les appels bloquants (threading) ou on
n'utilise qu'un seul thread avec une boucle d'événements qui multiplexe
beaucoup de sockets (asyncio), souvent avec moins de mémoire que des dizaines
de threads. asyncio + aiohttp est adapté aux nombreuses connexions HTTP en
parallèle ; threading reste pertinent pour du code synchrone déjà écrit.
"""

from __future__ import annotations

import asyncio
import hashlib  # noqa: F401 — checksums / intégrité (extensions futures)
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp
import requests
from requests.exceptions import Timeout as RequestsTimeout
from tqdm import tqdm

try:
    from tqdm.asyncio import tqdm as async_tqdm

    TQDM_ASYNC = True
except ImportError:
    async_tqdm = None  # type: ignore[misc, assignment]
    TQDM_ASYNC = False

from src.core.config import Config
from src.core.decorators import retry

logger = logging.getLogger(__name__)

BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
FILENAME_PAT = "yellow_tripdata_{year}-{month:02d}.parquet"
CHUNK_SIZE = 1024 * 1024  # 1 Mo par chunk de téléchargement
FILESET_TAG = hashlib.sha256(BASE_URL.encode("utf-8")).hexdigest()[:12]


def get_download_url(year: int, month: int) -> str:
    """URL complète d'un fichier trip-data."""
    return f"{BASE_URL}/{FILENAME_PAT.format(year=year, month=month)}"


def get_dest_path(year: int, month: int, dest_dir: str) -> Path:
    """Chemin local du fichier Parquet."""
    return Path(dest_dir) / FILENAME_PAT.format(year=year, month=month)


def file_already_exists(path: Path, min_size_mb: float = 10.0) -> bool:
    """True si le fichier existe et dépasse une taille minimale (évite re-téléchargement)."""
    return path.exists() and path.stat().st_size >= min_size_mb * 1e6


@retry(
    max_attempts=3,
    backoff=1.5,
    exceptions=(ConnectionError, TimeoutError, OSError, RequestsTimeout),
)
def _requests_get_stream(url: str, timeout: int = 60):
    """GET HTTP en streaming ; retry uniquement sur erreurs réseau/transitoires."""
    return requests.get(url, stream=True, timeout=timeout)


def download_month_sync(year: int, month: int, dest_dir: str = "data/raw") -> Dict:
    """Télécharge un mois avec requests (référence synchrone)."""
    path = get_dest_path(year, month, dest_dir)
    if file_already_exists(path):
        size_mb = path.stat().st_size / 1e6
        return {
            "month": month,
            "year": year,
            "path": str(path),
            "size_mb": size_mb,
            "elapsed_s": 0.0,
            "speed_mbps": 0.0,
            "skipped": True,
            "success": True,
        }

    url = get_download_url(year, month)
    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    try:
        t0 = time.perf_counter()
        response = _requests_get_stream(url, timeout=60)
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        with tqdm(total=total, unit="B", unit_scale=True, desc=f"{year}-{month:02d}") as pbar:
            with open(path, "wb") as f:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        elapsed = time.perf_counter() - t0
        size_mb = path.stat().st_size / 1e6
        speed = size_mb / elapsed if elapsed > 0 else 0.0
        return {
            "month": month,
            "year": year,
            "path": str(path),
            "size_mb": size_mb,
            "elapsed_s": elapsed,
            "speed_mbps": speed,
            "skipped": False,
            "success": True,
        }
    except Exception as e:  # pragma: no cover - chemins d'erreur réseau variés
        return {"month": month, "success": False, "error": str(e), "skipped": False}


async def download_month_async(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    year: int,
    month: int,
    dest_dir: str = "data/raw",
) -> Dict:
    """Télécharge un mois en asyncio (aiohttp), concurrence bornée par ``semaphore``."""
    path = get_dest_path(year, month, dest_dir)
    if file_already_exists(path):
        size_mb = path.stat().st_size / 1e6
        return {
            "month": month,
            "year": year,
            "path": str(path),
            "size_mb": size_mb,
            "elapsed_s": 0.0,
            "speed_mbps": 0.0,
            "skipped": True,
            "success": True,
        }

    url = get_download_url(year, month)
    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    async with semaphore:
        try:
            t0 = time.perf_counter()
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                resp.raise_for_status()
                with open(path, "wb") as f:
                    async for chunk in resp.content.iter_chunked(CHUNK_SIZE):
                        f.write(chunk)

            elapsed = time.perf_counter() - t0
            size_mb = path.stat().st_size / 1e6
            speed = size_mb / elapsed if elapsed > 0 else 0.0
            logger.info(
                "[ASYNC] %s-%02d téléchargé : %.1f Mo en %.1fs (%.1f Mo/s)",
                year,
                month,
                size_mb,
                elapsed,
                speed,
            )
            return {
                "month": month,
                "year": year,
                "path": str(path),
                "size_mb": size_mb,
                "elapsed_s": elapsed,
                "speed_mbps": speed,
                "skipped": False,
                "success": True,
            }
        except Exception as e:  # pragma: no cover
            return {"month": month, "success": False, "error": str(e), "skipped": False}


async def download_year_async(
    year: int = 2023,
    dest_dir: str = "data/raw",
    months: Optional[List[int]] = None,
    concurrency: int = 4,
) -> List[Dict]:
    """Lance tous les téléchargements en parallèle (asyncio.gather)."""
    months = months or list(range(1, 13))
    semaphore = asyncio.Semaphore(concurrency)

    print(
        f"[ASYNC] Téléchargement {len(months)} fichiers | concurrence={concurrency}"
    )

    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            download_month_async(session, semaphore, year, m, dest_dir) for m in months
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    processed: List[Dict] = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            processed.append(
                {"month": months[i], "success": False, "error": str(r), "skipped": False}
            )
        else:
            processed.append(r)

    success = sum(1 for r in processed if r.get("success"))
    skipped = sum(1 for r in processed if r.get("skipped"))
    failed = sum(1 for r in processed if not r.get("success"))
    total_mb = sum(r.get("size_mb", 0) for r in processed)

    print(
        f"""
╔══════════════════════════════════════════╗
║   TÉLÉCHARGEMENT ASYNCHRONE TERMINÉ      ║
╠══════════════════════════════════════════╣
║  Succès  : {success:<26}║
║  Ignorés : {skipped} (déjà présents)     ║
║  Échecs  : {failed:<26}║
║  Total   : {total_mb:.1f} Mo             ║
╚══════════════════════════════════════════╝
"""
    )

    return processed


def download_year_sync(
    year: int = 2023,
    dest_dir: str = "data/raw",
    months: Optional[List[int]] = None,
) -> List[Dict]:
    """Télécharge les mois un par un (référence pour benchmark)."""
    months = months or list(range(1, 13))
    results: List[Dict] = []
    t_global = time.perf_counter()

    for month in months:
        result = download_month_sync(year, month, dest_dir)
        results.append(result)
        if result["success"] and not result.get("skipped"):
            print(
                f"[SYNC] {year}-{month:02d} : {result['size_mb']:.1f} Mo "
                f"en {result['elapsed_s']:.1f}s"
            )

    elapsed_total = time.perf_counter() - t_global
    print(f"[SYNC] Total : {elapsed_total:.1f}s")
    return results


def benchmark_async_vs_sync(
    year: int = 2023,
    months: Optional[List[int]] = None,
    dest_dir: str = "data/raw",
    concurrency: int = 4,
) -> Dict:
    """Compare le temps total synchrone vs asynchrone."""
    months = months or list(range(1, 13))

    existing = [
        m for m in months if file_already_exists(get_dest_path(year, m, dest_dir))
    ]
    if len(existing) == len(months):
        print("[INFO] Tous les fichiers sont déjà téléchargés.")
        print("[INFO] Supprimez data/raw/*.parquet pour relancer le benchmark.")
        return {"all_skipped": True, "existing_months": existing}

    print("\n[BENCHMARK] Téléchargement SYNCHRONE...")
    t0 = time.perf_counter()
    results_sync = download_year_sync(year, dest_dir, months)
    t_sync = time.perf_counter() - t0

    print("\n[BENCHMARK] Téléchargement ASYNCHRONE...")
    t0 = time.perf_counter()
    results_async = asyncio.run(
        download_year_async(year, dest_dir, months, concurrency)
    )
    t_async = time.perf_counter() - t0

    speedup = t_sync / t_async if t_async > 0 else float("inf")

    total_mb_sync = sum(r.get("size_mb", 0) for r in results_sync if not r.get("skipped"))
    total_mb_async = sum(r.get("size_mb", 0) for r in results_async if not r.get("skipped"))
    debit_sync = total_mb_sync / t_sync if t_sync > 0 else 0.0
    debit_async = total_mb_async / t_async if t_async > 0 else 0.0

    print("╔══════════════════════════════════════════════════════╗")
    print("║     BENCHMARK : SYNC vs ASYNC                        ║")
    print("╠══════════════╦═══════════╦══════════════════════════╣")
    print("║   Mode       ║  Temps    ║  Débit moyen             ║")
    print("╠══════════════╬═══════════╬══════════════════════════╣")
    print(f"║ Synchrone    ║  {t_sync:>6.1f}s   ║  {debit_sync:.1f} Mo/s                ║")
    print(f"║ Asynchrone   ║  {t_async:>6.1f}s   ║  {debit_async:.1f} Mo/s                ║")
    print(f"║ Speedup      ║  {speedup:>6.1f}x   ║                          ║")
    print("╚══════════════╩═══════════╩══════════════════════════╝")

    return {
        "t_sync": t_sync,
        "t_async": t_async,
        "speedup": speedup,
        "results_sync": results_sync,
        "results_async": results_async,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"[INFO] Config raw_data_path = {Config().raw_data_path}")
    print(f"[INFO] Miroir trip-data (tag) = {FILESET_TAG}")
    print(f"[INFO] tqdm.asyncio disponible = {TQDM_ASYNC} | async_tqdm = {async_tqdm!r}")

    print("=== TEST : Téléchargement mois de janvier 2023 ===")
    results = asyncio.run(
        download_year_async(
            year=2023,
            dest_dir="data/raw",
            months=[1],
            concurrency=1,
        )
    )
    for r in results:
        if r.get("success"):
            if r.get("skipped"):
                print("[SKIP] Janvier 2023 déjà présent")
            else:
                print(
                    f"[OK] Janvier 2023 : {r['size_mb']:.1f} Mo "
                    f"en {r['elapsed_s']:.1f}s"
                )
        else:
            print(f"[ERREUR] {r.get('error')}")

    if "--all" in sys.argv:
        print("\n=== TÉLÉCHARGEMENT COMPLET 2023 (12 mois) ===")
        asyncio.run(
            download_year_async(
                year=2023,
                dest_dir="data/raw",
                months=list(range(1, 13)),
                concurrency=4,
            )
        )

    if "--benchmark" in sys.argv:
        print("\n=== BENCHMARK SYNC vs ASYNC ===")
        benchmark_async_vs_sync(
            year=2023,
            months=[1, 2],
            dest_dir="data/raw_benchmark_test",
            concurrency=2,
        )
