"""Gestionnaires de contexte pour instrumenter le pipeline."""

import contextlib
import logging
import os
import time
import tracemalloc
from typing import Dict, Generator, List

import pandas as pd
import psutil


# Gestionnaire de contexte pour mesurer la durée d'un bloc pipeline.
class Timer:
    """Mesure et journalise le temps d'exécution d'un bloc with."""

    def __init__(self, label: str = "", logger=None, unit="ms"):
        self.label = label
        self.unit = unit
        self._logger = logger or logging.getLogger("Timer")
        self.elapsed = None
        self.start_time = None

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        self._logger.debug("[Timer] '%s' démarré", self.label)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        raw = time.perf_counter() - self.start_time
        if self.unit == "ms":
            self.elapsed = raw * 1000
        elif self.unit == "s":
            self.elapsed = raw
        elif self.unit == "min":
            self.elapsed = raw / 60
        else:
            raise ValueError(f"Unité non supportée: {self.unit}")

        if exc_type is None:
            self._logger.info("[Timer] '%s' → %.3f %s", self.label, self.elapsed, self.unit)
        else:
            self._logger.error(
                "[Timer] '%s' interrompu après %.3f %s — %s: %s",
                self.label,
                self.elapsed,
                self.unit,
                exc_type.__name__,
                exc_val,
            )
        return False

    def __repr__(self):
        if self.elapsed is None:
            return f"<Timer '{self.label}' not started>"
        return f"<Timer '{self.label}' {self.elapsed:.3f} {self.unit}>"


# Gestionnaire de contexte pour surveiller la mémoire d'une étape.
class MemoryGuard:
    """Surveille le pic mémoire et protège contre les dépassements."""

    def __init__(self, threshold_mb: float = 2000.0, label: str = ""):
        self.threshold_mb = threshold_mb
        self.label = label
        self._logger = logging.getLogger("MemoryGuard")
        self.peak_mb = None
        self.start_mb = None

    def __enter__(self) -> "MemoryGuard":
        tracemalloc.start()
        process = psutil.Process(os.getpid())
        self.start_mb = process.memory_info().rss / 1e6
        self._logger.debug(
            "[MemoryGuard] '%s' — RAM initiale : %.1f Mo",
            self.label,
            self.start_mb,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.peak_mb = peak / 1e6
        delta_mb = self.peak_mb

        self._logger.info(
            "[MemoryGuard] '%s' — pic : %.1f Mo (seuil : %.1f Mo)",
            self.label,
            self.peak_mb,
            self.threshold_mb,
        )

        if exc_type is None and delta_mb > self.threshold_mb:
            raise MemoryError(
                f"[MemoryGuard] '{self.label}' a consommé {self.peak_mb:.1f} Mo "
                f"> seuil {self.threshold_mb:.1f} Mo"
            )
        return False


# Contexte utilitaire pour conversion temporaire de dtypes sur DataFrame.
@contextlib.contextmanager
def temp_dtypes(df: pd.DataFrame, mapping: Dict[str, str]) -> Generator:
    """
    Change temporairement les dtypes d'un DataFrame.
    Restaure les types d'origine même en cas d'exception.
    """
    original_dtypes = {col: str(df[col].dtype) for col in mapping if col in df.columns}

    for col, dtype in mapping.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)

    try:
        yield df
    finally:
        for col, dtype in original_dtypes.items():
            df[col] = df[col].astype(dtype)


# Contexte utilitaire pour ouvrir plusieurs parquets de manière sûre.
@contextlib.contextmanager
def open_parquets(directory: str, pattern: str = "*.parquet") -> Generator:
    """
    Ouvre tous les fichiers Parquet d'un dossier avec ExitStack.
    Garantit la fermeture des ressources même en cas d'échec.
    """
    import glob
    import pyarrow.parquet as pq

    logger = logging.getLogger("open_parquets")
    paths = sorted(glob.glob(os.path.join(directory, pattern)))
    if not paths:
        logger.warning("Aucun fichier parquet trouvé dans %s avec pattern %s", directory, pattern)
        yield []
        return

    dataframes: List[pd.DataFrame] = []
    with contextlib.ExitStack() as stack:
        for path in paths:
            pf = pq.ParquetFile(path)
            stack.callback(lambda parquet_file=pf: parquet_file.close())
            df = pf.read().to_pandas()
            dataframes.append(df)
            logger.info("Ouvert : %s — %s lignes", os.path.basename(path), len(df))
        yield dataframes


# Contexte composite pour instrumenter une étape complète de pipeline.
class PipelineContext:
    """Combine timing, mémoire et logs dans un seul with."""

    def __init__(
        self,
        step_name: str,
        memory_threshold_mb: float = 4000.0,
        time_unit: str = "ms",
    ):
        self.step_name = step_name
        self._timer = Timer(step_name, unit=time_unit)
        self._memory = MemoryGuard(memory_threshold_mb, step_name)
        self._logger = logging.getLogger("PipelineContext")
        self.elapsed = None
        self.peak_mb = None

    def __enter__(self) -> "PipelineContext":
        self._memory.__enter__()
        self._timer.__enter__()
        self._logger.info("[Pipeline] Début : '%s'", self.step_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._timer.__exit__(exc_type, exc_val, exc_tb)
        self._memory.__exit__(exc_type, exc_val, exc_tb)
        self.elapsed = self._timer.elapsed
        self.peak_mb = self._memory.peak_mb
        self._logger.info(
            "[Pipeline] Fin : '%s' | %.1f %s | %.1f Mo",
            self.step_name,
            self.elapsed,
            self._timer.unit,
            self.peak_mb,
        )
        return False


if __name__ == "__main__":
    # Section A — Timer.
    with Timer("Chargement parquet", unit="ms") as t:
        time.sleep(0.1)
    print(f"Durée : {t.elapsed:.1f} ms")
    print(repr(t))

    try:
        with Timer("Opération qui échoue") as t:
            time.sleep(0.05)
            raise ValueError("Erreur simulée")
    except ValueError:
        print(f"Timer capturé après erreur : {t.elapsed:.1f} ms")

    # Section B — MemoryGuard.
    with MemoryGuard(threshold_mb=10000.0, label="Test mémoire") as mg:
        data = list(range(1000000))
    print(f"Pic mémoire : {mg.peak_mb:.2f} Mo")

    try:
        with MemoryGuard(threshold_mb=0.001, label="Seuil minimal") as mg:
            data = list(range(100000))
    except MemoryError as e:
        print(f"[ATTENDU] {e}")

    # Section C — temp_dtypes.
    df = pd.DataFrame({"col_float": [1.0, 2.0, 3.0], "col_int": [100, 200, 300]})
    print(f"Avant : {df.dtypes.to_dict()}")
    with temp_dtypes(df, {"col_float": "float32", "col_int": "int16"}) as df_tmp:
        print(f"Pendant : {df_tmp.dtypes.to_dict()}")
    print(f"Après restauration : {df.dtypes.to_dict()}")

    # Section D — PipelineContext.
    with PipelineContext("feature_engineering", time_unit="ms") as ctx:
        time.sleep(0.05)
        data = [i**2 for i in range(100000)]
    print(f"Durée : {ctx.elapsed:.1f}ms | Mémoire : {ctx.peak_mb:.2f} Mo")
