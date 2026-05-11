"""Décorateurs utilitaires pour instrumentation non intrusive."""

import cProfile
import functools
import hashlib
import io
import logging
import pstats
import time
from collections import OrderedDict
from typing import Callable, Tuple, Type


# Décorateur de timing paramétrable.
# Il supporte @timeit, @timeit(), @timeit("s") et @timeit(unit="min").
def timeit(func_or_unit=None, unit="ms"):
    """Mesure le temps d'exécution d'une fonction et le journalise."""

    units = {"ms": 1000.0, "s": 1.0, "min": 1.0 / 60.0}
    logger = logging.getLogger("decorators.timeit")

    def _decorate(func, selected_unit):
        factor = units.get(selected_unit)
        if factor is None:
            raise ValueError(f"Unité '{selected_unit}' non supportée. Utilisez: ms, s, min")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_seconds = time.perf_counter() - start
            duration = elapsed_seconds * factor
            wrapper._last_duration = duration
            logger.info("[timeit] %s → %.3f %s", func.__name__, duration, selected_unit)
            return result

        wrapper._last_duration = None
        return wrapper

    if callable(func_or_unit):
        return _decorate(func_or_unit, unit)

    selected_unit = func_or_unit if isinstance(func_or_unit, str) else unit

    def decorator(func):
        return _decorate(func, selected_unit)

    return decorator


# Décorateur de logging d'appels.
# Il supporte @log_calls et @log_calls(...).
def log_calls(func=None, level="INFO", log_args=True, log_result=False):
    """Journalise les appels, résultats et exceptions d'une fonction."""

    def _truncate(value):
        text = repr(value)
        return text if len(text) <= 100 else text[:100] + "..."

    def _decorate(inner_func, selected_level, selected_log_args, selected_log_result):
        logger = logging.getLogger("decorators.log_calls")
        numeric_level = getattr(logging, str(selected_level).upper(), logging.INFO)

        @functools.wraps(inner_func)
        def wrapper(*args, **kwargs):
            try:
                if selected_log_args:
                    call_info = f"args={_truncate(args)}, kwargs={_truncate(kwargs)}"
                else:
                    call_info = "args/kwargs masqués"
                logger.log(numeric_level, "[log_calls] APPEL %s(%s)", inner_func.__name__, call_info)

                result = inner_func(*args, **kwargs)
                if selected_log_result:
                    logger.log(numeric_level, "[log_calls] RESULTAT %s -> %s", inner_func.__name__, _truncate(result))
                return result
            except Exception as exc:
                logger.error("[log_calls] EXCEPTION dans %s: %s", inner_func.__name__, exc)
                raise

        return wrapper

    if callable(func):
        return _decorate(func, level, log_args, log_result)

    def decorator(inner_func):
        return _decorate(inner_func, level, log_args, log_result)

    return decorator


# Décorateur de retry paramétré avec backoff exponentiel.
def retry(max_attempts=3, backoff=2.0, exceptions=(Exception,)):
    """Réessaie une fonction en cas d'exception capturée."""

    def decorator(func):
        logger = logging.getLogger("decorators.retry")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_attempts:
                        raise
                    delay = backoff ** (attempt - 1)
                    logger.warning(
                        "[retry] %s — tentative %s/%s échouée. Retry dans %.1fs. Erreur : %s",
                        func.__name__,
                        attempt,
                        max_attempts,
                        delay,
                        exc,
                    )
                    time.sleep(delay)

        return wrapper

    return decorator


# Décorateur de cache LRU pour fonctions recevant des DataFrames.
def memoize_dataframe(maxsize=4):
    """Mémorise le résultat selon un hash stable des DataFrames d'entrée."""

    def decorator(func):
        cache = OrderedDict()
        stats = {"hits": 0, "misses": 0}

        def _hash_df(df) -> str:
            import pandas as pd

            digest_source = str(pd.util.hash_pandas_object(df, index=True).sum()).encode("utf-8")
            return hashlib.md5(digest_source).hexdigest()

        def _build_key(args, kwargs):
            import pandas as pd

            parts = []
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    parts.append(("df", _hash_df(arg)))
                else:
                    parts.append(("arg", repr(arg)))
            for key, value in sorted(kwargs.items(), key=lambda item: item[0]):
                if isinstance(value, pd.DataFrame):
                    parts.append((f"kw_df:{key}", _hash_df(value)))
                else:
                    parts.append((f"kw:{key}", repr(value)))
            return repr(parts)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = _build_key(args, kwargs)
            stats["hits"] += 1
            if key in cache:
                cache.move_to_end(key)
                return cache[key]

            stats["misses"] += 1
            result = func(*args, **kwargs)
            cache[key] = result
            cache.move_to_end(key)
            if len(cache) > maxsize:
                cache.popitem(last=False)
            return result

        def cache_info():
            return {
                "hits": stats["hits"],
                "misses": stats["misses"],
                "size": len(cache),
                "maxsize": maxsize,
            }

        wrapper.cache_info = cache_info
        return wrapper

    return decorator


# Décorateur de profiling simple avec affichage top 10 cumulé.
def profile_func(func):
    """Profile une fonction avec cProfile et imprime les coûts cumulés."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()

        output = io.StringIO()
        stats = pstats.Stats(profiler, stream=output).sort_stats("cumulative")
        stats.print_stats(10)
        print(output.getvalue())
        return result

    return wrapper


if __name__ == "__main__":
    # Section A — @timeit.
    @timeit
    def operation_lente():
        time.sleep(0.1)
        return "ok"

    @timeit("s")
    def operation_rapide():
        return sum(range(1_000_000))

    operation_lente()
    operation_rapide()
    print(f"Dernière durée : {operation_rapide._last_duration:.4f}s")

    # Section B — @log_calls.
    @log_calls(level="INFO", log_args=True, log_result=True)
    def calculer(x, y, operation="add"):
        if operation == "div" and y == 0:
            raise ValueError("Division par zéro")
        return x + y if operation == "add" else x / y

    calculer(10, 20)
    try:
        calculer(10, 0, operation="div")
    except ValueError:
        pass

    # Section C — @retry.
    compteur = {"n": 0}

    @retry(max_attempts=3, backoff=0.1, exceptions=(IOError,))
    def operation_instable():
        compteur["n"] += 1
        if compteur["n"] < 3:
            raise IOError("Connexion échouée")
        return "Succès à la 3ème tentative"

    resultat = operation_instable()
    print(f"Résultat : {resultat}")

    # Section D — Chaînage de décorateurs.
    @retry(max_attempts=2, backoff=0.1)
    @timeit(unit="ms")
    @log_calls(log_args=True)
    def load_month(year, month):
        time.sleep(0.05)
        return f"data_{year}_{month:02d}"

    result = load_month(2023, 1)
    print(f"Chaînage OK : {result}")
    print("Ordre d'application : retry → timeit → log_calls → fonction")

    # Section E — @memoize_dataframe.
    import numpy as np
    import pandas as pd

    @memoize_dataframe(maxsize=3)
    def process_df(df):
        time.sleep(0.1)
        return df.describe()

    df = pd.DataFrame(np.random.randn(1000, 5))
    process_df(df)  # miss
    process_df(df)  # hit
    process_df(df)  # hit
    print(f"Cache info : {process_df.cache_info()}")

    # Section F — Explication de l'ordre des décorateurs.
    print(
        """
ORDRE DES DÉCORATEURS :
@retry
@timeit
@log_calls
def load_month(): ...

Équivalent à : retry(timeit(log_calls(load_month)))
Ordre d'exécution (de l'extérieur vers l'intérieur) :
1. retry intercepte → 2. timeit mesure → 3. log_calls logue → 4. fonction
C'est important car timeit mesure APRÈS log_calls,
donc inclut le temps de logging dans la mesure.
"""
    )
