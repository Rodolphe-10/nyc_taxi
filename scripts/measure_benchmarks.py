"""Mesures chiffrées pour le rapport et l'application Streamlit.

Commande, depuis la racine du dépôt :

    python scripts/measure_benchmarks.py

Écrit ``reports/benchmarks.json``. Les cibles de l'énoncé ne sont pas
recopiées : seules les durées observées sur cette machine sont enregistrées.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.optim.benchmarks import generate_taxi_df
from src.optim.jit_kernels import benchmark_haversine
from src.optim.memory import optimize_dtypes
from src.optim.parallel import benchmark_gil_cpu_bound, benchmark_gil_io_bound
from src.optim.vectorize import benchmark_vectorization

logger = logging.getLogger(__name__)

OUT = ROOT / "reports" / "benchmarks.json"
JANUARY = ROOT / "data" / "raw" / "yellow_tripdata_2023-01.parquet"


def _dtype_gain(df: pd.DataFrame) -> dict:
    """Gain mémoire de ``optimize_dtypes`` sur une copie."""
    before = int(df.memory_usage(deep=True).sum())
    optimized = optimize_dtypes(df.copy(), verbose=False)
    after = int(optimized.memory_usage(deep=True).sum())
    gain = (before - after) / before if before else 0.0
    return {
        "n_rows": int(len(df)),
        "bytes_before": before,
        "bytes_after": after,
        "gain_ratio": gain,
    }


def _vectorization(n_rows: int) -> dict:
    """Speedup iterrows vers NumPy, tel que renvoyé par le module."""
    table = benchmark_vectorization(n_rows=n_rows)
    records = table.to_dict(orient="records")
    naive = next(r for r in records if r["label"] == "naive")
    vect = next(r for r in records if r["label"] == "vectorized")
    return {
        "n_rows": n_rows,
        "n_runs_compare": 3,
        "naive_mean_ms": float(naive["mean_ms"]),
        "vectorized_mean_ms": float(vect["mean_ms"]),
        "speedup": float(naive["mean_ms"] / vect["mean_ms"]),
        "note": (
            "compare_functions relance chaque implémentation 3 fois. "
            "Le jeu est synthétique (generate_taxi_df), pas le mois de janvier."
        ),
    }


def main() -> None:
    """Enchaîne les benchmarks et écrit le JSON."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    payload: dict = {
        "measured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": "python scripts/measure_benchmarks.py",
        "python": "3.10.11",
    }

    logger.info("Vectorisation sur 20 000 lignes synthétiques")
    payload["vectorization"] = _vectorization(20_000)

    logger.info("Dtypes sur 100 000 lignes synthétiques")
    payload["dtypes_synthetic"] = _dtype_gain(generate_taxi_df(100_000))

    if JANUARY.is_file():
        logger.info("Dtypes sur les 200 000 premières lignes de janvier 2023")
        january = pd.read_parquet(JANUARY).head(200_000)
        payload["dtypes_january_head"] = _dtype_gain(january)
        payload["dtypes_january_head"]["source"] = str(JANUARY.name)
        payload["dtypes_january_head"]["note"] = (
            "head(200000) après lecture du parquet de janvier, pas les 12 mois."
        )
    else:
        payload["dtypes_january_head"] = None

    logger.info("Haversine, 1 000 000 de paires (Python extrapolé depuis 10 000)")
    hav = benchmark_haversine(n_points=1_000_000)
    payload["haversine"] = {
        key: (None if isinstance(val, float) and val != val else float(val) if isinstance(val, float) else int(val))
        for key, val in hav.items()
    }
    payload["haversine"]["python_sample_points"] = 10_000
    payload["haversine"]["note"] = (
        "Le temps Python est extrapolé : 10 000 appels mesurés, "
        "multipliés par n_points / 10 000. Cible de l'énoncé : 10^7 paires."
    )

    logger.info("GIL CPU")
    cpu = benchmark_gil_cpu_bound(n_workers=4, size=200_000)
    payload["gil_cpu"] = {k: float(v) if isinstance(v, float) else int(v) for k, v in cpu.items()}

    logger.info("GIL I/O (seul janvier 2023 est présent sur le disque)")
    io = benchmark_gil_io_bound(n_files=4)
    payload["gil_io"] = {k: float(v) if isinstance(v, float) else int(v) for k, v in io.items()}
    payload["gil_io"]["note"] = (
        "n_files=4. Seul yellow_tripdata_2023-01.parquet existe : "
        "les autres chemins tombent dans le sleep de 0,1 s de io_bound_task."
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Écrit %s", OUT)


if __name__ == "__main__":
    main()
