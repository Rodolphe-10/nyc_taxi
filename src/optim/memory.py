"""Optimisation mémoire (dtypes) et lecture Parquet par chunks (streaming)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.optim.benchmarks import generate_taxi_df

logger = logging.getLogger(__name__)

# Référence dtypes cibles (colonnes absentes du DataFrame sont ignorées).
DTYPE_OPTIMIZATION_MAP = {
    "passenger_count": "int8",
    "RatecodeID": "int8",
    "store_and_fwd_flag": "category",
    "PULocationID": "int16",
    "DOLocationID": "int16",
    "payment_type": "int8",
    "fare_amount": "float32",
    "extra": "float32",
    "mta_tax": "float32",
    "tip_amount": "float32",
    "tolls_amount": "float32",
    "improvement_surcharge": "float32",
    "total_amount": "float32",
    "trip_distance": "float32",
    "congestion_surcharge": "float32",
}


def optimize_dtypes(
    df: pd.DataFrame,
    custom_map: Optional[Dict[str, str]] = None,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Optimise les dtypes : carte NYC → int/float/cat, puis int64/float64 résiduels."""
    result = df.copy()
    ram_avant = result.memory_usage(deep=True).sum() / 1e6

    merged: Dict[str, str] = {**DTYPE_OPTIMIZATION_MAP, **(custom_map or {})}

    # PASSE 1 — Applique la carte (colonnes présentes uniquement).
    for col, target_dtype in merged.items():
        if col not in result.columns:
            continue
        try:
            if target_dtype == "category":
                result[col] = result[col].astype("category")
            elif target_dtype in ("int8", "int16", "int32", "int64"):
                result[col] = result[col].astype(np.dtype(target_dtype))
            elif target_dtype == "float32":
                result[col] = result[col].astype(np.float32)
            else:
                result[col] = result[col].astype(target_dtype)
        except (ValueError, TypeError) as e:
            logger.warning("optimize_dtypes passe 1 — colonne %s : %s", col, e)

    # PASSE 2 — int64 restants : plus petit entier signé possible sans perte.
    for col in result.columns:
        if result[col].dtype != np.int64:
            continue
        if not pd.api.types.is_integer_dtype(result[col]):
            continue
        try:
            col_min = int(result[col].min())
            col_max = int(result[col].max())
        except TypeError:
            continue
        if col_min >= -128 and col_max <= 127:
            result[col] = result[col].astype(np.int8)
        elif col_min >= -32768 and col_max <= 32767:
            result[col] = result[col].astype(np.int16)
        else:
            result[col] = result[col].astype(np.int32)

    # PASSE 3 — float64 restants : float32 si précision suffisante (pas de NaN artificiels).
    for col in result.columns:
        if result[col].dtype != np.float64:
            continue
        if not pd.api.types.is_float_dtype(result[col]):
            continue
        s = result[col]
        dropped = s.dropna()
        if len(dropped) == 0:
            result[col] = s.astype(np.float32)
            continue
        as32 = s.astype(np.float32)
        if np.allclose(s.values, as32.values, rtol=1e-3, equal_nan=True):
            result[col] = as32

    ram_apres = result.memory_usage(deep=True).sum() / 1e6
    gain_pct = ((ram_avant - ram_apres) / ram_avant * 100) if ram_avant > 0 else 0.0
    if verbose:
        print(f"RAM avant : {ram_avant:.1f} Mo")
        print(f"RAM après  : {ram_apres:.1f} Mo")
        print(f"Gain       : {gain_pct:.1f}%")

    return result


def memory_report(df: pd.DataFrame) -> pd.DataFrame:
    """Rapport mémoire par colonne (Mo, % du total), tri décroissant."""
    per_col = df.memory_usage(deep=True)
    total = per_col.sum()
    rows = []
    for col in df.columns:
        mo = per_col[col] / 1e6
        pct = (per_col[col] / total * 100) if total > 0 else 0.0
        nu = df[col].nunique(dropna=False)
        rows.append(
            {
                "colonne": col,
                "dtype": str(df[col].dtype),
                "mo": mo,
                "pct_total": pct,
                "n_unique": int(nu) if nu < 1000 else np.nan,
            }
        )
    report = pd.DataFrame(rows).sort_values("mo", ascending=False).reset_index(drop=True)
    print(report.to_string(index=False))
    return report


def iter_parquet_batches(
    path: str,
    batch_size: int = 100_000,
    columns: Optional[List[str]] = None,
) -> Iterator[pd.DataFrame]:
    """Lit un Parquet via PyArrow par lots, optimise les dtypes à chaque batch."""
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        df_batch = batch.to_pandas()
        df_batch = optimize_dtypes(df_batch, verbose=False)
        yield df_batch
    logger.info("Fichier %s lu par batches de %s lignes", path, batch_size)


def compute_stats_streaming(
    path: str,
    columns: Optional[List[str]] = None,
) -> Dict:
    """Stats (sommes, min, max, moyennes) sur un Parquet sans charger tout le fichier."""
    total_rows = 0
    col_sums: Dict[str, float] = {}
    col_mins: Dict[str, float] = {}
    col_maxs: Dict[str, float] = {}
    use_cols: Optional[List[str]] = None

    for batch in iter_parquet_batches(path, batch_size=100_000, columns=columns):
        if use_cols is None:
            cand = list(batch.columns) if columns is None else [c for c in columns if c in batch.columns]
            use_cols = [c for c in cand if pd.api.types.is_numeric_dtype(batch[c])]
        n = len(batch)
        total_rows += n
        for col in use_cols or []:
            s = batch[col]
            s_num = pd.to_numeric(s, errors="coerce")
            sm = float(s_num.sum(skipna=True))
            valid = s_num.dropna()
            if len(valid) == 0:
                continue
            mn = float(valid.min())
            mx = float(valid.max())
            if col not in col_sums:
                col_sums[col] = 0.0
                col_mins[col] = mn
                col_maxs[col] = mx
            else:
                col_mins[col] = min(col_mins[col], mn)
                col_maxs[col] = max(col_maxs[col], mx)
            col_sums[col] += sm

    means = {c: col_sums[c] / total_rows for c in col_sums} if total_rows > 0 else {}
    return {
        "total_rows": total_rows,
        "means": means,
        "mins": col_mins,
        "maxs": col_maxs,
    }


def load_all_months_optimized(
    raw_dir: str = "data/raw",
    months: Optional[List[int]] = None,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Charge les Parquet mensuels 2023 disponibles, dtypes optimisés, puis concat."""
    months = months or list(range(1, 13))
    frames: List[pd.DataFrame] = []
    n_ok = 0
    for m in months:
        fname = f"yellow_tripdata_2023-{m:02d}.parquet"
        fp = Path(raw_dir) / fname
        if not fp.exists():
            logger.warning("Fichier manquant : %s", fp)
            continue
        df_m = pd.read_parquet(fp, columns=columns)
        df_m = optimize_dtypes(df_m, verbose=False)
        frames.append(df_m)
        n_ok += 1

    if not frames:
        logger.warning("Aucun mois chargé depuis %s", raw_dir)
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    ram = df.memory_usage(deep=True).sum() / 1e6
    print(
        f"[OK] {n_ok} mois chargés | {len(df):,} lignes | {ram:.1f} Mo en RAM"
    )
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== EXERCICE 3.1 — OPTIMISATION MÉMOIRE ===\n")
    df = generate_taxi_df(n_rows=500_000)
    print(f"Dataset : {df.shape}")

    print("\n--- Rapport AVANT optimisation ---")
    memory_report(df)

    df_opt = optimize_dtypes(df.copy())

    print("\n--- Rapport APRÈS optimisation ---")
    memory_report(df_opt)

    print("\n--- CHUNKING STREAMING ---")
    parquet_files = list(Path("data/raw").glob("*.parquet"))
    if parquet_files:
        path = str(parquet_files[0])
        print(f"Fichier : {path}")
        stats = compute_stats_streaming(
            path,
            columns=["fare_amount", "trip_distance", "tip_amount"],
        )
        print(f"Lignes totales  : {stats['total_rows']:,}")
        fa = stats["means"].get("fare_amount", float("nan"))
        print(f"fare_amount moy : {fa:.2f}" if fa == fa else "fare_amount moy : N/A")
    else:
        print("[INFO] Aucun fichier Parquet disponible")
        print("[INFO] Simulation du chunking sur données générées...")
        df_big = generate_taxi_df(n_rows=1_000_000)
        ram_avant = df_big.memory_usage(deep=True).sum() / 1e6
        df_opt2 = optimize_dtypes(df_big)
        ram_apres = df_opt2.memory_usage(deep=True).sum() / 1e6
        print(f"RAM avant  : {ram_avant:.1f} Mo")
        print(f"RAM après  : {ram_apres:.1f} Mo")
        print(f"Gain       : {(ram_avant - ram_apres) / ram_avant * 100:.1f}%")
