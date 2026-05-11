"""Imputation avancée : benchmark médiane, KNN, MICE, multiple MICE."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import mean_squared_error

from src.core.context import Timer
from src.core.metaclasses import BasePipelineStep
from src.optim.benchmarks import generate_taxi_df

try:
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge

    ITERATIVE_AVAILABLE = True
except ImportError:
    ITERATIVE_AVAILABLE = False
    logging.warning("IterativeImputer indisponible")

try:
    import missingno as msno  # noqa: F401

    MISSINGNO_AVAILABLE = True
except ImportError:
    MISSINGNO_AVAILABLE = False


def mask_values(
    df: pd.DataFrame,
    columns: List[str],
    missing_pct: float = 0.05,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict]:
    """Masque artificiellement des valeurs pour évaluer les imputations."""
    logger = logging.getLogger(__name__)
    df_masked = df.copy()
    masks: Dict[str, Dict[str, np.ndarray | int]] = {}
    rng = np.random.default_rng(seed)

    for col in columns:
        n_mask = int(len(df) * missing_pct)
        mask_idx = rng.choice(len(df), size=n_mask, replace=False)
        true_values = df[col].values[mask_idx].copy()
        df_masked.loc[df_masked.index[mask_idx], col] = np.nan
        masks[col] = {
            "indices": mask_idx,
            "true_values": true_values,
            "n_masked": n_mask,
        }
        logger.info("Masqué %s valeurs (%.0f%%) dans '%s'", n_mask, missing_pct * 100, col)

    return df_masked, masks


def evaluate_imputation(
    df_imputed: pd.DataFrame,
    masks: Dict,
    df_original: pd.DataFrame,  # gardé pour signature demandée
) -> Dict:
    """Calcule RMSE/MAE uniquement sur les indices artificiellement masqués."""
    logger = logging.getLogger(__name__)
    _ = df_original
    results: Dict[str, Dict[str, float | int]] = {}

    for col, mask_info in masks.items():
        idx = mask_info["indices"]
        true_values = mask_info["true_values"]
        imputed_vals = df_imputed[col].values[idx]

        rmse = float(np.sqrt(mean_squared_error(true_values, imputed_vals)))
        mae = float(np.mean(np.abs(true_values - imputed_vals)))
        results[col] = {
            "rmse": rmse,
            "mae": mae,
            "n": int(len(true_values)),
        }
        logger.info("RMSE '%s' : %.4f | MAE : %.4f", col, rmse, mae)

    return results


class ImputationBenchmark(BasePipelineStep):
    """Compare 4 stratégies d'imputation sur valeurs masquées."""

    name = "imputation_benchmark"
    version = "1.0.0"

    def __init__(
        self,
        columns: List[str],
        missing_pct: float = 0.05,
        seed: int = 42,
        n_neighbors: int = 5,
        n_mice_iter: int = 10,
    ) -> None:
        self.columns = columns
        self.missing_pct = missing_pct
        self.seed = seed
        self.n_neighbors = n_neighbors
        self.n_mice_iter = n_mice_iter
        self._results: Dict[str, Dict] = {}
        self._logger = logging.getLogger(self.__class__.__name__)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lance le benchmark complet et retourne le DataFrame imputé."""
        df_masked, masks = mask_values(df, self.columns, self.missing_pct, self.seed)
        numeric_cols = df_masked.select_dtypes(include=[np.number]).columns
        self._results = {}

        with Timer("Médiane", unit="ms") as t:
            df_median = self._impute_median(df_masked[numeric_cols].copy())
        self._results["Médiane"] = {
            **evaluate_imputation(df_median, masks, df),
            "duration_ms": t.elapsed,
        }

        with Timer("KNN", unit="ms") as t:
            df_knn = self._impute_knn(df_masked[numeric_cols].copy())
        self._results["KNN"] = {
            **evaluate_imputation(df_knn, masks, df),
            "duration_ms": t.elapsed,
        }

        if ITERATIVE_AVAILABLE:
            with Timer("MICE", unit="ms") as t:
                df_mice = self._impute_mice(df_masked[numeric_cols].copy())
            self._results["MICE"] = {
                **evaluate_imputation(df_mice, masks, df),
                "duration_ms": t.elapsed,
            }

        if ITERATIVE_AVAILABLE:
            with Timer("Multiple MICE", unit="ms") as t:
                df_multi = self._impute_multiple(df_masked[numeric_cols].copy(), masks, df)
            self._results["Multiple MICE"] = {
                **evaluate_imputation(df_multi, masks, df),
                "duration_ms": t.elapsed,
            }

        self._print_report()
        return df_knn

    def _impute_median(self, df: pd.DataFrame) -> pd.DataFrame:
        """Imputation par médiane (baseline robuste/rapide)."""
        imputer = SimpleImputer(strategy="median")
        df_array = imputer.fit_transform(df)
        return pd.DataFrame(df_array, columns=df.columns, index=df.index)

    def _impute_knn(self, df: pd.DataFrame) -> pd.DataFrame:
        """Imputation KNN via voisins les plus proches."""
        imputer = KNNImputer(n_neighbors=self.n_neighbors)
        df_array = imputer.fit_transform(df)
        return pd.DataFrame(df_array, columns=df.columns, index=df.index)

    def _impute_mice(self, df: pd.DataFrame) -> pd.DataFrame:
        """MICE avec BayesianRidge ; fallback médiane si indisponible."""
        if not ITERATIVE_AVAILABLE:
            return self._impute_median(df)
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=self.n_mice_iter,
            random_state=self.seed,
        )
        df_array = imputer.fit_transform(df)
        return pd.DataFrame(df_array, columns=df.columns, index=df.index)

    def _impute_multiple(
        self,
        df: pd.DataFrame,
        masks: Dict,
        df_original: pd.DataFrame,
        n_imputations: int = 5,
    ) -> pd.DataFrame:
        """Multiple imputation (pooling simplifié : moyenne des imputations)."""
        _ = masks
        _ = df_original
        if not ITERATIVE_AVAILABLE:
            return self._impute_median(df)

        all_imputations: List[pd.DataFrame] = []
        for i in range(n_imputations):
            imputer = IterativeImputer(
                estimator=BayesianRidge(),
                max_iter=self.n_mice_iter,
                random_state=self.seed + i * 100,
            )
            df_imp = pd.DataFrame(
                imputer.fit_transform(df),
                columns=df.columns,
                index=df.index,
            )
            all_imputations.append(df_imp)

        df_pooled = pd.concat(all_imputations).groupby(level=0).mean()
        return df_pooled

    def _print_report(self) -> None:
        """Affiche un rapport comparatif lisible/pédagogique."""
        col_a = self.columns[0] if self.columns else "col1"
        col_b = self.columns[1] if len(self.columns) > 1 else col_a

        print("╔═══════════════════════════════════════════════════════════════╗")
        print("║          BENCHMARK IMPUTATION — RMSE par stratégie            ║")
        print("╠══════════════════╦══════════════════════╦════════════════════╣")
        print(f"║ Stratégie        ║ RMSE {col_a:<14} ║ RMSE {col_b:<12} ║")
        print("╠══════════════════╬══════════════════════╬════════════════════╣")
        for strategy, results in self._results.items():
            rmse_a = results.get(col_a, {}).get("rmse", np.nan)
            rmse_b = results.get(col_b, {}).get("rmse", np.nan)
            print(f"║ {strategy:<16} ║ {rmse_a:>18.4f} ║ {rmse_b:>18.4f} ║")
        print("╚══════════════════╩══════════════════════╩════════════════════╝")

        print("\nDurées (ms) :")
        for strategy, results in self._results.items():
            print(f"  - {strategy:<14}: {results.get('duration_ms', 0):.1f} ms")

        rmse_col = self.columns[-1] if self.columns else None
        if rmse_col is not None:
            best_rmse_strategy = min(
                self._results,
                key=lambda s: self._results[s].get(rmse_col, {}).get("rmse", np.inf),
            )
            print(f"\n→ Gagnant RMSE    : {best_rmse_strategy}")
        print("→ Gagnant vitesse : Médiane (toujours)")
        print("→ Recommandation  : KNN si temps disponible, Médiane en production")

    def summary(self) -> pd.DataFrame:
        """Retourne le résumé du benchmark sous forme tabulaire."""
        rows: List[Dict[str, float | str]] = []
        for strategy, results in self._results.items():
            row: Dict[str, float | str] = {
                "strategie": strategy,
                "duration_ms": float(results.get("duration_ms", 0)),
            }
            for col in self.columns:
                if col in results:
                    row[f"rmse_{col}"] = float(results[col]["rmse"])
                    row[f"mae_{col}"] = float(results[col]["mae"])
            rows.append(row)
        return pd.DataFrame(rows)


def analyze_missingness(df: pd.DataFrame, columns: List[str]) -> Dict:
    """Analyse % manquant, types et corrélation des patterns de manque."""
    results: Dict[str, Dict] = {}

    for col in columns:
        n_missing = df[col].isnull().sum()
        pct_missing = n_missing / len(df) * 100
        results[col] = {
            "n_missing": int(n_missing),
            "pct_missing": float(pct_missing),
            "dtype": str(df[col].dtype),
        }

    missing_matrix = df[columns].isnull().astype(int)
    if len(columns) >= 2:
        corr = missing_matrix.corr()
        results["missingness_correlation"] = corr.to_dict()

    print("╔══════════════════════════════════════════════════════╗")
    print("║         ANALYSE DES VALEURS MANQUANTES               ║")
    print("╠═══════════════════╦═══════════╦══════════════════════╣")
    print("║ Colonne           ║ Manquants ║ % manquant           ║")
    print("╠═══════════════════╬═══════════╬══════════════════════╣")
    for col in columns:
        print(
            f"║ {col:<17} ║ {results[col]['n_missing']:>9} ║ {results[col]['pct_missing']:>8.1f}%            ║"
        )
    print("╚═══════════════════╩═══════════╩══════════════════════╝")

    print(
        """
CLASSIFICATION DES DONNÉES MANQUANTES :
────────────────────────────────────────────
MCAR (Missing Completely At Random) :
  La probabilité de manquer est indépendante des données.
  Test : Little's MCAR test. Imputation simple possible.

MAR (Missing At Random) :
  La probabilité de manquer dépend d'autres variables observées.
  Exemple NYC Taxi : passenger_count manque plus souvent sur certains types de trajets.
  → KNN ou MICE recommandés.

MNAR (Missing Not At Random) :
  La probabilité de manquer dépend de la valeur elle-même.
  Exemple : tip_amount absent car zéro implicite non saisi.
  → Modélisation explicite du mécanisme de manque requise.

Pour NYC Taxi : passenger_count et trip_distance sont probablement
MAR (liés à l'heure, la zone, le type de paiement).
"""
    )

    return results


if __name__ == "__main__":
    print("=== EXERCICE 3.7 — IMPUTATION AVANCÉE ===\n")

    df_raw = generate_taxi_df(n_rows=50_000)

    print("--- A. Analyse des valeurs manquantes ---")
    df_masked, masks = mask_values(
        df_raw,
        columns=["passenger_count", "trip_distance"],
        missing_pct=0.05,
    )
    _ = masks
    _ = analyze_missingness(df_masked, ["passenger_count", "trip_distance"])

    print("\n--- B. Benchmark des stratégies d'imputation ---")
    benchmark = ImputationBenchmark(
        columns=["passenger_count", "trip_distance"],
        missing_pct=0.05,
        n_neighbors=5,
        n_mice_iter=5,
    )
    df_imputed = benchmark.run(df_raw)
    print("\nRésumé du benchmark :")
    print(benchmark.summary().to_string(index=False))

    print("\n--- C. Conclusion ---")
    summary = benchmark.summary()
    if "rmse_trip_distance" in summary.columns:
        best = summary.loc[summary["rmse_trip_distance"].idxmin(), "strategie"]
        fastest = summary.loc[summary["duration_ms"].idxmin(), "strategie"]
        print(f"\n✅ Meilleure précision (RMSE) : {best}")
        print(f"⚡ Plus rapide               : {fastest}")

    _ = df_imputed
