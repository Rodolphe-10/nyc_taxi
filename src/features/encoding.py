"""Target encoding bayésien, cross-fitting et embeddings de zones."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.context import Timer
from src.core.metaclasses import BasePipelineStep
from src.optim.benchmarks import generate_taxi_df

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch absent — embeddings indisponibles")


class BayesianTargetEncoder(BasePipelineStep):
    """Encodage target avec smoothing bayésien."""

    name = "bayesian_target_encoder"
    version = "1.0.0"

    def __init__(
        self,
        columns: List[str],
        target: str = "tip_pct",
        smoothing: float = 10.0,
        min_samples: int = 5,
    ) -> None:
        self.columns = columns
        self.target = target
        self.smoothing = smoothing
        self.min_samples = min_samples
        self._encoding_maps: Dict[str, Dict] = {}
        self._global_means: Dict[str, float] = {}
        self._fitted = False
        self._logger = logging.getLogger(self.__class__.__name__)

    def fit(self, df: pd.DataFrame) -> "BayesianTargetEncoder":
        """Apprend les encodages depuis les données d'entraînement."""
        for col in self.columns:
            global_mean = float(df[self.target].mean())
            self._global_means[col] = global_mean

            stats = df.groupby(col)[self.target].agg(["count", "mean"])
            n_c = stats["count"]
            mean_c = stats["mean"]
            m = self.smoothing
            encoded = (n_c * mean_c + m * global_mean) / (n_c + m)

            encoded = encoded[n_c >= self.min_samples]
            self._encoding_maps[col] = encoded.to_dict()
            self._logger.info(
                "BayesianTargetEncoder fit : '%s' | %s catégories | smoothing=%s",
                col,
                len(encoded),
                self.smoothing,
            )

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applique l'encodage sur de nouvelles données."""
        if not self._fitted:
            raise RuntimeError("Appelez fit() avant transform()")

        out = df.copy()
        for col in self.columns:
            new_col = f"{col}_target_enc"
            out[new_col] = (
                out[col]
                .map(self._encoding_maps[col])
                .fillna(self._global_means[col])
                .astype("float32")
            )
            out = out.drop(columns=[col])

        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Implémentation BasePipelineStep."""
        return self.fit_transform(df)

    def smoothing_effect(self, col: str) -> pd.DataFrame:
        """Analyse l'effet du smoothing sur la déviation au global mean."""
        if not self._fitted:
            raise RuntimeError("Appelez fit() avant smoothing_effect()")
        if col not in self._encoding_maps:
            raise KeyError(f"Colonne inconnue: {col}")

        encoding_map = self._encoding_maps[col]
        global_mean = self._global_means[col]
        result = pd.DataFrame(
            {
                "category": list(encoding_map.keys()),
                "encoded_value": list(encoding_map.values()),
                "global_mean": global_mean,
                "deviation": [abs(v - global_mean) for v in encoding_map.values()],
            }
        )
        return result.sort_values("deviation", ascending=False).reset_index(drop=True)


class CrossFittedTargetEncoder(BasePipelineStep):
    """Target encoding K-Fold pour limiter la fuite de données."""

    name = "cross_fitted_target_encoder"
    version = "1.0.0"

    def __init__(
        self,
        columns: List[str],
        target: str = "tip_pct",
        smoothing: float = 10.0,
        n_splits: int = 5,
        random_state: int = 42,
    ) -> None:
        self.columns = columns
        self.target = target
        self.smoothing = smoothing
        self.n_splits = n_splits
        self.random_state = random_state
        self._logger = logging.getLogger(self.__class__.__name__)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross-fitted target encoding en K folds (sans leakage intra-ligne)."""
        from sklearn.model_selection import KFold

        out = df.copy()
        kf = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        indices = np.arange(len(out))

        for col in self.columns:
            new_col = f"{col}_cross_enc"
            out[new_col] = np.nan
            global_mean = float(out[self.target].mean())

            for train_idx, val_idx in kf.split(indices):
                train_df = out.iloc[train_idx]
                stats = train_df.groupby(col)[self.target].agg(["count", "mean"])
                n_c = stats["count"]
                mean_c = stats["mean"]
                m = self.smoothing
                encoded = (n_c * mean_c + m * global_mean) / (n_c + m)
                encoding_map = encoded.to_dict()

                out.loc[out.index[val_idx], new_col] = (
                    out.iloc[val_idx][col].map(encoding_map).fillna(global_mean).values
                )

            out[new_col] = out[new_col].fillna(global_mean).astype("float32")
            self._logger.info(
                "CrossFitted '%s' : %s folds | smoothing=%s",
                col,
                self.n_splits,
                self.smoothing,
            )

        return out

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit_transform(df)


def compare_smoothing_effect(
    df: pd.DataFrame,
    col: str,
    target: str,
    smoothing_values: List[float],
) -> pd.DataFrame:
    """Compare la variance des encodages selon m (smoothing)."""
    results: List[Dict[str, float]] = []

    for m in smoothing_values:
        encoder = BayesianTargetEncoder(columns=[col], target=target, smoothing=m)
        encoder.fit(df)
        encoded_values = list(encoder._encoding_maps[col].values())
        if not encoded_values:
            stats_tuple: Tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
        else:
            arr = np.asarray(encoded_values, dtype=np.float64)
            stats_tuple = (
                float(np.var(arr)),
                float(np.std(arr)),
                float(np.min(arr)),
                float(np.max(arr)),
                float(np.max(arr) - np.min(arr)),
            )

        variance, std, min_v, max_v, range_v = stats_tuple
        results.append(
            {
                "smoothing": float(m),
                "variance": variance,
                "std": std,
                "min": min_v,
                "max": max_v,
                "range": range_v,
            }
        )

    df_result = pd.DataFrame(results)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║   EFFET DU SMOOTHING SUR LA VARIANCE DES ENCODAGES       ║")
    print("╠═══════════╦═══════════╦═══════════╦═════════════════════╣")
    print("║ Smoothing ║ Variance  ║   Std     ║   Range             ║")
    print("╠═══════════╬═══════════╬═══════════╬═════════════════════╣")
    for _, row in df_result.iterrows():
        print(
            f"║ {row['smoothing']:>8.0f}  ║ {row['variance']:>8.5f}  ║ {row['std']:>8.5f}  ║ {row['range']:>10.5f}          ║"
        )
    print("╚═══════════╩═══════════╩═══════════╩═════════════════════╝")
    print(
        "→ Plus m est grand, plus les encodages tirent vers global_mean\n"
        "→ Les catégories peu fréquentes sont davantage régularisées\n"
        "→ m=10 est un bon compromis biais/variance"
    )
    return df_result


if TORCH_AVAILABLE:

    class ZoneEmbeddingModel(nn.Module):
        """Petit réseau d'embeddings denses pour les zones."""

        def __init__(
            self,
            n_zones: int = 265,
            embedding_dim: int = 8,
            hidden_dim: int = 32,
            output_dim: int = 1,
        ) -> None:
            super().__init__()
            self.embedding = nn.Embedding(n_zones + 1, embedding_dim, padding_idx=0)
            self.network = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, zone_ids: torch.Tensor) -> torch.Tensor:
            embedded = self.embedding(zone_ids)
            return self.network(embedded).squeeze(-1)

        def get_embeddings(self) -> np.ndarray:
            """Retourne les embeddings appris sous forme NumPy."""
            return self.embedding.weight.detach().cpu().numpy()


def train_zone_embeddings(
    df: pd.DataFrame,
    zone_col: str = "PULocationID",
    target: str = "tip_pct",
    epochs: int = 10,
    batch_size: int = 1024,
    lr: float = 0.001,
) -> Optional["ZoneEmbeddingModel"]:
    """Entraîne un modèle d'embeddings de zones."""
    if not TORCH_AVAILABLE:
        logging.warning("PyTorch absent — embeddings ignorés")
        return None

    zone_ids = torch.tensor(df[zone_col].values.astype(np.int64))
    targets = torch.tensor(df[target].values.astype(np.float32))

    model = ZoneEmbeddingModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(len(zone_ids))
        losses: List[float] = []
        for i in range(0, len(zone_ids), batch_size):
            idx = perm[i : i + batch_size]
            preds = model(zone_ids[idx])
            loss = criterion(preds, targets[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        if (epoch + 1) % 5 == 0:
            logging.info("Epoch %s/%s | Loss: %.4f", epoch + 1, epochs, float(np.mean(losses)))

    return model


if __name__ == "__main__":
    print("=== EXERCICE 3.6 — TARGET ENCODING & EMBEDDINGS ===\n")

    df_raw = generate_taxi_df(n_rows=100_000)
    df_prep = df_raw.copy()
    df_prep["tip_pct"] = (df_prep["tip_amount"] / df_prep["fare_amount"].clip(lower=0.01)).clip(0, 2)

    print("--- A. Bayesian Target Encoding ---")
    encoder = BayesianTargetEncoder(
        columns=["PULocationID", "DOLocationID"],
        target="tip_pct",
        smoothing=10.0,
    )
    with Timer("BayesianTargetEncoder", unit="ms") as t:
        df_encoded = encoder.fit_transform(df_prep.copy())
    print(f"Encodé en {t.elapsed:.1f}ms")
    print(f"Shape : {df_prep.shape} → {df_encoded.shape}")
    print("\nAperçu PULocationID_target_enc :")
    print(df_encoded["PULocationID_target_enc"].describe().to_string())

    print("\n--- B. Effet du Smoothing ---")
    smoothing_df = compare_smoothing_effect(
        df_prep,
        col="PULocationID",
        target="tip_pct",
        smoothing_values=[0, 1, 5, 10, 50, 100, 1000],
    )
    print(smoothing_df.to_string(index=False))

    print("\n--- C. Cross-Fitted Target Encoding (5 folds) ---")
    cross_encoder = CrossFittedTargetEncoder(
        columns=["PULocationID"],
        target="tip_pct",
        smoothing=10.0,
        n_splits=5,
    )
    with Timer("CrossFittedTargetEncoder", unit="ms") as t:
        df_cross = cross_encoder.fit_transform(df_prep.copy())
    print(f"Cross-fitted en {t.elapsed:.1f}ms")
    print(df_cross["PULocationID_cross_enc"].describe().to_string())

    corr = df_cross["PULocationID_cross_enc"].corr(df_encoded["PULocationID_target_enc"])
    print(f"\nCorrélation Bayesian vs Cross-fitted : {corr:.4f}")

    if TORCH_AVAILABLE:
        print("\n--- D. Zone Embeddings (PyTorch) ---")
        model = train_zone_embeddings(
            df_prep,
            zone_col="PULocationID",
            target="tip_pct",
            epochs=10,
            lr=0.001,
        )
        if model is not None:
            embeddings = model.get_embeddings()
            print(f"Embeddings shape : {embeddings.shape}")
