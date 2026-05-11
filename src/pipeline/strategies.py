"""Stratégies d'encodage et ingénierie de features pour le pipeline."""

from abc import ABC, abstractmethod
import logging

import numpy as np
import pandas as pd

from src.core.metaclasses import BasePipelineStep


# Interface abstraite commune à toutes les stratégies d'encodage.
class EncodingStrategy(ABC):
    """Contrat minimal : fit() apprend, encode() transforme."""

    # Encode une colonne et retourne un nouveau DataFrame.
    @abstractmethod
    def encode(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        pass

    # Apprend les paramètres d'encodage depuis le jeu d'entraînement.
    @abstractmethod
    def fit(
        self,
        df: pd.DataFrame,
        column: str,
        target: str = None,
    ) -> "EncodingStrategy":
        pass

    # Indique si la stratégie a été entraînée.
    def is_fitted(self) -> bool:
        return hasattr(self, "_fitted") and self._fitted

    # Représentation compacte de l'état de la stratégie.
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} fitted={self.is_fitted()}>"


# Encodage One-Hot.
# Avantage : aucune hypothèse d'ordre.
# Inconvénient : explosion du nombre de colonnes en haute cardinalité.
class OneHotStrategy(EncodingStrategy):
    """Encodage one-hot limité aux catégories les plus fréquentes."""

    # Configure le comportement de one-hot.
    def __init__(self, drop_first=True, max_categories=50):
        self.drop_first = drop_first
        self.max_categories = max_categories
        self._categories = None
        self._fitted = False

    # Apprend les catégories dominantes à encoder.
    def fit(self, df, column, target=None):
        self._categories = (
            df[column].value_counts().head(self.max_categories).index.tolist()
        )
        self._fitted = True
        return self

    # Encode en one-hot en ne gardant que les catégories apprises.
    def encode(self, df, column):
        if not self.is_fitted():
            raise RuntimeError("Appelez fit() avant encode()")

        encoded_df = df.copy()
        filtered = encoded_df[column].where(encoded_df[column].isin(self._categories), other=np.nan)
        dummies = pd.get_dummies(filtered, prefix=column, drop_first=self.drop_first)
        encoded_df = encoded_df.drop(columns=[column]).join(dummies)
        return encoded_df


# Encodage par fréquence.
# Avantage : une seule colonne, compact.
# Inconvénient : collisions possibles entre catégories de même fréquence.
class FrequencyEncodingStrategy(EncodingStrategy):
    """Remplace la catégorie par sa fréquence relative."""

    # Initialise la structure de mapping des fréquences.
    def __init__(self):
        self._freq_map = None
        self._fitted = False

    # Calcule la fréquence relative de chaque catégorie.
    def fit(self, df, column, target=None):
        self._freq_map = df[column].value_counts(normalize=True).to_dict()
        self._fitted = True
        return self

    # Applique le mapping de fréquence appris.
    def encode(self, df, column):
        if not self.is_fitted():
            raise RuntimeError("Appelez fit() avant encode()")

        encoded_df = df.copy()
        encoded_df[column] = encoded_df[column].map(self._freq_map).fillna(0.0)
        encoded_df = encoded_df.rename(columns={column: f"{column}_freq"})
        return encoded_df


# Encodage target avec smoothing bayésien.
# Avantage : exploite le signal de la cible.
# Inconvénient : nécessite précautions contre la fuite de données.
class TargetEncodingStrategy(EncodingStrategy):
    """Encode une catégorie par la moyenne cible régularisée."""

    # Configure la force de régularisation du smoothing.
    def __init__(self, smoothing=10.0):
        self.smoothing = smoothing
        self._encoding_map = None
        self._global_mean = None
        self._fitted = False

    # Apprend la table d'encodage target par catégorie.
    def fit(self, df, column, target):
        if target is None:
            raise ValueError("Le paramètre 'target' est requis pour le target encoding")

        self._global_mean = df[target].mean()
        grouped = df.groupby(column)[target].agg(["count", "mean"])
        encoded = (
            grouped["count"] * grouped["mean"] + self.smoothing * self._global_mean
        ) / (grouped["count"] + self.smoothing)
        self._encoding_map = encoded.to_dict()
        self._fitted = True
        return self

    # Applique la table target encoding apprise.
    def encode(self, df, column):
        if not self.is_fitted():
            raise RuntimeError("Appelez fit() avant encode()")

        encoded_df = df.copy()
        encoded_df[column] = (
            encoded_df[column].map(self._encoding_map).fillna(self._global_mean)
        )
        encoded_df = encoded_df.rename(columns={column: f"{column}_target_enc"})
        return encoded_df


# Étape pipeline qui délègue l'encodage à une stratégie injectable.
class FeatureEngineer(BasePipelineStep):
    """Applique une stratégie d'encodage sur une colonne donnée."""

    name = "feature_engineer"
    version = "1.0.0"

    # Injecte la stratégie et le contexte de colonne/cible.
    def __init__(self, strategy: EncodingStrategy, column: str, target: str = None):
        self.strategy = strategy
        self.column = column
        self.target = target
        self._logger = logging.getLogger(self.__class__.__name__)

    # Entraîne la stratégie puis encode la colonne.
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.column not in df.columns:
            raise ValueError(f"Colonne '{self.column}' absente du DataFrame")

        self.strategy.fit(df, self.column, self.target)
        encoded_df = self.strategy.encode(df, self.column)
        self._logger.info(
            "Encodage '%s' appliqué sur '%s'",
            self.strategy.__class__.__name__,
            self.column,
        )
        return encoded_df

    # Permet de changer dynamiquement la stratégie d'encodage.
    def set_strategy(self, strategy: EncodingStrategy) -> None:
        self.strategy = strategy
        self._logger.info("Stratégie changée pour '%s'", strategy.__class__.__name__)

    # Représentation utile de l'état courant de l'ingénieur de features.
    def __repr__(self):
        return (
            f"<FeatureEngineer column='{self.column}' "
            f"strategy='{self.strategy.__class__.__name__}'>"
        )


if __name__ == "__main__":
    # Jeu de données simulé proche du contexte NYC Taxi.
    df = pd.DataFrame(
        {
            "PULocationID": np.random.randint(1, 265, size=10000),
            "tip_pct": np.random.uniform(0, 0.5, size=10000),
            "fare_amount": np.random.uniform(5, 50, size=10000),
        }
    )

    # Cas 1 — OneHot.
    engineer = FeatureEngineer(OneHotStrategy(max_categories=10), column="PULocationID")
    df_oh = engineer.run(df.copy())
    created_cols = [c for c in df_oh.columns if c.startswith("PULocationID_")]
    print(f"[OneHot] shape avant={df.shape}, après={df_oh.shape}")
    print(f"[OneHot] nouvelles colonnes: {created_cols[:10]} ... total={len(created_cols)}")

    # Cas 2 — Frequency.
    engineer.set_strategy(FrequencyEncodingStrategy())
    df_freq = engineer.run(df.copy())
    print(df_freq[["PULocationID_freq"]].describe())

    # Cas 3 — Target Encoding.
    engineer.set_strategy(TargetEncodingStrategy(smoothing=5.0))
    engineer.target = "tip_pct"
    df_te = engineer.run(df.copy())
    print(df_te[["PULocationID_target_enc"]].describe())

    # Cas 4 — Changement de stratégie à chaud.
    print(f"Avant hot-swap: {engineer!r}")
    engineer.set_strategy(OneHotStrategy(max_categories=5))
    print(f"Après hot-swap: {engineer!r}")

    # Cas 5 — Erreur encode sans fit.
    strat = OneHotStrategy()
    try:
        strat.encode(df, "PULocationID")
    except RuntimeError as error:
        print(f"[ATTENDU] {error}")
