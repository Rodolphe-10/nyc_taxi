"""Loaders de données et factory de sélection par extension."""

# Import de la classe de base des étapes de pipeline.
from src.core.metaclasses import BasePipelineStep

# Import du décorateur timeit réel.
from src.core.decorators import timeit


# Import de la configuration centrale (utile pour intégration pipeline).
from src.core.config import Config

import logging
import os

import pandas as pd


# Registre global des loaders mappés par extension de fichier.
_LOADER_REGISTRY = {}


# Décorateur de classe pour enregistrer automatiquement un loader.
def register_loader(extension: str):
    """Enregistre une classe loader pour une extension donnée."""

    # Vérifie la validité de l'extension au format '.ext'.
    if not extension.startswith("."):
        raise ValueError("L'extension doit commencer par '.'")

    def decorator(cls):
        # Ajoute la classe au registre global et journalise l'opération.
        _LOADER_REGISTRY[extension] = cls
        logging.getLogger(__name__).debug(
            "Loader '%s' enregistré pour '%s'",
            cls.__name__,
            extension,
        )
        return cls

    return decorator


# Classe intermédiaire commune à tous les loaders concrets.
class BaseLoader(BasePipelineStep):
    """Base abstraite des loaders de fichiers."""

    name: str = "base_loader"
    version: str = "1.0.0"
    supported_extension: str = ""

    # Initialise le loader avec son chemin source et ses options.
    def __init__(self, path: str, **kwargs):
        self.path = path
        self.kwargs = kwargs
        self._logger = logging.getLogger(self.__class__.__name__)

    # Valide le chemin puis délègue le chargement à la sous-classe.
    def run(self, df=None) -> pd.DataFrame:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Fichier introuvable : {self.path}")
        return self._load()

    # Contrat de chargement spécifique à chaque format.
    def _load(self) -> pd.DataFrame:
        raise NotImplementedError("Implémentez _load() dans la sous-classe")

    # Représentation utile en debug.
    def __repr__(self):
        return f"<{self.__class__.__name__} path='{self.path}'>"


# Loader parquet enregistré automatiquement à l'import.
@register_loader(".parquet")
class ParquetLoader(BaseLoader):
    """Loader dédié aux fichiers Parquet."""

    name = "parquet_loader"
    supported_extension = ".parquet"

    # Point d'entrée compatible avec le contrat de la métaclasse.
    @timeit
    def run(self, df=None) -> pd.DataFrame:
        return super().run(df=df)

    # Charge un parquet avec l'engine pyarrow.
    def _load(self) -> pd.DataFrame:
        df = pd.read_parquet(self.path, engine="pyarrow", **self.kwargs)
        self._logger.info("Parquet chargé : %s — %s lignes", self.path, len(df))
        return df


# Loader CSV enregistré automatiquement à l'import.
@register_loader(".csv")
class CSVLoader(BaseLoader):
    """Loader dédié aux fichiers CSV."""

    name = "csv_loader"
    supported_extension = ".csv"

    # Point d'entrée compatible avec le contrat de la métaclasse.
    @timeit
    def run(self, df=None) -> pd.DataFrame:
        return super().run(df=df)

    # Charge un CSV avec les options passées au constructeur.
    def _load(self) -> pd.DataFrame:
        df = pd.read_csv(self.path, **self.kwargs)
        self._logger.info("CSV chargé : %s — %s lignes", self.path, len(df))
        return df


# Loader JSON enregistré automatiquement à l'import.
@register_loader(".json")
class JSONLoader(BaseLoader):
    """Loader dédié aux fichiers JSON."""

    name = "json_loader"
    supported_extension = ".json"

    # Point d'entrée compatible avec le contrat de la métaclasse.
    @timeit
    def run(self, df=None) -> pd.DataFrame:
        return super().run(df=df)

    # Charge un JSON avec les options passées au constructeur.
    def _load(self) -> pd.DataFrame:
        df = pd.read_json(self.path, **self.kwargs)
        self._logger.info("JSON chargé : %s — %s lignes", self.path, len(df))
        return df


# Factory utilitaire pour créer automatiquement le bon loader.
class LoaderFactory:
    """Fabrique de loaders basée sur l'extension du fichier."""

    # Crée le loader approprié selon l'extension du chemin.
    @classmethod
    def create(cls, path: str, **kwargs) -> BaseLoader:
        extension = os.path.splitext(path)[1].lower()
        if extension not in _LOADER_REGISTRY:
            raise ValueError(
                f"Format '{extension}' non supporté. "
                f"Formats disponibles : {list(_LOADER_REGISTRY.keys())}"
            )
        return _LOADER_REGISTRY[extension](path, **kwargs)

    # Retourne la liste des extensions supportées.
    @classmethod
    def supported_formats(cls) -> list:
        return list(_LOADER_REGISTRY.keys())

    # Enregistre manuellement un loader personnalisé.
    @classmethod
    def register(cls, extension: str, loader_class) -> None:
        _LOADER_REGISTRY[extension] = loader_class

    # Fournit une vue lisible extension -> nom de classe loader.
    @classmethod
    def registry_info(cls) -> dict:
        return {ext: loader_cls.__name__ for ext, loader_cls in _LOADER_REGISTRY.items()}


# Démonstration rapide du registre, de la factory et des erreurs attendues.
if __name__ == "__main__":
    # Référence légère à Config pour illustrer l'intégration au pipeline.
    _ = Config()

    print("Loaders enregistrés :", LoaderFactory.registry_info())

    # Test Factory sur fichier inexistant.
    try:
        loader = LoaderFactory.create("data/raw/test.parquet")
        _df = loader.run()
    except FileNotFoundError as error:
        print(f"[ATTENDU] {error}")

    # Test format non supporté.
    try:
        loader = LoaderFactory.create("data/raw/test.xlsx")
    except ValueError as error:
        print(f"[ATTENDU] {error}")

    # Test de chargement CSV réel temporaire.
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False, encoding="utf-8") as file:
        file.write("col1,col2\n1,2\n3,4\n")
        tmp_path = file.name

    loader = LoaderFactory.create(tmp_path)
    df = loader.run()
    print(f"[OK] CSV chargé : {df.shape}")
    os.unlink(tmp_path)
