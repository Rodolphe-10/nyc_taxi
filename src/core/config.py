"""Implémentations Singleton et configuration centrale du pipeline."""

import copy
import os

import yaml


# Méthode A : Singleton via __new__.
# Avantage : simple à lire et rapide à mettre en place.
# Inconvénient : la logique est collée à la classe, moins pratique à mutualiser.
class SingletonNew:
    """Singleton basé sur __new__ avec garde de réinitialisation."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, value=None):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self.value = value

    @classmethod
    def reset(cls):
        cls._instance = None


# Méthode B : Singleton via métaclasse.
# Avantage : logique isolée/réutilisable, meilleure testabilité (reset par classe).
# Inconvénient : un peu plus abstrait pour les débutants.
class SingletonMeta(type):
    """Métaclasse qui garantit une unique instance par classe cible."""

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

    @staticmethod
    def reset(cls):
        if cls in SingletonMeta._instances:
            del SingletonMeta._instances[cls]


class SingletonMeta_(metaclass=SingletonMeta):
    """Classe de démonstration pour le Singleton via métaclasse."""

    def __init__(self, value=None):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self.value = value


class Config(metaclass=SingletonMeta):
    """Configuration centrale du pipeline, exposée en lecture seule."""

    def __init__(self, config_path="config/config.yaml"):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self._config_path = config_path
        self._data = {}
        self._load()

    # Charge le YAML ou applique une configuration par défaut si fichier absent.
    def _load(self) -> None:
        default_config = {
            "project": {
                "name": "NYC Taxi Tip Predictor",
                "version": "1.0.0",
                "seed": 42,
            },
            "paths": {
                "raw_data": "data/raw",
                "processed_data": "data/processed",
                "features_data": "data/features",
                "mlruns": "mlruns",
            },
            "data": {
                "year": 2023,
                "base_url": "https://d37ci6vzurychx.cloudfront.net/trip-data",
                "filename_pattern": "yellow_tripdata_{year}-{month:02d}.parquet",
            },
            "model": {
                "target": "tip_pct",
                "learning_rate": 0.05,
                "n_estimators": 500,
                "max_depth": 6,
                "objective": "regression",
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "model_name": "TipPredictor",
                "model_stage": "Production",
            },
        }

        if os.path.exists(self._config_path):
            with open(self._config_path, "r", encoding="utf-8") as file:
                loaded = yaml.safe_load(file) or {}
            self._data = loaded if loaded else copy.deepcopy(default_config)
            return

        self._data = copy.deepcopy(default_config)

    @property
    def raw_data_path(self) -> str:
        return self._data["paths"]["raw_data"]

    @raw_data_path.setter
    def raw_data_path(self, value):
        raise AttributeError("'raw_data_path' est en lecture seule")

    @property
    def processed_data_path(self) -> str:
        return self._data["paths"]["processed_data"]

    @processed_data_path.setter
    def processed_data_path(self, value):
        raise AttributeError("'processed_data_path' est en lecture seule")

    @property
    def features_data_path(self) -> str:
        return self._data["paths"]["features_data"]

    @features_data_path.setter
    def features_data_path(self, value):
        raise AttributeError("'features_data_path' est en lecture seule")

    @property
    def mlruns_path(self) -> str:
        return self._data["paths"]["mlruns"]

    @mlruns_path.setter
    def mlruns_path(self, value):
        raise AttributeError("'mlruns_path' est en lecture seule")

    @property
    def seed(self) -> int:
        return self._data["project"]["seed"]

    @seed.setter
    def seed(self, value):
        raise AttributeError("'seed' est en lecture seule")

    @property
    def project_name(self) -> str:
        return self._data["project"]["name"]

    @project_name.setter
    def project_name(self, value):
        raise AttributeError("'project_name' est en lecture seule")

    @property
    def version(self) -> str:
        return self._data["project"]["version"]

    @version.setter
    def version(self, value):
        raise AttributeError("'version' est en lecture seule")

    @property
    def base_url(self) -> str:
        return self._data["data"]["base_url"]

    @base_url.setter
    def base_url(self, value):
        raise AttributeError("'base_url' est en lecture seule")

    @property
    def data_year(self) -> int:
        return self._data["data"]["year"]

    @data_year.setter
    def data_year(self, value):
        raise AttributeError("'data_year' est en lecture seule")

    @property
    def model_target(self) -> str:
        return self._data["model"]["target"]

    @model_target.setter
    def model_target(self, value):
        raise AttributeError("'model_target' est en lecture seule")

    @property
    def api_host(self) -> str:
        return self._data["api"]["host"]

    @api_host.setter
    def api_host(self, value):
        raise AttributeError("'api_host' est en lecture seule")

    @property
    def api_port(self) -> int:
        return self._data["api"]["port"]

    @api_port.setter
    def api_port(self, value):
        raise AttributeError("'api_port' est en lecture seule")

    def __repr__(self) -> str:
        return f"<Config project='{self.project_name}' version='{self.version}' seed={self.seed}>"

    # Accès à une clé imbriquée avec notation pointée : ex. "model.learning_rate".
    def get(self, key: str, default=None):
        current = self._data
        for part in key.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    # Retourne une copie profonde pour éviter les modifications externes.
    def all(self) -> dict:
        return copy.deepcopy(self._data)


if __name__ == "__main__":
    # Section A — Démonstration Singleton via __new__.
    SingletonNew.reset()
    c1 = SingletonNew("première instance")
    c2 = SingletonNew("deuxième tentative")
    print(f"c1 is c2 -> {c1 is c2}")
    print(f"c1.value -> {c1.value!r}")

    # Section B — Démonstration Singleton via métaclasse.
    SingletonMeta.reset(SingletonMeta_)
    s1 = SingletonMeta_("première instance")
    s2 = SingletonMeta_("deuxième tentative")
    print(f"s1 is s2 -> {s1 is s2}")
    print(f"s1.value -> {s1.value!r}")

    # Section C — Configuration réelle du pipeline.
    SingletonMeta.reset(Config)
    config = Config()
    print(config)
    print(config.raw_data_path)
    print(config.seed)
    print(config.get("model.learning_rate", 0.05))
    try:
        config.seed = 99
    except AttributeError as exc:
        print(exc)
    config2 = Config()
    print(f"config is config2 -> {config is config2}")

    print(
        "Méthode A (__new__) vs Méthode B (métaclasse) :\n"
        "- __new__ : simple, mais fragile si héritage multiple\n"
        "- Métaclasse : plus robuste, isole la logique Singleton\n"
        "  dans la métaclasse, meilleure testabilité via reset()"
    )
