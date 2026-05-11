"""Tests unitaires des loaders et de la factory."""

import pandas as pd
import pytest

from src.core.metaclasses import BasePipelineStep
from src.pipeline.loaders import (
    CSVLoader,
    JSONLoader,
    LoaderFactory,
    ParquetLoader,
    BaseLoader,
    _LOADER_REGISTRY,
)


def test_registry_contains_all_formats():
    assert ".parquet" in _LOADER_REGISTRY
    assert ".csv" in _LOADER_REGISTRY
    assert ".json" in _LOADER_REGISTRY


def test_factory_creates_correct_loader(tmp_path):
    file_path = tmp_path / "data.csv"
    file_path.write_text("a,b\n1,2\n", encoding="utf-8")
    loader = LoaderFactory.create(str(file_path))
    assert isinstance(loader, CSVLoader)


def test_factory_unknown_extension_raises():
    with pytest.raises(ValueError, match="non supporté"):
        LoaderFactory.create("fichier.xlsx")


def test_parquet_loader_file_not_found():
    loader = ParquetLoader("inexistant.parquet")
    with pytest.raises(FileNotFoundError):
        loader.run()


def test_csv_loader_loads_correctly(tmp_path):
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    file_path = tmp_path / "rows.csv"
    df.to_csv(file_path, index=False)

    loader = LoaderFactory.create(str(file_path))
    result = loader.run()

    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_json_loader_loads_correctly(tmp_path):
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    file_path = tmp_path / "rows.json"
    df.to_json(file_path, orient="records")

    loader = LoaderFactory.create(str(file_path))
    result = loader.run()

    assert isinstance(result, pd.DataFrame)


def test_register_custom_loader(tmp_path):
    class CustomLoader(BaseLoader):
        name = "custom_loader"
        supported_extension = ".custom"

        def run(self, df=None):
            return super().run(df=df)

        def _load(self):
            return pd.DataFrame({"ok": [1]})

    LoaderFactory.register(".custom", CustomLoader)
    assert ".custom" in LoaderFactory.supported_formats()

    custom_path = tmp_path / "test.custom"
    custom_path.write_text("dummy", encoding="utf-8")
    loader = LoaderFactory.create(str(custom_path))
    result = loader.run()
    assert isinstance(result, pd.DataFrame)


def test_loader_repr():
    loader = CSVLoader("test.csv")
    text = repr(loader)
    assert "CSVLoader" in text
    assert "test.csv" in text


def test_base_loader_is_pipeline_step(tmp_path):
    file_path = tmp_path / "data.parquet"
    file_path.write_text("placeholder", encoding="utf-8")
    loader = ParquetLoader(str(file_path))
    assert isinstance(loader, BasePipelineStep)
