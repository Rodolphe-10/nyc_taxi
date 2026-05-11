"""Tests unitaires pour les descripteurs de validation."""

import pytest

from src.core.descriptors import BoundedFloat, ModelConfig, NotEmpty, OneOf, Positive, TypedAttr


class _BoundedDummy:
    """Classe de test pour BoundedFloat."""

    x = BoundedFloat(0.0, 1.0)


class _PositiveDummy:
    """Classe de test pour Positive."""

    x = Positive(min_value=10)


class _OneOfDummy:
    """Classe de test pour OneOf."""

    x = OneOf("regression", "binary", "multiclass")


class _TypedDummy:
    """Classe de test pour TypedAttr."""

    x = TypedAttr(int)


class _NotEmptyDummy:
    """Classe de test pour NotEmpty."""

    x = NotEmpty()


def test_bounded_float_valid():
    obj = _BoundedDummy()
    obj.x = 0.5
    assert obj.x == 0.5


def test_bounded_float_out_of_range():
    obj = _BoundedDummy()
    with pytest.raises(ValueError, match=r"^x doit être dans \[0\.0, 1\.0\], reçu 1\.5$"):
        obj.x = 1.5


def test_bounded_float_wrong_type():
    obj = _BoundedDummy()
    with pytest.raises(TypeError, match=r"^x doit être un nombre, reçu str$"):
        obj.x = "abc"


def test_positive_valid():
    obj = _PositiveDummy()
    obj.x = 11
    assert obj.x == 11


def test_positive_too_small():
    obj = _PositiveDummy()
    with pytest.raises(ValueError, match=r"^x doit être > 10, reçu 10$"):
        obj.x = 10


def test_positive_wrong_type():
    obj = _PositiveDummy()
    with pytest.raises(TypeError, match=r"^x doit être un nombre, reçu str$"):
        obj.x = "12"


def test_one_of_valid():
    obj = _OneOfDummy()
    obj.x = "regression"
    assert obj.x == "regression"


def test_one_of_invalid():
    obj = _OneOfDummy()
    with pytest.raises(
        ValueError,
        match=r"^x doit être parmi \('regression', 'binary', 'multiclass'\), reçu 'clustering'$",
    ):
        obj.x = "clustering"


def test_typed_attr_valid():
    obj = _TypedDummy()
    obj.x = 6
    assert obj.x == 6


def test_typed_attr_invalid():
    obj = _TypedDummy()
    with pytest.raises(TypeError, match=r"^x doit être de type int, reçu float$"):
        obj.x = 6.5


def test_not_empty_valid():
    obj = _NotEmptyDummy()
    obj.x = "LightGBM_v1"
    assert obj.x == "LightGBM_v1"


def test_not_empty_empty_string():
    obj = _NotEmptyDummy()
    with pytest.raises(ValueError, match=r"^x ne peut pas être vide$"):
        obj.x = ""


def test_model_config_valid():
    cfg = ModelConfig(
        learning_rate=0.05,
        n_estimators=500,
        objective="regression",
        max_depth=6,
        model_name="LightGBM_v1",
    )
    params = cfg.to_dict()
    expected_keys = {
        "learning_rate",
        "n_estimators",
        "objective",
        "max_depth",
        "model_name",
        "random_state",
    }
    assert expected_keys.issubset(params.keys())


def test_model_config_invalid_learning_rate():
    with pytest.raises(
        ValueError,
        match=r"^learning_rate doit être dans \[0\.0, 1\.0\], reçu 2\.0$",
    ):
        ModelConfig(
            learning_rate=2.0,
            n_estimators=500,
            objective="regression",
            max_depth=6,
            model_name="LightGBM_v1",
        )


def test_model_config_invalid_objective():
    with pytest.raises(
        ValueError,
        match=r"^objective doit être parmi \('regression', 'binary', 'multiclass'\), reçu 'deep_learning'$",
    ):
        ModelConfig(
            learning_rate=0.05,
            n_estimators=500,
            objective="deep_learning",
            max_depth=6,
            model_name="LightGBM_v1",
        )


def test_bounded_float_init_invalid():
    with pytest.raises(ValueError, match=r"^low doit être strictement inférieur à high$"):
        BoundedFloat(1.0, 0.0)
