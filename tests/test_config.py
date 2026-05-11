"""Tests unitaires des implémentations Singleton et de Config."""

import pytest

from src.core.config import Config, SingletonMeta, SingletonMeta_, SingletonNew


def test_singleton_new_same_instance():
    SingletonNew.reset()
    c1 = SingletonNew("a")
    c2 = SingletonNew("b")
    assert c1 is c2
    assert c1.value == "a"


def test_singleton_meta_same_instance():
    SingletonMeta.reset(SingletonMeta_)
    s1 = SingletonMeta_("x")
    s2 = SingletonMeta_("y")
    assert s1 is s2


def test_config_is_singleton():
    SingletonMeta.reset(Config)
    c1 = Config()
    c2 = Config()
    assert c1 is c2


def test_config_readonly_properties():
    SingletonMeta.reset(Config)
    config = Config()
    with pytest.raises(AttributeError):
        config.seed = 99
    with pytest.raises(AttributeError):
        config.raw_data_path = "/tmp"


def test_config_get_nested_key():
    SingletonMeta.reset(Config)
    config = Config()
    val = config.get("project.seed", 0)
    assert isinstance(val, int)


def test_config_all_returns_copy():
    SingletonMeta.reset(Config)
    config = Config()
    data = config.all()
    data["project"]["seed"] = 9999
    assert config.seed != 9999


def test_config_repr():
    SingletonMeta.reset(Config)
    config = Config()
    assert "Config" in repr(config)
    assert "version" in repr(config)
