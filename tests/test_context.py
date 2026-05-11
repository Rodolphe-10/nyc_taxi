"""Tests unitaires des gestionnaires de contexte du pipeline."""

import time

import pandas as pd
import pytest

from src.core.context import MemoryGuard, PipelineContext, Timer, temp_dtypes


def test_timer_measures_duration():
    with Timer("test") as t:
        time.sleep(0.05)
    assert t.elapsed is not None
    assert t.elapsed > 0


def test_timer_unit_ms():
    with Timer("test", unit="ms") as t:
        time.sleep(0.01)
    assert t.elapsed >= 10.0


def test_timer_unit_s():
    with Timer("test", unit="s") as t:
        time.sleep(0.01)
    assert t.elapsed < 1.0


def test_timer_does_not_suppress_exception():
    with pytest.raises(ValueError):
        with Timer("test") as t:
            raise ValueError("test")
    assert t.elapsed is not None


def test_timer_elapsed_accessible_after_exit():
    with Timer("test") as t:
        pass
    assert t.elapsed >= 0


def test_memory_guard_no_trigger():
    with MemoryGuard(threshold_mb=10000.0, label="test") as mg:
        _ = list(range(1000))
    assert mg.peak_mb is not None


def test_memory_guard_triggers():
    with pytest.raises(MemoryError, match="seuil"):
        with MemoryGuard(threshold_mb=0.0001, label="test"):
            _ = list(range(100000))


def test_temp_dtypes_changes_during_context():
    df = pd.DataFrame({"col": [1.0, 2.0, 3.0]})
    with temp_dtypes(df, {"col": "float32"}) as df_tmp:
        assert str(df_tmp["col"].dtype) == "float32"


def test_temp_dtypes_restores_after_context():
    df = pd.DataFrame({"col": [1.0, 2.0, 3.0]})
    original_dtype = str(df["col"].dtype)
    with temp_dtypes(df, {"col": "float32"}):
        pass
    assert str(df["col"].dtype) == original_dtype


def test_temp_dtypes_restores_on_exception():
    df = pd.DataFrame({"col": [1.0, 2.0, 3.0]})
    original_dtype = str(df["col"].dtype)
    try:
        with temp_dtypes(df, {"col": "float32"}):
            raise RuntimeError("erreur test")
    except RuntimeError:
        pass
    assert str(df["col"].dtype) == original_dtype


def test_pipeline_context_stores_metrics():
    with PipelineContext("test_step", time_unit="ms") as ctx:
        time.sleep(0.01)
    assert ctx.elapsed is not None
    assert ctx.peak_mb is not None
    assert ctx.elapsed > 0
