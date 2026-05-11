"""Tests unitaires des décorateurs avancés."""

import logging
import time

import pytest

from src.core.decorators import log_calls, memoize_dataframe, retry, timeit


def test_timeit_without_parentheses():
    @timeit
    def f():
        return 42

    assert f() == 42


def test_timeit_with_unit():
    @timeit("s")
    def f():
        time.sleep(0.01)

    f()
    assert hasattr(f, "_last_duration")


def test_timeit_preserves_return_value():
    @timeit
    def f():
        return [1, 2, 3]

    assert f() == [1, 2, 3]


def test_log_calls_logs_call(caplog):
    with caplog.at_level(logging.INFO):
        @log_calls(log_args=True)
        def f(x):
            return x * 2

        f(5)
    assert "f" in caplog.text


def test_log_calls_reraises_exception():
    @log_calls()
    def f():
        raise ValueError("test")

    with pytest.raises(ValueError):
        f()


def test_retry_succeeds_eventually():
    counter = {"n": 0}

    @retry(max_attempts=3, backoff=0.01, exceptions=(IOError,))
    def f():
        counter["n"] += 1
        if counter["n"] < 3:
            raise IOError("fail")
        return "ok"

    assert f() == "ok"
    assert counter["n"] == 3


def test_retry_raises_after_max_attempts():
    @retry(max_attempts=2, backoff=0.01, exceptions=(IOError,))
    def f():
        raise IOError("toujours échoue")

    with pytest.raises(IOError):
        f()


def test_retry_does_not_catch_other_exceptions():
    @retry(max_attempts=3, backoff=0.01, exceptions=(IOError,))
    def f():
        raise ValueError("pas capturé")

    with pytest.raises(ValueError):
        f()


def test_memoize_dataframe_cache_hit():
    import pandas as pd

    @memoize_dataframe(maxsize=2)
    def f(df):
        return df.sum()

    df = pd.DataFrame({"a": [1, 2, 3]})
    f(df)
    f(df)
    assert f.cache_info()["hits"] == 2


def test_memoize_dataframe_different_dfs():
    import pandas as pd

    @memoize_dataframe(maxsize=2)
    def f(df):
        return df.mean()

    df1 = pd.DataFrame({"a": [1, 2]})
    df2 = pd.DataFrame({"a": [3, 4]})
    f(df1)
    f(df2)
    assert f.cache_info()["misses"] == 2


def test_memoize_dataframe_lru_eviction():
    import pandas as pd

    @memoize_dataframe(maxsize=2)
    def f(df):
        return df.max()

    for i in range(3):
        f(pd.DataFrame({"a": [i]}))
    assert f.cache_info()["size"] <= 2


def test_functools_wraps_preserved():
    @timeit
    def ma_fonction():
        """Ma docstring"""

    assert ma_fonction.__name__ == "ma_fonction"
    assert ma_fonction.__doc__ == "Ma docstring"


def test_decorator_chaining():
    @retry(max_attempts=2, backoff=0.01)
    @timeit
    @log_calls()
    def f(x):
        return x * 2

    assert f(5) == 10
