"""Tests unitaires des benchmarks et de l'implémentation naïve."""

import time

import numpy as np
import pandas as pd
import pytest

from src.optim.benchmarks import (
    benchmark_function,
    compare_functions,
    compute_features_naive,
    generate_taxi_df,
    profile_with_memory,
)
from src.optim.vectorize import (
    compute_features_vectorized,
    compute_zone_distance_matrix,
    compute_zone_distance_matrix_haversine,
    optimize_dtypes,
)
from src.optim.jit_kernels import (
    haversine_python,
    haversine_numpy,
    haversine_numba,
    haversine_numba_parallel,
    rolling_weighted_median_numba,
    warmup_numba,
    benchmark_haversine,
    NUMBA_AVAILABLE,
)


def test_generate_taxi_df_shape():
    df = generate_taxi_df(n_rows=1000)
    assert len(df) == 1000
    assert "fare_amount" in df.columns
    assert "trip_distance" in df.columns


def test_generate_taxi_df_no_nulls():
    df = generate_taxi_df(n_rows=1000)
    assert df.isnull().sum().sum() == 0


def test_compute_features_naive_columns():
    df = generate_taxi_df(n_rows=100)
    result = compute_features_naive(df)
    assert "duration_min" in result.columns
    assert "speed_mph" in result.columns
    assert "penalty" in result.columns


def test_compute_features_naive_values():
    df = generate_taxi_df(n_rows=100)
    result = compute_features_naive(df)
    assert (result["duration_min"] > 0).all()
    assert (result["speed_mph"] > 0).all()
    assert (result["penalty"] > 0).all()


def test_benchmark_function_returns_stats():
    df = generate_taxi_df(n_rows=500)
    stats = benchmark_function(compute_features_naive, df, label="test", n_runs=2)
    assert "mean_ms" in stats
    assert "rows_per_second" in stats
    assert stats["mean_ms"] > 0


def test_profile_with_memory_returns_dict():
    df = generate_taxi_df(n_rows=500)
    result = profile_with_memory(compute_features_naive, df, n_rows=500)
    assert "peak_mb" in result
    assert result["peak_mb"] > 0


def test_compare_functions_returns_dataframe():
    df = generate_taxi_df(n_rows=500)
    funcs = {
        "naive": compute_features_naive,
        "naive_copy": lambda d: compute_features_naive(d.copy()),
    }
    result = compare_functions(funcs, df)
    assert isinstance(result, pd.DataFrame)
    assert "speedup" in result.columns
    assert len(result) == 2


def test_vectorized_same_results_as_naive():
    df = generate_taxi_df(n_rows=500)
    naive = compute_features_naive(df.copy())
    vect = compute_features_vectorized(df.copy())
    np.testing.assert_allclose(
        naive["duration_min"].values,
        vect["duration_min"].values,
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        naive["penalty"].values,
        vect["penalty"].values,
        rtol=1e-4,
    )


def test_vectorized_is_faster_than_naive():
    df = generate_taxi_df(n_rows=5_000)
    t0 = time.perf_counter()
    compute_features_naive(df.copy())
    t_naive = time.perf_counter() - t0

    t0 = time.perf_counter()
    compute_features_vectorized(df.copy())
    t_vect = time.perf_counter() - t0

    speedup = t_naive / t_vect
    assert speedup >= 10


def test_vectorized_columns_exist():
    df = generate_taxi_df(n_rows=100)
    result = compute_features_vectorized(df)
    for col in ["duration_min", "speed_mph", "penalty"]:
        assert col in result.columns


def test_vectorized_no_nulls():
    df = generate_taxi_df(n_rows=1000)
    result = compute_features_vectorized(df)
    assert result["duration_min"].isnull().sum() == 0
    assert result["speed_mph"].isnull().sum() == 0
    assert result["penalty"].isnull().sum() == 0


def test_optimize_dtypes_reduces_memory():
    df = generate_taxi_df(n_rows=10_000)
    ram_avant = df.memory_usage(deep=True).sum()
    df_opt = optimize_dtypes(df.copy())
    ram_apres = df_opt.memory_usage(deep=True).sum()
    assert ram_apres < ram_avant


def test_zone_distance_matrix_shape():
    coords = np.random.uniform(0, 1, size=(265, 2))
    mat = compute_zone_distance_matrix(coords)
    assert mat.shape == (265, 265)


def test_zone_distance_matrix_symmetric():
    coords = np.random.uniform(0, 1, size=(265, 2))
    mat = compute_zone_distance_matrix(coords)
    np.testing.assert_allclose(mat, mat.T)


def test_zone_distance_matrix_diagonal_zero():
    coords = np.random.uniform(0, 1, size=(265, 2))
    mat = compute_zone_distance_matrix(coords)
    np.testing.assert_allclose(np.diag(mat), 0.0)


def test_haversine_matrix_shape():
    coords = np.random.uniform(low=[40.5, -74.3], high=[40.9, -73.7], size=(265, 2))
    mat = compute_zone_distance_matrix_haversine(coords)
    assert mat.shape == (265, 265)


def test_haversine_matrix_positive():
    coords = np.random.uniform(low=[40.5, -74.3], high=[40.9, -73.7], size=(265, 2))
    mat = compute_zone_distance_matrix_haversine(coords)
    assert (mat >= 0).all()


def test_haversine_python_nyc_paris():
    dist = haversine_python(40.7128, -74.0060, 48.8566, 2.3522)
    assert abs(dist - 5837) < 50


def test_haversine_numpy_same_as_python():
    lat1 = np.array([40.7128, 40.6892])
    lon1 = np.array([-74.0060, -74.0445])
    lat2 = np.array([48.8566, 51.5074])
    lon2 = np.array([2.3522, -0.1278])
    result_np = haversine_numpy(lat1, lon1, lat2, lon2)
    for i in range(len(lat1)):
        py_dist = haversine_python(lat1[i], lon1[i], lat2[i], lon2[i])
        assert abs(result_np[i] - py_dist) < 0.1


def test_haversine_numpy_shape():
    n = 1000
    lat1 = np.random.uniform(40.5, 40.9, n)
    lon1 = np.random.uniform(-74.3, -73.7, n)
    result = haversine_numpy(lat1, lon1, lat1, lon1)
    assert result.shape == (n,)
    np.testing.assert_allclose(result, 0.0, atol=1e-6)


def test_haversine_zero_distance():
    lat = np.array([40.7128])
    lon = np.array([-74.0060])
    dist = haversine_numpy(lat, lon, lat, lon)
    assert abs(dist[0]) < 1e-6


def test_haversine_positive_distances():
    lat1 = np.random.uniform(40.5, 40.9, 100)
    lon1 = np.random.uniform(-74.3, -73.7, 100)
    lat2 = np.random.uniform(40.5, 40.9, 100)
    lon2 = np.random.uniform(-74.3, -73.7, 100)
    result = haversine_numpy(lat1, lon1, lat2, lon2)
    assert (result >= 0).all()


def test_haversine_numba_matches_numpy():
    if not NUMBA_AVAILABLE:
        pytest.skip("Numba non installé")
    lat1 = np.random.uniform(40.5, 40.9, 1000).astype(np.float64)
    lon1 = np.random.uniform(-74.3, -73.7, 1000).astype(np.float64)
    lat2 = np.random.uniform(40.5, 40.9, 1000).astype(np.float64)
    lon2 = np.random.uniform(-74.3, -73.7, 1000).astype(np.float64)
    warmup_numba()
    result_np = haversine_numpy(lat1, lon1, lat2, lon2)
    result_nb = haversine_numba(lat1, lon1, lat2, lon2)
    np.testing.assert_allclose(result_np, result_nb, rtol=1e-5)


def test_haversine_parallel_matches_serial():
    if not NUMBA_AVAILABLE:
        pytest.skip("Numba non installé")
    lat1 = np.random.uniform(40.5, 40.9, 1000).astype(np.float64)
    lon1 = np.random.uniform(-74.3, -73.7, 1000).astype(np.float64)
    lat2 = np.random.uniform(40.5, 40.9, 1000).astype(np.float64)
    lon2 = np.random.uniform(-74.3, -73.7, 1000).astype(np.float64)
    warmup_numba()
    result_serial = haversine_numba(lat1, lon1, lat2, lon2)
    result_parallel = haversine_numba_parallel(lat1, lon1, lat2, lon2)
    np.testing.assert_allclose(result_serial, result_parallel, rtol=1e-5)


def test_rolling_weighted_median_shape():
    if not NUMBA_AVAILABLE:
        pytest.skip("Numba non installé")
    n, w = 1000, 10
    values = np.random.uniform(0, 50, n).astype(np.float64)
    weights = np.random.uniform(0.1, 1.0, n).astype(np.float64)
    rolling_weighted_median_numba(values[:100], weights[:100], w)
    result = rolling_weighted_median_numba(values, weights, w)
    assert result.shape == (n,)
    assert np.isnan(result[: w - 1]).all()
    assert not np.isnan(result[w - 1 :]).any()


def test_benchmark_haversine_returns_dict():
    results = benchmark_haversine(n_points=10_000)
    assert "t_python_extrap" in results
    assert "t_numpy" in results
    assert "speedup_numpy" in results
    assert results["t_numpy"] > 0
