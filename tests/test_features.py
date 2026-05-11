"""Tests optimisation mémoire (dtypes) et streaming Parquet."""

import numpy as np
import pandas as pd
import pytest

from src.features.dask_processing import (
    DASK_AVAILABLE,
    benchmark_pandas_vs_dask,
    generate_dask_dataframe,
)
from src.features.temporal import (
    HOLIDAYS_AVAILABLE,
    TemporalFeaturizer,
    decompose_seasonality,
    encode_cyclic,
)
from src.features.timeseries import (
    STATSMODELS_AVAILABLE,
    TimeSeriesFeaturizer,
    adf_test,
    add_differencing,
    add_fourier_features,
    add_lag_features,
    add_rolling_features,
    aggregate_hourly,
)
from src.features.encoding import (
    TORCH_AVAILABLE,
    BayesianTargetEncoder,
    CrossFittedTargetEncoder,
    compare_smoothing_effect,
    train_zone_embeddings,
)
from src.features.imputation import (
    ITERATIVE_AVAILABLE,
    ImputationBenchmark,
    analyze_missingness,
    evaluate_imputation,
    mask_values,
)

from src.features.windowing import (
    agg_with_custom_functions,
    build_activity_multiindex,
    build_pivot_table,
    compute_all_window_features,
    compute_expanding_tip_pct,
    compute_fare_rank,
    compute_lag_features,
    compute_rolling_tip,
    prepare_dataframe,
    stack_unstack_demo,
    top_zones_by_tip,
    trimmed_mean,
)
from src.optim.benchmarks import generate_taxi_df
from src.optim.memory import (
    DTYPE_OPTIMIZATION_MAP,
    compute_stats_streaming,
    iter_parquet_batches,
    memory_report,
    optimize_dtypes,
)


def test_optimize_dtypes_reduces_memory():
    df = generate_taxi_df(n_rows=10_000)
    ram_avant = df.memory_usage(deep=True).sum()
    df_opt = optimize_dtypes(df.copy())
    ram_apres = df_opt.memory_usage(deep=True).sum()
    assert ram_apres < ram_avant
    gain = (ram_avant - ram_apres) / ram_avant * 100
    assert gain >= 30


def test_optimize_dtypes_preserves_values():
    df = generate_taxi_df(n_rows=1_000)
    df_opt = optimize_dtypes(df.copy())
    for col in ["fare_amount", "trip_distance"]:
        if col in df.columns and col in df_opt.columns:
            np.testing.assert_allclose(
                df[col].values,
                df_opt[col].values.astype("float64"),
                rtol=1e-3,
            )


def test_optimize_dtypes_int_columns():
    df = generate_taxi_df(n_rows=1_000)
    df_opt = optimize_dtypes(df.copy())
    if "PULocationID" in df_opt.columns:
        assert df_opt["PULocationID"].dtype in [
            np.dtype("int8"),
            np.dtype("int16"),
            np.dtype("int32"),
        ]


def test_optimize_dtypes_no_nulls_introduced():
    df = generate_taxi_df(n_rows=1_000)
    nulls_avant = df.isnull().sum().sum()
    df_opt = optimize_dtypes(df.copy())
    nulls_apres = df_opt.isnull().sum().sum()
    assert nulls_apres == nulls_avant


def test_memory_report_returns_dataframe():
    df = generate_taxi_df(n_rows=1_000)
    report = memory_report(df)
    assert isinstance(report, pd.DataFrame)
    assert "colonne" in report.columns
    assert "dtype" in report.columns
    assert "mo" in report.columns
    assert len(report) == len(df.columns)


def test_memory_report_sorted_by_size():
    df = generate_taxi_df(n_rows=1_000)
    report = memory_report(df)
    assert report["mo"].is_monotonic_decreasing


def test_iter_parquet_batches_with_real_file(tmp_path):
    df = generate_taxi_df(n_rows=5_000)
    path = str(tmp_path / "test.parquet")
    df.to_parquet(path, index=False)

    batches = list(iter_parquet_batches(path, batch_size=1_000))
    assert len(batches) == 5
    total_rows = sum(len(b) for b in batches)
    assert total_rows == 5_000


def test_iter_parquet_batches_optimized_dtypes(tmp_path):
    df = generate_taxi_df(n_rows=2_000)
    path = str(tmp_path / "test.parquet")
    df.to_parquet(path, index=False)

    for batch in iter_parquet_batches(path, batch_size=500):
        _ = batch.memory_usage(deep=True).sum()
        assert isinstance(batch, pd.DataFrame)
        assert len(batch) <= 500


def test_compute_stats_streaming(tmp_path):
    df = generate_taxi_df(n_rows=3_000)
    path = str(tmp_path / "test.parquet")
    df.to_parquet(path, index=False)

    stats = compute_stats_streaming(path, columns=["fare_amount", "trip_distance"])
    assert stats["total_rows"] == 3_000
    assert "fare_amount" in stats["means"]
    assert "trip_distance" in stats["mins"]
    np.testing.assert_allclose(
        stats["means"]["fare_amount"],
        df["fare_amount"].mean(),
        rtol=1e-3,
    )


def test_dtype_optimization_map_valid():
    for col, dtype in DTYPE_OPTIMIZATION_MAP.items():
        assert isinstance(col, str)
        assert isinstance(dtype, str)


def test_prepare_dataframe_adds_columns():
    df = generate_taxi_df(n_rows=1_000)
    result = prepare_dataframe(df)
    for col in ["pickup_hour", "pickup_weekday", "tip_pct", "duration_min"]:
        assert col in result.columns


def test_prepare_dataframe_filters_aberrants():
    df = generate_taxi_df(n_rows=1_000)
    result = prepare_dataframe(df)
    assert (result["fare_amount"] >= 2.5).all()
    assert (result["trip_distance"] >= 0.1).all()
    assert (result["passenger_count"] >= 1).all()


def test_build_activity_multiindex_shape():
    df = prepare_dataframe(generate_taxi_df(n_rows=5_000))
    activity = build_activity_multiindex(df)
    assert isinstance(activity.index, pd.MultiIndex)
    assert activity.index.names == ["PULocationID", "pickup_hour", "pickup_weekday"]


def test_build_activity_multiindex_columns():
    df = prepare_dataframe(generate_taxi_df(n_rows=5_000))
    activity = build_activity_multiindex(df)
    for col in ["trip_count", "revenue", "tip_pct_mean"]:
        assert col in activity.columns


def test_build_activity_trip_count_positive():
    df = prepare_dataframe(generate_taxi_df(n_rows=5_000))
    activity = build_activity_multiindex(df)
    assert (activity["trip_count"] > 0).all()


def test_build_pivot_table_shape():
    df = prepare_dataframe(generate_taxi_df(n_rows=5_000))
    pivot = build_pivot_table(df)
    assert isinstance(pivot, pd.DataFrame)
    assert isinstance(pivot.columns, pd.MultiIndex)
    assert pivot.shape[1] > 0


def test_stack_unstack_demo_returns_dict():
    df = prepare_dataframe(generate_taxi_df(n_rows=5_000))
    activity = build_activity_multiindex(df)
    demos = stack_unstack_demo(activity)
    assert "unstacked_hour" in demos
    assert "unstacked_weekday" in demos
    assert "swaplevel" in demos


def test_stack_unstack_hour_columns():
    df = prepare_dataframe(generate_taxi_df(n_rows=5_000))
    activity = build_activity_multiindex(df)
    demos = stack_unstack_demo(activity)
    cols = demos["unstacked_hour"].columns.tolist()
    assert all(0 <= int(c) <= 23 for c in cols)


def test_top_zones_by_tip_length():
    df = prepare_dataframe(generate_taxi_df(n_rows=5_000))
    activity = build_activity_multiindex(df)
    top10 = top_zones_by_tip(activity, top_n=10)
    assert len(top10) <= 10


def test_top_zones_sorted_descending():
    df = prepare_dataframe(generate_taxi_df(n_rows=5_000))
    activity = build_activity_multiindex(df)
    top10 = top_zones_by_tip(activity, top_n=10)
    assert top10["tip_pct_mean"].is_monotonic_decreasing


def test_trimmed_mean_basic():
    s = pd.Series([1, 2, 3, 4, 5, 100])
    result = trimmed_mean(s, pct=0.1)
    assert result < s.mean()


def test_trimmed_mean_no_truncation():
    s = pd.Series([1.0, 2.0, 3.0])
    assert trimmed_mean(s, pct=0.1) == s.mean()


def test_agg_with_custom_functions_columns():
    df = prepare_dataframe(generate_taxi_df(n_rows=5_000))
    result = agg_with_custom_functions(df)
    for col in ["tip_pct_mean", "tip_pct_trimmed", "fare_median", "trip_count"]:
        assert col in result.columns


def test_trimmed_less_than_mean_with_outlier():
    s = pd.Series([1.0] * 90 + [1000.0] * 10)
    assert trimmed_mean(s, pct=0.1) < s.mean()


def test_rolling_tip_column_created():
    df = prepare_dataframe(generate_taxi_df(n_rows=1_000))
    result = compute_rolling_tip(df, window=5)
    assert "tip_rolling_mean" in result.columns


def test_rolling_tip_no_data_leakage():
    """
    tip_rolling_mean doit utiliser shift(1) pour exclure la course
    courante. Vérifie qu'il n'y a pas de fuite de données.
    """
    df = prepare_dataframe(generate_taxi_df(n_rows=1_000))
    result = compute_rolling_tip(df, window=5)
    assert result["tip_rolling_mean"].notna().any()


def test_rolling_tip_values_reasonable():
    df = prepare_dataframe(generate_taxi_df(n_rows=5_000))
    result = compute_rolling_tip(df, window=5)
    valid = result["tip_rolling_mean"].dropna()
    assert (valid >= 0).all()
    assert (valid <= result["tip_amount"].max() * 2).all()


def test_fare_rank_column_created():
    df = prepare_dataframe(generate_taxi_df(n_rows=1_000))
    result = compute_fare_rank(df)
    assert "fare_rank_pct" in result.columns


def test_fare_rank_between_0_and_1():
    df = prepare_dataframe(generate_taxi_df(n_rows=2_000))
    result = compute_fare_rank(df)
    valid = result["fare_rank_pct"].dropna()
    assert (valid >= 0).all()
    assert (valid <= 1).all()


def test_expanding_tip_pct_column_created():
    df = prepare_dataframe(generate_taxi_df(n_rows=1_000))
    result = compute_expanding_tip_pct(df)
    assert "tip_pct_expanding" in result.columns


def test_expanding_converges():
    """
    L'expanding mean doit converger vers la vraie moyenne
    au fur et à mesure des observations.
    """
    df = prepare_dataframe(generate_taxi_df(n_rows=5_000))
    result = compute_expanding_tip_pct(df)
    valid = result["tip_pct_expanding"].dropna()
    assert len(valid) > 0
    assert (valid >= 0).all()


def test_lag_features_columns_created():
    df = prepare_dataframe(generate_taxi_df(n_rows=2_000))
    result = compute_lag_features(df, lags=[1, 3, 7])
    for lag in [1, 3, 7]:
        assert f"trip_count_lag_{lag}d" in result.columns


def test_lag_features_custom_lags():
    df = prepare_dataframe(generate_taxi_df(n_rows=2_000))
    result = compute_lag_features(df, lags=[2, 14])
    assert "trip_count_lag_2d" in result.columns
    assert "trip_count_lag_14d" in result.columns


def test_lag_features_no_future_leakage():
    """
    lag_1d = nombre de courses HIER, pas aujourd'hui.
    La première date de chaque zone doit avoir NaN pour lag_1d.
    """
    df = prepare_dataframe(generate_taxi_df(n_rows=5_000))
    result = compute_lag_features(df, lags=[1])
    assert result["trip_count_lag_1d"].isnull().any()


def test_compute_all_window_features_shape():
    df_raw = generate_taxi_df(n_rows=2_000)
    result = compute_all_window_features(df_raw, rolling_window=3, lags=[1, 7])
    assert result.shape[0] > 0
    assert result.shape[1] > df_raw.shape[1]


def test_compute_all_window_features_new_cols():
    df_raw = generate_taxi_df(n_rows=2_000)
    result = compute_all_window_features(df_raw, rolling_window=3, lags=[1, 7])
    expected = [
        "tip_rolling_mean",
        "fare_rank_pct",
        "tip_pct_expanding",
        "trip_count_lag_1d",
        "trip_count_lag_7d",
    ]
    for col in expected:
        assert col in result.columns, f"Colonne manquante : {col}"


def test_dask_available_import():
    """Le module doit s'importer même sans Dask (DASK_AVAILABLE booléen)."""
    from src.features.dask_processing import DASK_AVAILABLE as da

    assert isinstance(da, bool)


def test_generate_dask_dataframe_returns_data():
    result = generate_dask_dataframe(n_rows=10_000, n_partitions=2)
    if DASK_AVAILABLE:
        import dask.dataframe as dd

        assert isinstance(result, dd.DataFrame)
        df = result.compute()
        assert len(df) == 10_000
    else:
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10_000


def test_generate_dask_dataframe_columns():
    result = generate_dask_dataframe(n_rows=5_000, n_partitions=2)
    cols = result.columns.tolist()
    assert "fare_amount" in cols
    assert "trip_distance" in cols
    assert "tip_amount" in cols


def test_generate_dask_npartitions():
    if not DASK_AVAILABLE:
        pytest.skip("Dask non installé")
    result = generate_dask_dataframe(n_rows=10_000, n_partitions=4)
    assert result.npartitions == 4


def test_benchmark_pandas_vs_dask_returns_dict():
    results = benchmark_pandas_vs_dask(n_rows=50_000)
    assert "t_pandas" in results
    assert "n_rows" in results
    assert results["t_pandas"] > 0
    assert results["n_rows"] == 50_000


def test_benchmark_pandas_time_positive():
    results = benchmark_pandas_vs_dask(n_rows=10_000)
    assert results["t_pandas"] > 0


def test_benchmark_dask_time_if_available():
    if not DASK_AVAILABLE:
        pytest.skip("Dask non installé")
    results = benchmark_pandas_vs_dask(n_rows=50_000)
    assert results["t_dask"] is not None
    assert results["t_dask"] > 0


def test_benchmark_results_equal_if_dask():
    if not DASK_AVAILABLE:
        pytest.skip("Dask non installé")
    results = benchmark_pandas_vs_dask(n_rows=50_000)
    assert results["results_equal"] is True


def test_dask_dataframe_lazy_evaluation():
    if not DASK_AVAILABLE:
        pytest.skip("Dask non installé")
    import dask.dataframe as dd

    ddf = generate_dask_dataframe(n_rows=10_000, n_partitions=2)
    assert isinstance(ddf, dd.DataFrame)
    assert not isinstance(ddf, pd.DataFrame)


def test_encode_cyclic_period_24():
    series = pd.Series([0, 6, 12, 18, 23])
    sin_enc, cos_enc = encode_cyclic(series, 24)
    assert len(sin_enc) == 5
    assert len(cos_enc) == 5
    sin0, cos0 = encode_cyclic(pd.Series([0]), 24)
    sin24, cos24 = encode_cyclic(pd.Series([24]), 24)
    np.testing.assert_allclose(sin0.values, sin24.values, atol=1e-10)
    np.testing.assert_allclose(cos0.values, cos24.values, atol=1e-10)


def test_encode_cyclic_0_and_23_close():
    """0h et 23h doivent être proches dans l'espace sin/cos."""
    sin0, cos0 = encode_cyclic(pd.Series([0]), 24)
    sin23, cos23 = encode_cyclic(pd.Series([23]), 24)
    sin12, cos12 = encode_cyclic(pd.Series([12]), 24)
    dist_0_23 = np.sqrt((sin0.values[0] - sin23.values[0]) ** 2 + (cos0.values[0] - cos23.values[0]) ** 2)
    dist_0_12 = np.sqrt((sin0.values[0] - sin12.values[0]) ** 2 + (cos0.values[0] - cos12.values[0]) ** 2)
    assert dist_0_23 < dist_0_12


def test_encode_cyclic_range():
    series = pd.Series(range(24))
    sin_enc, cos_enc = encode_cyclic(series, 24)
    assert (sin_enc.between(-1, 1)).all()
    assert (cos_enc.between(-1, 1)).all()


def test_temporal_featurizer_is_pipeline_step():
    from src.core.metaclasses import BasePipelineStep

    featurizer = TemporalFeaturizer()
    assert isinstance(featurizer, BasePipelineStep)


def test_temporal_featurizer_calendar_features():
    df = generate_taxi_df(n_rows=1_000)
    featurizer = TemporalFeaturizer(cyclic_encoding=False, add_holidays=False)
    result = featurizer.run(df)
    for col in ["pickup_hour", "pickup_weekday", "pickup_month", "pickup_quarter", "is_weekend"]:
        assert col in result.columns, f"Colonne manquante : {col}"


def test_temporal_featurizer_cyclic_features():
    df = generate_taxi_df(n_rows=1_000)
    featurizer = TemporalFeaturizer(cyclic_encoding=True, add_holidays=False)
    result = featurizer.run(df)
    for col in [
        "pickup_hour_sin",
        "pickup_hour_cos",
        "pickup_weekday_sin",
        "pickup_weekday_cos",
        "pickup_month_sin",
        "pickup_month_cos",
    ]:
        assert col in result.columns


def test_temporal_featurizer_cyclic_range():
    df = generate_taxi_df(n_rows=1_000)
    featurizer = TemporalFeaturizer(cyclic_encoding=True, add_holidays=False)
    result = featurizer.run(df)
    assert result["pickup_hour_sin"].between(-1, 1).all()
    assert result["pickup_hour_cos"].between(-1, 1).all()


def test_temporal_featurizer_rush_hours():
    df = generate_taxi_df(n_rows=1_000)
    featurizer = TemporalFeaturizer(add_rush_hours=True, add_holidays=False)
    result = featurizer.run(df)
    assert "is_rush_hour" in result.columns
    assert "is_late_night" in result.columns
    assert result["is_rush_hour"].isin([0, 1]).all()
    assert result["is_late_night"].isin([0, 1]).all()


def test_temporal_featurizer_is_weekend_correct():
    df = generate_taxi_df(n_rows=1_000)
    featurizer = TemporalFeaturizer(add_holidays=False)
    result = featurizer.run(df)
    expected = (result["pickup_weekday"] >= 5).astype("int8")
    pd.testing.assert_series_equal(result["is_weekend"], expected, check_names=False)


def test_temporal_featurizer_holiday_features():
    if not HOLIDAYS_AVAILABLE:
        pytest.skip("holidays non installé")
    df = generate_taxi_df(n_rows=1_000)
    featurizer = TemporalFeaturizer(add_holidays=True)
    result = featurizer.run(df)
    assert "is_holiday" in result.columns
    assert "is_holiday_eve" in result.columns
    assert result["is_holiday"].isin([0, 1]).all()


def test_temporal_featurizer_execute_method():
    """execute() déclenche le pattern Template Method + Observer."""
    df = generate_taxi_df(n_rows=500)
    featurizer = TemporalFeaturizer(add_holidays=False)
    result = featurizer.execute(df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(df)


def test_temporal_featurizer_no_nulls_introduced():
    df = generate_taxi_df(n_rows=1_000)
    featurizer = TemporalFeaturizer(add_holidays=False)
    result = featurizer.run(df)
    new_cols = [c for c in result.columns if c not in df.columns]
    nulls_new = result[new_cols].isnull().sum().sum()
    assert nulls_new == 0


def test_decompose_seasonality_returns_dataframe():
    if not STATSMODELS_AVAILABLE:
        pytest.skip("statsmodels non installé")
    df = generate_taxi_df(n_rows=10_000)
    decomp = decompose_seasonality(df, freq="h", period=24)
    assert decomp is not None
    assert isinstance(decomp, pd.DataFrame)
    for col in ["observed", "trend", "seasonal", "residual"]:
        assert col in decomp.columns


def test_decompose_seasonality_no_statsmodels():
    """Sans statsmodels, doit retourner None sans erreur."""
    import src.features.temporal as temporal_module

    original = temporal_module.STATSMODELS_AVAILABLE
    temporal_module.STATSMODELS_AVAILABLE = False
    df = generate_taxi_df(n_rows=1_000)
    result = decompose_seasonality(df)
    temporal_module.STATSMODELS_AVAILABLE = original
    assert result is None


def _make_encoding_df(n_rows: int = 5_000) -> pd.DataFrame:
    df = generate_taxi_df(n_rows=n_rows).copy()
    df["tip_pct"] = (df["tip_amount"] / df["fare_amount"].clip(lower=0.01)).clip(0, 2)
    return df


def test_bayesian_encoder_is_pipeline_step():
    from src.core.metaclasses import BasePipelineStep

    encoder = BayesianTargetEncoder(columns=["PULocationID"], target="tip_pct")
    assert isinstance(encoder, BasePipelineStep)


def test_bayesian_fit_transform_adds_columns_and_drops_original():
    df = _make_encoding_df(3_000)
    encoder = BayesianTargetEncoder(
        columns=["PULocationID", "DOLocationID"],
        target="tip_pct",
        smoothing=10.0,
    )
    result = encoder.fit_transform(df)
    assert "PULocationID_target_enc" in result.columns
    assert "DOLocationID_target_enc" in result.columns
    assert "PULocationID" not in result.columns
    assert "DOLocationID" not in result.columns
    assert result["PULocationID_target_enc"].dtype == np.float32


def test_bayesian_transform_requires_fit():
    df = _make_encoding_df(500)
    encoder = BayesianTargetEncoder(columns=["PULocationID"], target="tip_pct")
    with pytest.raises(RuntimeError):
        encoder.transform(df)


def test_bayesian_unknown_category_falls_back_to_global_mean():
    train_df = _make_encoding_df(2_000)
    encoder = BayesianTargetEncoder(
        columns=["PULocationID"],
        target="tip_pct",
        smoothing=10.0,
        min_samples=5,
    ).fit(train_df)

    test_df = train_df.head(20).copy()
    test_df["PULocationID"] = 9999
    transformed = encoder.transform(test_df)
    expected = np.float32(encoder._global_means["PULocationID"])
    assert np.allclose(transformed["PULocationID_target_enc"].values, expected)


def test_bayesian_smoothing_effect_returns_dataframe():
    df = _make_encoding_df(2_000)
    encoder = BayesianTargetEncoder(columns=["PULocationID"], target="tip_pct").fit(df)
    effect = encoder.smoothing_effect("PULocationID")
    assert isinstance(effect, pd.DataFrame)
    for col in ["category", "encoded_value", "global_mean", "deviation"]:
        assert col in effect.columns


def test_cross_fitted_target_encoder_adds_column_no_nan():
    df = _make_encoding_df(4_000)
    encoder = CrossFittedTargetEncoder(
        columns=["PULocationID"],
        target="tip_pct",
        smoothing=10.0,
        n_splits=5,
    )
    result = encoder.fit_transform(df)
    assert "PULocationID_cross_enc" in result.columns
    assert result["PULocationID_cross_enc"].dtype == np.float32
    assert not result["PULocationID_cross_enc"].isna().any()


def test_cross_fitted_run_returns_dataframe():
    df = _make_encoding_df(1_500)
    encoder = CrossFittedTargetEncoder(columns=["PULocationID"], target="tip_pct", n_splits=3)
    result = encoder.run(df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(df)


def test_compare_smoothing_effect_returns_stats():
    df = _make_encoding_df(3_000)
    result = compare_smoothing_effect(
        df,
        col="PULocationID",
        target="tip_pct",
        smoothing_values=[0, 10, 1000],
    )
    assert isinstance(result, pd.DataFrame)
    for col in ["smoothing", "variance", "std", "min", "max", "range"]:
        assert col in result.columns
    # Avec fort smoothing, la dispersion doit en général diminuer.
    assert result.loc[result["smoothing"] == 1000, "variance"].iloc[0] <= result.loc[
        result["smoothing"] == 0, "variance"
    ].iloc[0]


def test_train_zone_embeddings_if_torch_available():
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch non installé")
    df = _make_encoding_df(2_000)
    model = train_zone_embeddings(
        df,
        zone_col="PULocationID",
        target="tip_pct",
        epochs=1,
        batch_size=512,
        lr=0.001,
    )
    assert model is not None
    embeddings = model.get_embeddings()
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] >= 266


def test_mask_values_creates_nans():
    df = generate_taxi_df(n_rows=2_000)
    df_masked, masks = mask_values(df, ["passenger_count", "trip_distance"], missing_pct=0.05)
    assert masks["passenger_count"]["n_masked"] > 0
    assert df_masked["passenger_count"].isnull().sum() > 0
    assert df_masked["trip_distance"].isnull().sum() > 0


def test_mask_values_correct_count():
    df = generate_taxi_df(n_rows=2_000)
    df_masked, masks = mask_values(df, ["trip_distance"], missing_pct=0.1)
    expected = int(2_000 * 0.1)
    assert masks["trip_distance"]["n_masked"] == expected
    assert df_masked["trip_distance"].isnull().sum() == expected


def test_mask_values_preserves_true_values():
    df = generate_taxi_df(n_rows=1_000)
    df_masked, masks = mask_values(df, ["fare_amount"], missing_pct=0.05)
    assert df_masked["fare_amount"].isnull().sum() > 0
    idx = masks["fare_amount"]["indices"]
    true_values = masks["fare_amount"]["true_values"]
    original_vals = df["fare_amount"].values[idx]
    np.testing.assert_array_equal(true_values, original_vals)


def test_evaluate_imputation_returns_rmse():
    df = generate_taxi_df(n_rows=1_000)
    df_masked, masks = mask_values(df, ["fare_amount"], missing_pct=0.05)
    df_imputed = df_masked.copy()
    df_imputed["fare_amount"] = df_masked["fare_amount"].fillna(df_masked["fare_amount"].median())
    results = evaluate_imputation(df_imputed, masks, df)
    assert "fare_amount" in results
    assert "rmse" in results["fare_amount"]
    assert results["fare_amount"]["rmse"] >= 0


def test_evaluate_perfect_imputation_zero_rmse():
    """Imputation parfaite doit donner RMSE = 0."""
    df = generate_taxi_df(n_rows=1_000)
    df_masked, masks = mask_values(df, ["fare_amount"], missing_pct=0.05)
    df_perfect = df_masked.copy()
    idx = masks["fare_amount"]["indices"]
    df_perfect.loc[df_perfect.index[idx], "fare_amount"] = masks["fare_amount"]["true_values"]
    results = evaluate_imputation(df_perfect, masks, df)
    assert results["fare_amount"]["rmse"] < 1e-10


def test_imputation_benchmark_is_pipeline_step():
    from src.core.metaclasses import BasePipelineStep

    bm = ImputationBenchmark(["passenger_count", "trip_distance"])
    assert isinstance(bm, BasePipelineStep)


def test_imputation_benchmark_run_returns_df():
    df = generate_taxi_df(n_rows=2_000)
    bm = ImputationBenchmark(
        columns=["passenger_count", "trip_distance"],
        missing_pct=0.05,
        n_neighbors=3,
        n_mice_iter=2,
    )
    result = bm.run(df)
    assert isinstance(result, pd.DataFrame)
    assert result["passenger_count"].isnull().sum() == 0
    assert result["trip_distance"].isnull().sum() == 0


def test_imputation_benchmark_summary_columns():
    df = generate_taxi_df(n_rows=2_000)
    bm = ImputationBenchmark(
        columns=["passenger_count"],
        missing_pct=0.05,
        n_mice_iter=2,
    )
    bm.run(df)
    summary = bm.summary()
    assert "strategie" in summary.columns
    assert "duration_ms" in summary.columns
    assert "rmse_passenger_count" in summary.columns


def test_knn_better_than_median_on_correlated_data():
    """Sur données corrélées, KNN doit battre la médiane."""
    np.random.seed(42)
    n = 3_000
    base = np.random.uniform(5, 50, n)
    df = pd.DataFrame(
        {
            "fare_amount": base,
            "trip_distance": base * 0.3 + np.random.normal(0, 0.5, n),
            "passenger_count": np.random.randint(1, 6, n).astype(float),
        }
    )
    df_masked, masks = mask_values(df, ["trip_distance"], missing_pct=0.1, seed=42)

    from sklearn.impute import KNNImputer, SimpleImputer

    numeric_cols = df_masked.select_dtypes(include=[np.number]).columns

    imp_median = SimpleImputer(strategy="median")
    df_median = pd.DataFrame(imp_median.fit_transform(df_masked[numeric_cols]), columns=numeric_cols)
    imp_knn = KNNImputer(n_neighbors=5)
    df_knn = pd.DataFrame(imp_knn.fit_transform(df_masked[numeric_cols]), columns=numeric_cols)

    rmse_median = evaluate_imputation(df_median, masks, df)
    rmse_knn = evaluate_imputation(df_knn, masks, df)
    assert rmse_knn["trip_distance"]["rmse"] < rmse_median["trip_distance"]["rmse"]


def test_analyze_missingness_returns_dict():
    df = generate_taxi_df(n_rows=1_000)
    df_masked, _ = mask_values(df, ["passenger_count", "trip_distance"], missing_pct=0.05)
    result = analyze_missingness(df_masked, ["passenger_count", "trip_distance"])
    assert "passenger_count" in result
    assert "trip_distance" in result
    assert result["passenger_count"]["n_missing"] > 0


def test_analyze_missingness_pct_correct():
    df = generate_taxi_df(n_rows=2_000)
    df_masked, _ = mask_values(df, ["fare_amount"], missing_pct=0.10)
    result = analyze_missingness(df_masked, ["fare_amount"])
    assert abs(result["fare_amount"]["pct_missing"] - 10.0) < 1.0


def test_aggregate_hourly_shape():
    df = generate_taxi_df(n_rows=5_000)
    result = aggregate_hourly(df)
    assert isinstance(result, pd.DataFrame)
    assert "pickup_hour_ts" in result.columns
    assert "trip_count" in result.columns
    assert "tip_mean" in result.columns
    assert "fare_mean" in result.columns


def test_aggregate_hourly_no_nulls():
    df = generate_taxi_df(n_rows=5_000)
    result = aggregate_hourly(df)
    assert result["trip_count"].isnull().sum() == 0
    assert result["fare_mean"].isnull().sum() == 0


def test_aggregate_hourly_trip_count_positive():
    df = generate_taxi_df(n_rows=5_000)
    result = aggregate_hourly(df)
    assert (result["trip_count"] > 0).all()


def test_add_lag_features_columns_created():
    df = generate_taxi_df(n_rows=5_000)
    df_hourly = aggregate_hourly(df)
    result = add_lag_features(df_hourly, lags=[1, 24])
    assert "trip_count_lag_1h" in result.columns
    assert "trip_count_lag_24h" in result.columns


def test_add_lag_features_has_nans():
    """Les premières valeurs de chaque groupe doivent être NaN."""
    df = generate_taxi_df(n_rows=5_000)
    df_hourly = aggregate_hourly(df)
    result = add_lag_features(df_hourly, lags=[1])
    assert result["trip_count_lag_1h"].isnull().any()


def test_add_rolling_features_columns():
    df = generate_taxi_df(n_rows=5_000)
    df_hourly = aggregate_hourly(df)
    result = add_rolling_features(df_hourly, windows=[3, 24])
    for w in [3, 24]:
        assert f"trip_count_roll_mean_{w}h" in result.columns
        assert f"trip_count_roll_std_{w}h" in result.columns
        assert f"trip_count_roll_min_{w}h" in result.columns
        assert f"trip_count_roll_max_{w}h" in result.columns


def test_add_rolling_features_no_leakage():
    """Rolling avec shift(1) — pas de fuite de la valeur courante."""
    df = generate_taxi_df(n_rows=5_000)
    df_hourly = aggregate_hourly(df)
    result = add_rolling_features(df_hourly, windows=[3])
    assert result["trip_count_roll_mean_3h"].isnull().any()


def test_add_differencing_columns():
    df = generate_taxi_df(n_rows=5_000)
    df_hourly = aggregate_hourly(df)
    result = add_differencing(df_hourly)
    assert "trip_count_diff_1" in result.columns
    assert "trip_count_diff_24" in result.columns


def test_add_differencing_has_nans():
    df = generate_taxi_df(n_rows=5_000)
    df_hourly = aggregate_hourly(df)
    result = add_differencing(df_hourly)
    assert result["trip_count_diff_1"].isnull().any()


def test_add_fourier_features_columns():
    df = generate_taxi_df(n_rows=5_000)
    df_hourly = aggregate_hourly(df)
    result = add_fourier_features(df_hourly, n_harmonics=3)
    for k in range(1, 4):
        assert f"fourier_sin_{k}" in result.columns
        assert f"fourier_cos_{k}" in result.columns


def test_add_fourier_features_range():
    df = generate_taxi_df(n_rows=5_000)
    df_hourly = aggregate_hourly(df)
    result = add_fourier_features(df_hourly, n_harmonics=5)
    for k in range(1, 6):
        assert result[f"fourier_sin_{k}"].between(-1, 1).all()
        assert result[f"fourier_cos_{k}"].between(-1, 1).all()


def test_add_fourier_n_harmonics():
    df = generate_taxi_df(n_rows=2_000)
    df_hourly = aggregate_hourly(df)
    for n in [3, 5]:
        result = add_fourier_features(df_hourly.copy(), n_harmonics=n)
        fourier_cols = [c for c in result.columns if "fourier" in c]
        assert len(fourier_cols) == 2 * n


def test_adf_test_returns_dict():
    if not STATSMODELS_AVAILABLE:
        pytest.skip("statsmodels absent")
    ts = pd.Series(np.random.randn(200))
    result = adf_test(ts, label="test")
    assert "adf_stat" in result
    assert "p_value" in result
    assert "is_stationary" in result


def test_adf_test_stationary_series():
    if not STATSMODELS_AVAILABLE:
        pytest.skip("statsmodels absent")
    np.random.seed(42)
    ts = pd.Series(np.random.randn(800))
    result = adf_test(ts, label="bruit_blanc")
    assert result["is_stationary"] is True


def test_adf_test_non_stationary_series():
    if not STATSMODELS_AVAILABLE:
        pytest.skip("statsmodels absent")
    np.random.seed(42)
    ts = pd.Series(np.random.randn(2_000).cumsum())
    result = adf_test(ts, label="marche_aleatoire")
    assert result["is_stationary"] is False


def test_timeseries_featurizer_fit_transform():
    df = generate_taxi_df(n_rows=10_000)
    featurizer = TimeSeriesFeaturizer(
        lags=[1, 24],
        windows=[3],
        n_harmonics=3,
    )
    result = featurizer.fit_transform(df)
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] > 0
    assert "trip_count" in result.columns
    assert "trip_count_lag_1h" in result.columns
    assert "trip_count_lag_24h" in result.columns
    assert "fourier_sin_1" in result.columns


def test_timeseries_featurizer_more_features_than_input():
    df = generate_taxi_df(n_rows=10_000)
    df_hourly = aggregate_hourly(df)
    featurizer = TimeSeriesFeaturizer(lags=[1], windows=[3], n_harmonics=2)
    result = featurizer.fit_transform(df)
    assert result.shape[1] > df_hourly.shape[1]
