"""Tests unitaires des stratégies d'encodage et de FeatureEngineer."""

import pandas as pd
import pytest

from src.pipeline.strategies import (
    FeatureEngineer,
    FrequencyEncodingStrategy,
    OneHotStrategy,
    TargetEncodingStrategy,
)


def test_onehot_fit_encode():
    df = pd.DataFrame({"PULocationID": [1, 2, 1, 3], "tip_pct": [0.1, 0.2, 0.1, 0.3]})
    strat = OneHotStrategy(drop_first=False, max_categories=3).fit(df, "PULocationID")
    result = strat.encode(df, "PULocationID")

    assert "PULocationID" not in result.columns
    assert any(col.startswith("PULocationID_") for col in result.columns)


def test_frequency_encoding():
    df = pd.DataFrame({"PULocationID": [1, 1, 2, 3, 3, 3]})
    strat = FrequencyEncodingStrategy().fit(df, "PULocationID")
    result = strat.encode(df, "PULocationID")

    assert "PULocationID_freq" in result.columns
    assert result["PULocationID_freq"].between(0, 1).all()


def test_target_encoding_smoothing():
    df = pd.DataFrame(
        {
            "PULocationID": ["A", "A", "B", "B", "B"],
            "tip_pct": [0.1, 0.3, 0.8, 0.6, 0.7],
        }
    )

    no_smooth = TargetEncodingStrategy(smoothing=0).fit(df, "PULocationID", "tip_pct")
    strong_smooth = TargetEncodingStrategy(smoothing=1000).fit(df, "PULocationID", "tip_pct")

    result_no_smooth = no_smooth.encode(df, "PULocationID")
    result_strong_smooth = strong_smooth.encode(df, "PULocationID")

    mean_a = df[df["PULocationID"] == "A"]["tip_pct"].mean()
    mean_b = df[df["PULocationID"] == "B"]["tip_pct"].mean()
    global_mean = df["tip_pct"].mean()

    assert pytest.approx(result_no_smooth.loc[0, "PULocationID_target_enc"], rel=1e-8) == mean_a
    assert pytest.approx(result_no_smooth.loc[2, "PULocationID_target_enc"], rel=1e-8) == mean_b

    dist_a_strong = abs(result_strong_smooth.loc[0, "PULocationID_target_enc"] - global_mean)
    dist_b_strong = abs(result_strong_smooth.loc[2, "PULocationID_target_enc"] - global_mean)
    assert dist_a_strong < abs(mean_a - global_mean)
    assert dist_b_strong < abs(mean_b - global_mean)


def test_encode_without_fit_raises():
    df = pd.DataFrame({"col": ["a", "b"]})
    with pytest.raises(RuntimeError, match="fit\\(\\)"):
        OneHotStrategy().encode(df, "col")


def test_feature_engineer_run():
    df = pd.DataFrame({"PULocationID": [1, 2, 1], "tip_pct": [0.1, 0.2, 0.3]})
    engineer = FeatureEngineer(FrequencyEncodingStrategy(), "PULocationID")
    result = engineer.run(df)

    assert isinstance(result, pd.DataFrame)
    assert "PULocationID" not in result.columns
    assert "PULocationID_freq" in result.columns


def test_set_strategy_hot_swap():
    engineer = FeatureEngineer(OneHotStrategy(), "PULocationID")
    engineer.set_strategy(FrequencyEncodingStrategy())
    assert isinstance(engineer.strategy, FrequencyEncodingStrategy)


def test_feature_engineer_missing_column():
    engineer = FeatureEngineer(FrequencyEncodingStrategy(), "PULocationID")
    with pytest.raises(ValueError, match="absente"):
        engineer.run(pd.DataFrame({"autre_col": [1, 2, 3]}))


def test_target_encoding_no_target_raises():
    df = pd.DataFrame({"col": ["a", "b"], "y": [0.1, 0.2]})
    with pytest.raises(ValueError):
        TargetEncodingStrategy().fit(df, "col", target=None)


def test_is_fitted():
    df = pd.DataFrame({"PULocationID": [1, 2, 1]})
    strat = FrequencyEncodingStrategy()
    assert not strat.is_fitted()
    strat.fit(df, "PULocationID")
    assert strat.is_fitted()
