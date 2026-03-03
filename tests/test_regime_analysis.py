import pandas as pd
import pytest

from volbit.analytics.regime_analysis import (
    regime_summary,
    rolling_return_vol_correlation,
    segment_regimes,
)


def test_rolling_return_vol_correlation_computes_expected_windowed_values() -> None:
    returns = pd.Series([0.01, -0.02, 0.03, -0.04, 0.05, -0.06])
    vol_proxy = returns.abs()
    correlations = rolling_return_vol_correlation(returns, vol_proxy, window=3)
    assert correlations.isna().sum() == 2
    assert correlations.iloc[-1] < 0.0


def test_segment_regimes_assigns_bull_bear_and_high_vol_labels() -> None:
    returns = pd.Series([0.03, -0.02, 0.01, -0.04, 0.02, -0.01])
    vol_proxy = pd.Series([0.01, 0.02, 0.015, 0.05, 0.03, 0.01])
    labels = segment_regimes(returns, vol_proxy, vol_quantile=0.7)
    assert set(labels.unique()) <= {"bull", "bear", "high_vol"}
    assert "high_vol" in labels.values


def test_regime_summary_returns_grouped_statistics() -> None:
    regime_labels = pd.Series(["bull", "bull", "bear", "high_vol", "bear"])
    rho_estimates = pd.Series([-0.2, -0.1, -0.5, -0.7, -0.4])
    summary = regime_summary(regime_labels, rho_estimates)
    assert isinstance(summary, pd.DataFrame)
    assert list(summary.columns) == ["regime", "count", "rho_mean", "rho_std"]
    assert set(summary["regime"].tolist()) == {"bull", "bear", "high_vol"}
    bear_row = summary.loc[summary["regime"] == "bear"].iloc[0]
    assert bear_row["rho_mean"] == pytest.approx(-0.45)
