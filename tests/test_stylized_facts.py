import numpy as np
import pandas as pd
import pytest

from volbit.analytics.stylized_facts import (
    excess_kurtosis,
    stylized_facts_summary,
    tail_event_ratio,
    volatility_clustering_acf,
)


def test_excess_kurtosis_matches_manual_formula() -> None:
    returns = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=float)
    centered = returns - returns.mean()
    variance = np.mean(centered**2)
    expected = np.mean(centered**4) / (variance**2) - 3.0
    assert excess_kurtosis(returns) == pytest.approx(expected)


def test_tail_event_ratio_is_above_one_for_fat_tailed_data() -> None:
    rng = np.random.default_rng(0)
    returns = rng.laplace(loc=0.0, scale=1.0, size=20_000)
    ratio = tail_event_ratio(returns, z_threshold=2.0)
    assert ratio > 1.0


def test_volatility_clustering_acf_supports_absolute_and_squared_returns() -> None:
    returns = np.array([0.01, -0.03, 0.02, -0.04, 0.03, -0.05, 0.04], dtype=float)
    acf_abs = volatility_clustering_acf(returns, lags=3, use_squared=False)
    acf_sq = volatility_clustering_acf(returns, lags=3, use_squared=True)
    assert acf_abs.shape == (4,)
    assert acf_sq.shape == (4,)
    assert acf_abs[0] == pytest.approx(1.0)
    assert acf_sq[0] == pytest.approx(1.0)


def test_stylized_facts_summary_returns_reproducible_table() -> None:
    empirical = np.array([0.01, -0.03, 0.02, -0.04, 0.03, -0.05, 0.04], dtype=float)
    simulated = np.array([0.01, -0.02, 0.01, -0.03, 0.02, -0.03, 0.02], dtype=float)
    summary = stylized_facts_summary(empirical, simulated, lags=2)
    assert isinstance(summary, pd.DataFrame)
    assert list(summary.columns) == ["metric", "empirical", "simulated", "abs_diff"]
    assert "excess_kurtosis" in summary["metric"].tolist()
    assert "tail_event_ratio" in summary["metric"].tolist()
