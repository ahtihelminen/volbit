from __future__ import annotations

from math import erf, sqrt

import numpy as np
import pandas as pd


def _as_array(values: np.ndarray | list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("Input returns must be one-dimensional.")
    if array.size < 2:
        raise ValueError("At least two return observations are required.")
    return array


def excess_kurtosis(returns: np.ndarray | list[float]) -> float:
    values = _as_array(returns)
    centered = values - float(np.mean(values))
    variance = float(np.mean(centered**2))
    if variance == 0.0:
        raise ValueError("Variance is zero; kurtosis is undefined.")
    return float(np.mean(centered**4) / (variance**2) - 3.0)


def tail_event_ratio(
    returns: np.ndarray | list[float], z_threshold: float = 2.0
) -> float:
    values = _as_array(returns)
    if z_threshold <= 0:
        raise ValueError("z_threshold must be positive.")
    centered = values - float(np.mean(values))
    std = float(np.std(centered))
    if std == 0.0:
        raise ValueError("Standard deviation is zero; tail ratio is undefined.")
    z_scores = np.abs(centered) / std
    empirical_tail_prob = float(np.mean(z_scores > z_threshold))
    gaussian_tail_prob = 1.0 - erf(z_threshold / sqrt(2.0))
    if gaussian_tail_prob == 0.0:
        raise ValueError("Gaussian tail probability is zero for this threshold.")
    return empirical_tail_prob / gaussian_tail_prob


def volatility_clustering_acf(
    returns: np.ndarray | list[float],
    lags: int = 10,
    use_squared: bool = False,
) -> np.ndarray:
    values = _as_array(returns)
    if lags < 1 or lags >= values.size:
        raise ValueError("lags must be between 1 and len(returns)-1.")
    transformed = values**2 if use_squared else np.abs(values)
    transformed = transformed - np.mean(transformed)
    denom = float(np.dot(transformed, transformed))
    if denom == 0.0:
        raise ValueError("Constant transformed series; autocorrelation undefined.")
    acf = np.empty(lags + 1, dtype=float)
    acf[0] = 1.0
    for lag in range(1, lags + 1):
        numerator = float(np.dot(transformed[:-lag], transformed[lag:]))
        acf[lag] = numerator / denom
    return acf


def stylized_facts_summary(
    empirical_returns: np.ndarray | list[float],
    simulated_returns: np.ndarray | list[float],
    lags: int = 10,
) -> pd.DataFrame:
    empirical_acf = volatility_clustering_acf(empirical_returns, lags=lags)
    simulated_acf = volatility_clustering_acf(simulated_returns, lags=lags)
    rows = [
        (
            "excess_kurtosis",
            excess_kurtosis(empirical_returns),
            excess_kurtosis(simulated_returns),
        ),
        (
            "tail_event_ratio",
            tail_event_ratio(empirical_returns),
            tail_event_ratio(simulated_returns),
        ),
        (
            "volatility_clustering_acf_lag1",
            float(empirical_acf[1]),
            float(simulated_acf[1]),
        ),
    ]
    summary = pd.DataFrame(rows, columns=["metric", "empirical", "simulated"])
    summary["abs_diff"] = (summary["empirical"] - summary["simulated"]).abs()
    return summary
