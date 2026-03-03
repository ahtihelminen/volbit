from __future__ import annotations

import pandas as pd


def rolling_return_vol_correlation(
    returns: pd.Series,
    vol_proxy: pd.Series,
    window: int = 30,
) -> pd.Series:
    if window < 2:
        raise ValueError("window must be at least 2.")
    if len(returns) != len(vol_proxy):
        raise ValueError("returns and vol_proxy must have the same length.")
    return returns.rolling(window=window).corr(vol_proxy)


def segment_regimes(
    returns: pd.Series,
    vol_proxy: pd.Series,
    vol_quantile: float = 0.8,
) -> pd.Series:
    if not 0.0 < vol_quantile < 1.0:
        raise ValueError("vol_quantile must be between 0 and 1.")
    if len(returns) != len(vol_proxy):
        raise ValueError("returns and vol_proxy must have the same length.")

    high_vol_threshold = float(vol_proxy.quantile(vol_quantile))
    labels = pd.Series(index=returns.index, dtype="object")
    labels.loc[vol_proxy >= high_vol_threshold] = "high_vol"
    labels.loc[(vol_proxy < high_vol_threshold) & (returns >= 0)] = "bull"
    labels.loc[(vol_proxy < high_vol_threshold) & (returns < 0)] = "bear"
    return labels


def regime_summary(regime_labels: pd.Series, rho_estimates: pd.Series) -> pd.DataFrame:
    if len(regime_labels) != len(rho_estimates):
        raise ValueError("regime_labels and rho_estimates must have the same length.")

    df = pd.DataFrame({"regime": regime_labels, "rho": rho_estimates}).dropna()
    grouped = (
        df.groupby("regime")["rho"]
        .agg(count="count", rho_mean="mean", rho_std="std")
        .reset_index()
    )
    return grouped.sort_values("regime").reset_index(drop=True)
