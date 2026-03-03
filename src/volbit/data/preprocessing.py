from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd  # noqa: TC002


def calculate_log_returns(df: pd.DataFrame, price_col: str = "close") -> pd.Series:
    """
    Calculate logarithmic returns from price series.
    r_t = ln(P_t / P_{t-1})

    Args:
        df: Input DataFrame containing prices.
        price_col: Name of the column containing prices.

    Returns:
        Series of log returns. First value will be NaN.

    Raises:
        KeyError: If price_col is not in df.
    """
    if price_col not in df.columns:
        raise KeyError(f"Column '{price_col}' not found in DataFrame.")

    prices = df[price_col]
    # np.log handles Series element-wise
    return cast("pd.Series", np.log(prices / prices.shift(1)))


def calculate_rolling_variance(returns: pd.Series, window: int) -> pd.Series:
    """
    Calculate rolling variance of returns.

    Args:
        returns: Series of returns.
        window: Window size for rolling calculation.

    Returns:
        Series of rolling variance. First `window-1` values will be NaN.
    """
    return returns.rolling(window=window).var()


def split_train_test(
    df: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time-series data into training and testing sets.
    The split is done sequentially to preserve temporal order.

    Args:
        df: Input DataFrame.
        test_size: Proportion of data to include in the test split (0.0 to 1.0).

    Returns:
        Tuple of (train_df, test_df).

    Raises:
        ValueError: If test_size is not between 0 and 1.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")

    n = len(df)
    split_idx = int(n * (1 - test_size))

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    return train_df, test_df
