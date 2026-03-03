import numpy as np
import pandas as pd
import pytest

from volbit.data.preprocessing import (
    calculate_log_returns,
    calculate_rolling_variance,
    split_train_test,
)


@pytest.fixture
def sample_df():
    dates = pd.date_range(start="2023-01-01", periods=10, freq="H")
    prices = [100.0, 101.0, 100.5, 102.0, 103.0, 102.5, 104.0, 103.5, 105.0, 106.0]
    return pd.DataFrame({"close": prices}, index=dates)


def test_calculate_log_returns(sample_df):
    """Test log returns calculation."""
    returns = calculate_log_returns(sample_df, "close")

    # Check length
    assert len(returns) == len(sample_df)

    # Check first value is NaN or 0? Usually NaN.
    assert np.isnan(returns.iloc[0])

    # Check a value manually: ln(101/100) approx 0.00995
    expected = np.log(101.0 / 100.0)
    assert returns.iloc[1] == pytest.approx(expected)


def test_calculate_log_returns_missing_col(sample_df):
    with pytest.raises(KeyError):
        calculate_log_returns(sample_df, "non_existent")


def test_calculate_rolling_variance():
    """Test rolling variance calculation."""
    # Constant returns -> 0 variance
    returns = pd.Series([0.01, 0.01, 0.01, 0.01, 0.01])
    variance = calculate_rolling_variance(returns, window=3)

    assert len(variance) == len(returns)
    assert np.isnan(variance.iloc[0])
    assert np.isnan(variance.iloc[1])
    assert variance.iloc[2] == pytest.approx(0.0)


def test_split_train_test(sample_df):
    """Test train/test split."""
    train, test = split_train_test(sample_df, test_size=0.2)

    assert len(train) == 8
    assert len(test) == 2

    # Ensure ordered split (time series)
    assert train.index.max() < test.index.min()

    # Check no overlap
    assert len(pd.concat([train, test])) == len(sample_df)


def test_split_train_test_invalid_size(sample_df):
    with pytest.raises(ValueError):
        split_train_test(sample_df, test_size=1.5)
