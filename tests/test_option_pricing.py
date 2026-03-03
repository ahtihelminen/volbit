import numpy as np
import pandas as pd
import pytest

from volbit.analytics.option_pricing import (
    black_scholes_call_price,
    build_smile_dataset,
    implied_volatility_call,
    mc_european_call_price,
)


def test_mc_european_call_price_matches_manual_expectation() -> None:
    terminal_prices = np.array([90.0, 100.0, 120.0, 130.0])
    strike = 100.0
    price = mc_european_call_price(terminal_prices, strike)
    expected = np.mean([0.0, 0.0, 20.0, 30.0])
    assert price == pytest.approx(expected)


def test_black_scholes_call_price_known_value() -> None:
    price = black_scholes_call_price(
        spot=100.0, strike=100.0, maturity=1.0, rate=0.0, volatility=0.2
    )
    assert price == pytest.approx(7.9655674554)


def test_implied_volatility_inverts_black_scholes_price() -> None:
    target_vol = 0.35
    call_price = black_scholes_call_price(
        spot=100.0, strike=95.0, maturity=0.5, rate=0.01, volatility=target_vol
    )
    implied = implied_volatility_call(
        call_price=call_price, spot=100.0, strike=95.0, maturity=0.5, rate=0.01
    )
    assert implied == pytest.approx(target_vol, abs=1e-6)


def test_implied_volatility_rejects_prices_outside_bounds() -> None:
    with pytest.raises(ValueError):
        implied_volatility_call(
            call_price=200.0, spot=100.0, strike=100.0, maturity=1.0, rate=0.0
        )


def test_build_smile_dataset_outputs_expected_columns_and_rows() -> None:
    terminal_by_maturity = {
        0.5: np.array([90.0, 95.0, 100.0, 110.0, 130.0]),
        1.0: np.array([85.0, 90.0, 100.0, 120.0, 140.0]),
    }
    dataset = build_smile_dataset(
        spot=100.0,
        strikes=np.array([90.0, 100.0, 110.0]),
        maturities=np.array([0.5, 1.0]),
        rate=0.0,
        terminal_by_maturity=terminal_by_maturity,
    )
    assert isinstance(dataset, pd.DataFrame)
    assert list(dataset.columns) == [
        "maturity",
        "strike",
        "call_price",
        "implied_volatility",
    ]
    assert len(dataset) == 6
    assert np.all(np.isfinite(dataset["implied_volatility"]))
