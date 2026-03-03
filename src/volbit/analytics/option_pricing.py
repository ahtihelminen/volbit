from __future__ import annotations

from math import erf, exp, log, sqrt

import numpy as np
import pandas as pd


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def mc_european_call_price(
    terminal_prices: np.ndarray | list[float],
    strike: float,
    discount_factor: float = 1.0,
) -> float:
    prices = np.asarray(terminal_prices, dtype=float)
    if prices.ndim != 1:
        raise ValueError("terminal_prices must be one-dimensional.")
    if strike <= 0:
        raise ValueError("strike must be positive.")
    if discount_factor <= 0:
        raise ValueError("discount_factor must be positive.")
    payoff = np.maximum(prices - strike, 0.0)
    return float(discount_factor * np.mean(payoff))


def black_scholes_call_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
) -> float:
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive.")
    if maturity < 0:
        raise ValueError("maturity must be non-negative.")
    if volatility < 0:
        raise ValueError("volatility must be non-negative.")
    if maturity == 0:
        return max(spot - strike, 0.0)
    if volatility == 0:
        forward_intrinsic = spot - strike * exp(-rate * maturity)
        return max(forward_intrinsic, 0.0)

    vol_sqrt_t = volatility * sqrt(maturity)
    d1 = (log(spot / strike) + (rate + 0.5 * volatility**2) * maturity) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return spot * _normal_cdf(d1) - strike * exp(-rate * maturity) * _normal_cdf(d2)


def implied_volatility_call(
    call_price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> float:
    if maturity <= 0:
        raise ValueError("maturity must be positive.")
    lower_bound = max(0.0, spot - strike * exp(-rate * maturity))
    upper_bound = spot
    if call_price < lower_bound - tol or call_price > upper_bound + tol:
        raise ValueError("call_price violates no-arbitrage bounds.")
    if abs(call_price - lower_bound) <= tol:
        return 0.0

    low = 1e-8
    high = 5.0
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        mid_price = black_scholes_call_price(spot, strike, maturity, rate, mid)
        if abs(mid_price - call_price) <= tol:
            return mid
        if mid_price < call_price:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def build_smile_dataset(
    spot: float,
    strikes: np.ndarray | list[float],
    maturities: np.ndarray | list[float],
    rate: float,
    terminal_by_maturity: dict[float, np.ndarray],
) -> pd.DataFrame:
    strikes_array = np.asarray(strikes, dtype=float)
    maturities_array = np.asarray(maturities, dtype=float)
    rows: list[tuple[float, float, float, float]] = []

    for maturity in maturities_array:
        if maturity not in terminal_by_maturity:
            raise KeyError(f"Missing terminal prices for maturity {maturity}.")
        terminal_prices = terminal_by_maturity[float(maturity)]
        discount_factor = exp(-rate * float(maturity))
        for strike in strikes_array:
            call_price = mc_european_call_price(
                terminal_prices, float(strike), discount_factor=discount_factor
            )
            implied_vol = implied_volatility_call(
                call_price=call_price,
                spot=spot,
                strike=float(strike),
                maturity=float(maturity),
                rate=rate,
            )
            rows.append((float(maturity), float(strike), call_price, implied_vol))

    return pd.DataFrame(
        rows, columns=["maturity", "strike", "call_price", "implied_volatility"]
    )
