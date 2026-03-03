from __future__ import annotations

import numpy as np

from volbit.calibration.heston import HestonParameters


def simulate_heston(
    params: HestonParameters,
    T: float,
    n_steps: int,
    n_sims: int,
    seed: int | None = None,
    S0: float = 100.0,
    mu: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate Heston model paths using vectorized Euler-Maruyama scheme.
    Uses Full Truncation scheme for variance positivity.

    Args:
        params: Heston model parameters.
        T: Time horizon in years.
        n_steps: Number of time steps.
        n_sims: Number of simulation paths.
        seed: Random seed for reproducibility.
        S0: Initial asset price.
        mu: Drift of the asset price (under P or Q).

    Returns:
        Tuple of (S, v) arrays of shape (n_steps + 1, n_sims).
    """
    rng = np.random.default_rng(seed)

    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    # Initialize arrays
    S = np.zeros((n_steps + 1, n_sims))
    v = np.zeros((n_steps + 1, n_sims))

    S[0, :] = S0
    v[0, :] = params.v0

    # Pre-compute constants
    rho = params.rho
    sqrt_1_minus_rho2 = np.sqrt(1.0 - rho**2)
    kappa = params.kappa
    theta = params.theta
    xi = params.xi

    for t in range(n_steps):
        # Generate correlated Brownian motions
        Z1 = rng.standard_normal(n_sims)
        Z2 = rng.standard_normal(n_sims)

        dW1 = sqrt_dt * Z1
        dW2 = sqrt_dt * (rho * Z1 + sqrt_1_minus_rho2 * Z2)

        v_t = v[t, :]
        S_t = S[t, :]

        # Full Truncation Scheme for variance
        # drift and diffusion terms use max(v_t, 0)
        v_pos = np.maximum(v_t, 0.0)
        sqrt_v_pos = np.sqrt(v_pos)

        # Update Variance
        # dv = kappa * (theta - v_pos) * dt + xi * sqrt(v_pos) * dW2
        dv = kappa * (theta - v_pos) * dt + xi * sqrt_v_pos * dW2
        v[t + 1, :] = v_t + dv

        # Update Asset Price
        # dS = mu * S * dt + sqrt(v) * S * dW1
        # Use log-Euler for better stability?
        # S_{t+1} = S_t * exp((mu - 0.5 * v_pos) * dt + sqrt(v_pos) * dW1)
        # Using standard Euler as requested in scope "vectorized Euler-Maruyama"
        dS = mu * S_t * dt + sqrt_v_pos * S_t * dW1
        S[t + 1, :] = S_t + dS

    return S, np.maximum(v, 0.0)
