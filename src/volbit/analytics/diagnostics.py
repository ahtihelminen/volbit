from __future__ import annotations

import numpy as np

from volbit.calibration.heston import HestonParameters  # noqa: TC001


def feller_ratio(params: HestonParameters) -> float:
    """
    Calculate the Feller ratio: (2 * kappa * theta) / xi^2.

    If > 1, the Feller condition is satisfied (variance > 0 almost surely).
    If <= 1, variance can reach zero.

    Args:
        params: Heston model parameters.

    Returns:
        The Feller ratio.
    """
    numerator = 2.0 * params.kappa * params.theta
    denominator = params.xi**2

    if denominator == 0:
        return float("inf")

    return numerator / denominator


def check_feller(params: HestonParameters) -> bool:
    """
    Check if the Feller condition is satisfied.
    2 * kappa * theta > xi^2

    Args:
        params: Heston model parameters.

    Returns:
        True if satisfied, False otherwise.
    """
    return feller_ratio(params) > 1.0


def calculate_zero_hits(paths: np.ndarray, tolerance: float = 1e-6) -> int:
    """
    Count the total number of times variance paths hit zero (or fall below tolerance).

    Args:
        paths: Variance paths array of shape (n_steps, n_sims) or similar.
        tolerance: Threshold below which variance is considered zero.

    Returns:
        Total count of zero hits across all paths and steps.
    """
    hits = np.sum(paths <= tolerance)
    return int(hits)
