import numpy as np
import pytest

from volbit.analytics.diagnostics import calculate_zero_hits, check_feller, feller_ratio
from volbit.calibration.heston import HestonParameters


def test_feller_condition_pass():
    # 2 * k * th > xi^2
    # 2 * 1.0 * 0.04 = 0.08
    # xi = 0.1 -> xi^2 = 0.01
    # 0.08 > 0.01 -> Pass
    params = HestonParameters(kappa=1.0, theta=0.04, xi=0.1, rho=-0.5, v0=0.04)
    assert check_feller(params) is True
    assert feller_ratio(params) == pytest.approx(8.0)


def test_feller_condition_fail():
    # 2 * 1.0 * 0.04 = 0.08
    # xi = 0.5 -> xi^2 = 0.25
    # 0.08 < 0.25 -> Fail
    params = HestonParameters(kappa=1.0, theta=0.04, xi=0.5, rho=-0.5, v0=0.04)
    assert check_feller(params) is False
    assert feller_ratio(params) == pytest.approx(0.08 / 0.25)


def test_calculate_zero_hits():
    # Create synthetic variance path (n_steps x n_sims)
    v_paths = np.array([[0.04, 0.04], [0.00, 0.02], [0.01, 0.00], [0.00, 0.00]])
    # path 1: 0.04 -> 0.00 -> 0.01 -> 0.00 (2 hits)
    # path 2: 0.04 -> 0.02 -> 0.00 -> 0.00 (2 hits)
    # Total hits = 4

    hits = calculate_zero_hits(v_paths, tolerance=1e-6)
    assert hits == 4

    # Test strict zero vs tolerance
    v_paths_small = np.array([[1e-7]])
    assert calculate_zero_hits(v_paths_small, tolerance=1e-6) == 1
    assert calculate_zero_hits(v_paths_small, tolerance=1e-8) == 0
