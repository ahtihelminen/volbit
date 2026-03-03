import numpy as np
import pandas as pd
import pytest

from volbit.calibration.heston import HestonCalibrator, HestonParameters


def test_calibration_initialization():
    calibrator = HestonCalibrator()
    assert calibrator is not None


def test_calibration_result_structure():
    # Test that we can instantiate the result class
    params = HestonParameters(kappa=1.0, theta=0.04, xi=0.3, rho=-0.5, v0=0.04)
    assert params.kappa == 1.0
    assert params.theta == 0.04


def test_calibration_run():
    """Test calibration on synthetic data (constant variance)."""
    # Generate synthetic returns with constant variance (kappa=high, xi=0)
    rng = np.random.default_rng(42)
    # 1000 days of returns, volatility 20%
    returns = pd.Series(rng.normal(0, 0.2 / np.sqrt(252), 1000))

    calibrator = HestonCalibrator()
    result = calibrator.calibrate(returns)

    assert isinstance(result, HestonParameters)
    # Check bounds
    assert result.kappa > 0
    assert result.theta > 0
    assert result.xi >= 0
    assert -1 <= result.rho <= 1
    assert result.v0 > 0


def test_calibration_short_data():
    """Test that calibration handles short data gracefully."""
    returns = pd.Series([0.01, -0.01, 0.01])
    calibrator = HestonCalibrator()
    # Should probably raise an error or warning for too few data points
    with pytest.raises(ValueError):
        calibrator.calibrate(returns)
