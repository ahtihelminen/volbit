import numpy as np

from volbit.calibration.heston import HestonParameters
from volbit.simulation.heston import simulate_heston


def test_simulation_shape():
    params = HestonParameters(kappa=1.0, theta=0.04, xi=0.3, rho=-0.5, v0=0.04)
    n_sims = 100
    n_steps = 50
    T = 1.0

    S, v = simulate_heston(params, T, n_steps, n_sims)

    assert S.shape == (n_steps + 1, n_sims)
    assert v.shape == (n_steps + 1, n_sims)

    # Check initial values
    assert np.allclose(v[0, :], params.v0)


def test_simulation_reproducibility():
    params = HestonParameters(kappa=1.0, theta=0.04, xi=0.3, rho=-0.5, v0=0.04)
    S1, v1 = simulate_heston(params, 1.0, 10, 100, seed=42)
    S2, v2 = simulate_heston(params, 1.0, 10, 100, seed=42)

    assert np.allclose(S1, S2)
    assert np.allclose(v1, v2)

    S3, v3 = simulate_heston(params, 1.0, 10, 100, seed=43)
    assert not np.allclose(S1, S3)


def test_variance_positivity():
    # High vol of vol to trigger potential negative variance
    params = HestonParameters(kappa=0.5, theta=0.04, xi=1.0, rho=-0.5, v0=0.04)
    S, v = simulate_heston(params, 1.0, 100, 100, seed=42)

    assert np.all(v >= 0)


def test_correlation_structure():
    # We can't easily check correlation of paths directly without many sims,
    # but we can check if the function accepts rho.
    # A simple check: if rho=1, dW1 ~ dW2.
    # This requires inspecting internal increments or checking path correlation.
    # For now, just ensure it runs.
    params = HestonParameters(kappa=1.0, theta=0.04, xi=0.1, rho=0.9, v0=0.04)
    S, v = simulate_heston(params, 1.0, 100, 1000, seed=42)

    # Log returns
    log_ret = np.diff(np.log(S), axis=0)
    # Variance changes
    vol_changes = np.diff(v, axis=0)

    # Correlation between returns and vol changes should be roughly rho
    # (complicated by drift and other terms, but sign should match)
    corrs = []
    for i in range(1000):
        corrs.append(np.corrcoef(log_ret[:, i], vol_changes[:, i])[0, 1])

    mean_corr = np.mean(corrs)
    # It won't be exactly rho due to discretization and drift, but should be positive
    assert mean_corr > 0.0
