# M1: Core Engine - Milestone Summary

## Overview
This milestone established the foundational components of the `volbit` Python package for stochastic volatility research. We successfully implemented the data ingestion pipeline, preprocessing utilities, Heston model calibration, vectorized simulation, and diagnostic tools.

## Implemented Features (Issues #1 - #6)

### 1. Project Scaffolding
- Established standard Python project structure with `src/` and `tests/`.
- Configured `pyproject.toml` for dependency management and build system.
- Set up `pytest`, `mypy`, and `ruff` for quality assurance.

### 2. Data Ingestion
- Implemented `DataLoader` in `volbit.data.loader`.
- Added support for loading BTC/USDT hourly CSV data.
- Enforced schema validation (timestamp, open, high, low, close, volume).

### 3. Preprocessing
- Developed utilities for calculating log-returns and realized variance.
- Implemented train/test splitting logic for time-series data.
- Handled pandas type safety issues to satisfy strict mypy checks.

### 4. Heston Calibration
- Created `HestonParameters` dataclass for model configuration.
- Built `HestonCalibrator` using `scipy.optimize.minimize`.
- Implemented Method of Moments objective function to estimate $\kappa, \theta, \xi, \rho, v_0$.

### 5. Vectorized Simulation
- Implemented Euler-Maruyama discretization for the Heston model.
- Used "Full Truncation" scheme to handle negative variance in intermediate steps.
- Fully vectorized using `numpy` for high-performance path generation.

### 6. Diagnostics
- Added Feller condition check (`2 \kappa \theta > \xi^2`) to ensure variance positivity.
- Implemented zero-variance hit counters to quantify numerical stability.
- Verified logic with rigorous unit tests including edge cases.

## Technical Challenges & Solutions

### Floating Point Precision
- **Issue:** Theoretical Feller ratios (e.g., exactly 1.0) failed equality checks due to floating-point representation.
- **Solution:** Used `pytest.approx` and epsilon-based comparisons in tests.

### Type Safety with Pandas
- **Issue:** `mypy` often inferred `Any` from pandas aggregations like `.var()` or `.mean()`.
- **Solution:** Used explicit `cast("float", ...)` to enforce strict typing compliance.

### Negative Variance in Simulation
- **Issue:** Standard Euler-Maruyama can produce negative variance values when discretized.
- **Solution:** Adopted the Full Truncation scheme ($max(v, 0)$) for drift/diffusion terms and clamped final outputs to be non-negative.

### Python Version Compatibility
- **Issue:** Initial setup targeted Python 3.11 features, but the environment was 3.9.6.
- **Solution:** Adjusted `pyproject.toml` requires-python and used `from __future__ import annotations` to support modern type hinting syntax.

## Verification
All features are verified by a comprehensive test suite.
To run tests:
```bash
pytest tests/
```
To run quality gates:
```bash
mypy src tests
ruff check src tests
```

## GitHub Workflow Status
- **Pull Requests:** All features merged via PR (e.g., PR #15 for diagnostics).
- **Issues:** All milestone issues (#1-#6) are closed.
- **Branching:** Feature branches used throughout; `main` protected.

## Conclusion
The core engine is now operationally ready for research tasks, capable of loading market data, calibrating parameters, and generating Monte Carlo simulations with stability guarantees.
