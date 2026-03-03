# M2: Research Analysis Summary

Implemented issues: #7, #8, #9.

## Issue #7 — Stylized facts (fat tails, clustering)
- Added `src/volbit/analytics/stylized_facts.py` with tested utilities:
  - `excess_kurtosis`
  - `tail_event_ratio`
  - `volatility_clustering_acf`
  - `stylized_facts_summary`
- Added tests in `tests/test_stylized_facts.py`.
- PR merged: #19.

## Issue #8 — Leverage-effect coupling and regimes
- Added `src/volbit/analytics/regime_analysis.py` with tested utilities:
  - `rolling_return_vol_correlation`
  - `segment_regimes`
  - `regime_summary`
- Added tests in `tests/test_regime_analysis.py`.
- PR merged: #20.

## Issue #9 — Volatility smile from Monte Carlo option pricing
- Added `src/volbit/analytics/option_pricing.py` with tested utilities:
  - `mc_european_call_price`
  - `black_scholes_call_price`
  - `implied_volatility_call` (bisection inversion)
  - `build_smile_dataset`
- Added tests in `tests/test_option_pricing.py`.
- PR merged: #21.

## Shared updates
- Exported new analytics functions via `src/volbit/analytics/__init__.py`.
- Ran quality gates on each issue branch:
  - `uv run pytest tests/`
  - `uv run mypy src/`
  - `uv run ruff check src/ tests/ --fix`
  - `uv run ruff format src/ tests/`

## Problems faced
- `mypy src/` initially failed due missing stubs (`pandas-stubs`, `scipy-stubs`); installed stubs in the project environment.
- `ruff` was not initially installed in the environment; installed and reran lint/format gates.
- `gh pr merge --delete-branch` merged PRs successfully but local branch cleanup failed in this linked worktree (`main` already checked out elsewhere).
