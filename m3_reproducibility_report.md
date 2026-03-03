# M3 Reproducibility & Report Summary

## Implemented milestone issues

### Issue #10 - Reproducible experiment runner and artifact management
- Added TDD coverage for experiment config parsing and orchestration artifact outputs.
- Added CLI integration coverage for `python -m volbit.main run --config ...`.
- Updated test fixture generation to provide sufficient calibration history while keeping deterministic input data.

### Issue #11 - Final Jupyter notebook report
- Added `m3_final_report.ipynb` with required sections:
  - Introduction
  - SDE Theory
  - Methods
  - Calibration
  - Simulation
  - Stylized Facts
  - Feller Analysis
  - Smile
  - Discussion
- Notebook code cells load persisted artifacts (`metadata.json`, `heston_parameters.json`, `summary_metrics.json`) for deterministic analysis.
- Added tests validating section coverage and artifact-loading expectations.

### Issue #12 - CI quality gates
- Added GitHub Actions workflow: `.github/workflows/quality-gates.yml`.
- Workflow runs on pull requests and pushes to `main`.
- Included mandatory checks:
  - `uv run pytest tests/`
  - `uv run mypy src/`
  - `uv run ruff check src/ tests/`
  - `uv run ruff format --check src/ tests/`
- Updated `README.md` with local pre-check commands for contributors.

## Problems encountered and resolutions

- **Local branch setup conflict for `main`**: could not switch directly to `main` in this worktree; used `origin/main` when creating feature branches.
- **Baseline type checking dependency gap**: `mypy` initially failed due missing third-party stubs; installed `pandas-stubs` and `scipy-stubs` locally to satisfy quality-gate execution.
- **Calibration data threshold in tests**: calibration requires at least 100 return observations; replaced short literal CSV fixtures with deterministic generated series of sufficient length.
- **Lint tool missing in environment**: `ruff` not available initially; installed it locally, then fixed formatting/lint issues.
