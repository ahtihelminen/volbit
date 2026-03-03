# VolBit

## Local quality gates

Run these before pushing:

- `uv run pytest tests/`
- `uv run mypy src/`
- `uv run ruff check src/ tests/`
- `uv run ruff format --check src/ tests/`

## Project-outline analysis script

Run the full Heston/Langevin analyses and visualizations described in
`project-outline.md`:

- `uv run python scripts/outline_analysis.py --output-dir outputs/outline_analysis`
- `uv run python scripts/outline_analysis.py --data-path path/to/btc_hourly.csv --output-dir outputs/outline_analysis`
- `uv run python scripts/outline_analysis.py --use-kaggle-data --kaggle-dataset aklimarimi/bitcoin-historical-data-1min-interval --resample-rule 1h --output-dir outputs/outline_analysis`

Generated outputs include:

- `stylized_facts_summary.csv`
- `feller_diagnostics.json`
- `regime_summary.csv`
- `volatility_smile.csv`
- `returns_fat_tails.png`
- `volatility_clustering_acf.png`
- `variance_paths.png`
- `leverage_effect_rolling_corr.png`
- `volatility_smile.png`
