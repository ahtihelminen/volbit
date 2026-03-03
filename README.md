# VolBit

## Local quality gates

Run these before pushing:

- `uv run pytest tests/`
- `uv run mypy src/`
- `uv run ruff check src/ tests/`
- `uv run ruff format --check src/ tests/`
