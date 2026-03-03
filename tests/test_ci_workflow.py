from __future__ import annotations

from pathlib import Path


def test_ci_workflow_runs_required_quality_gates() -> None:
    workflow_path = Path(".github/workflows/quality-gates.yml")
    assert workflow_path.exists()
    workflow_text = workflow_path.read_text()

    assert "on:" in workflow_text
    assert "pull_request" in workflow_text
    assert "push" in workflow_text
    assert "pytest tests/" in workflow_text
    assert "mypy src/" in workflow_text
    assert "ruff check src/ tests/" in workflow_text
    assert "ruff format --check src/ tests/" in workflow_text


def test_readme_includes_local_precheck_commands() -> None:
    readme = Path("README.md").read_text()
    assert "uv run pytest tests/" in readme
    assert "uv run mypy src/" in readme
    assert "uv run ruff check src/ tests/" in readme
    assert "uv run ruff format --check src/ tests/" in readme
