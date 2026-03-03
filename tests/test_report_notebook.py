from __future__ import annotations

import json
from pathlib import Path


def _read_notebook(path: Path) -> dict:
    return json.loads(path.read_text())


def _cell_text(cell: dict) -> str:
    source = cell.get("source", [])
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def test_final_report_notebook_has_required_sections() -> None:
    notebook_path = Path("m3_final_report.ipynb")
    assert notebook_path.exists()

    notebook = _read_notebook(notebook_path)
    markdown_text = "\n".join(
        _cell_text(cell)
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "markdown"
    ).lower()

    for heading in [
        "introduction",
        "sde theory",
        "methods",
        "calibration",
        "simulation",
        "stylized facts",
        "feller analysis",
        "smile",
        "discussion",
    ]:
        assert heading in markdown_text


def test_final_report_notebook_loads_saved_artifacts() -> None:
    notebook = _read_notebook(Path("m3_final_report.ipynb"))
    code_text = "\n".join(
        _cell_text(cell)
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ).lower()

    assert "metadata.json" in code_text
    assert "heston_parameters.json" in code_text
    assert "summary_metrics.json" in code_text
    assert "json.loads" in code_text
    assert "np.random" not in code_text
