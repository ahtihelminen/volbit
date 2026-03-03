import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from volbit import main


def test_package_import():
    """Test that the package can be imported."""
    import volbit

    assert volbit is not None


def test_cli_entrypoint_exists():
    """Test that the main entry point function exists."""
    assert hasattr(main, "cli")


def test_cli_help():
    """Test that the CLI can be invoked and shows help."""
    # Find the src directory relative to this test file
    # tests/test_cli.py -> parent -> root -> src
    src_path = Path(__file__).resolve().parents[1] / "src"

    # Create a clean environment with PYTHONPATH pointing to src
    env = os.environ.copy()
    env["PYTHONPATH"] = str(src_path)

    # This assumes we will use argparse or similar
    result = subprocess.run(
        [sys.executable, "-m", "volbit.main", "--help"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower() or "options:" in result.stdout.lower()


def test_cli_run_executes_experiment_from_config(tmp_path):
    src_path = Path(__file__).resolve().parents[1] / "src"
    market_csv = tmp_path / "market.csv"
    start = datetime(2023, 1, 1, 0, 0, 0)
    lines = ["timestamp,open,high,low,close,volume"]
    close = 105.0
    for i in range(130):
        ts = (start + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S")
        open_ = close
        high = close + 5.0
        low = close - 5.0
        close = close + 2.0
        volume = 10 + i
        lines.append(f"{ts},{open_},{high},{low},{close},{volume}")
    market_csv.write_text("\n".join(lines) + "\n")

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "data_path": str(market_csv),
                "output_dir": str(tmp_path / "artifacts"),
                "run_name": "cli_run",
                "seed": 7,
                "simulation": {"horizon_years": 0.2, "n_steps": 5, "n_sims": 3},
            }
        )
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(src_path)
    result = subprocess.run(
        [sys.executable, "-m", "volbit.main", "run", "--config", str(config_path)],
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0
    assert (tmp_path / "artifacts" / "cli_run" / "metadata.json").exists()
