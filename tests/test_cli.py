import os
import subprocess
import sys
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
