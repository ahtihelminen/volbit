import subprocess
import sys

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
    # This assumes we will use argparse or similar
    result = subprocess.run(
        [sys.executable, "-m", "volbit.main", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower() or "options:" in result.stdout.lower()
