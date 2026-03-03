from __future__ import annotations

import argparse
import sys

from volbit.experiments.runner import load_experiment_config, run_experiment


def cli(args: list[str] | None = None) -> int:
    """
    Main CLI entry point for volbit.

    Args:
        args: Command line arguments. If None, uses sys.argv[1:].

    Returns:
        Exit code (0 for success).
    """
    parser = argparse.ArgumentParser(
        description="VolBit: Stochastic Volatility Dynamics in Bitcoin"
    )

    # We can add subparsers here later for different modules
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Example command: run-experiment
    parser_run = subparsers.add_parser("run", help="Run an experiment")
    parser_run.add_argument("--config", required=True, help="Path to configuration file")

    parsed_args = parser.parse_args(args) if args is not None else parser.parse_args()

    if parsed_args.command == "run":
        config = load_experiment_config(parsed_args.config)
        run_dir = run_experiment(config)
        print(f"Experiment complete. Artifacts: {run_dir}")
    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    sys.exit(cli())
