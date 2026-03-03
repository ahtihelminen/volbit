from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from volbit.analytics.diagnostics import calculate_zero_hits, feller_ratio
from volbit.calibration.heston import HestonCalibrator
from volbit.data.loader import DataLoader
from volbit.data.preprocessing import calculate_log_returns
from volbit.simulation.heston import simulate_heston


@dataclass(frozen=True)
class ExperimentConfig:
    data_path: Path
    output_dir: Path
    run_name: str
    seed: int
    horizon_years: float
    n_steps: int
    n_sims: int


def load_experiment_config(config_path: Path | str) -> ExperimentConfig:
    config = json.loads(Path(config_path).read_text())
    simulation = config["simulation"]
    return ExperimentConfig(
        data_path=Path(config["data_path"]),
        output_dir=Path(config["output_dir"]),
        run_name=str(config["run_name"]),
        seed=int(config["seed"]),
        horizon_years=float(simulation["horizon_years"]),
        n_steps=int(simulation["n_steps"]),
        n_sims=int(simulation["n_sims"]),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def run_experiment(config: ExperimentConfig) -> Path:
    run_dir = config.output_dir / config.run_name
    params_dir = run_dir / "params"
    metrics_dir = run_dir / "metrics"
    figures_dir = run_dir / "figures"
    params_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    market = DataLoader().load_csv(config.data_path)
    returns = calculate_log_returns(market).dropna()
    params = HestonCalibrator().calibrate(returns)
    _, variance_paths = simulate_heston(
        params=params,
        T=config.horizon_years,
        n_steps=config.n_steps,
        n_sims=config.n_sims,
        seed=config.seed,
        S0=float(market["close"].iloc[-1]),
    )

    params_path = params_dir / "heston_parameters.json"
    metrics_path = metrics_dir / "summary_metrics.json"
    metadata_path = run_dir / "metadata.json"
    figure_note_path = figures_dir / "README.txt"

    _write_json(params_path, asdict(params))
    _write_json(
        metrics_path,
        {
            "feller_ratio": feller_ratio(params),
            "zero_hits": calculate_zero_hits(variance_paths),
            "n_returns": int(returns.shape[0]),
        },
    )
    _write_json(
        metadata_path,
        {
            "run_name": config.run_name,
            "seed": config.seed,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "data_path": str(config.data_path),
            "artifacts": {
                "params": "params/heston_parameters.json",
                "metrics": "metrics/summary_metrics.json",
                "figures": "figures/",
            },
        },
    )
    figure_note_path.write_text(
        "Store deterministic figures generated from saved parameters and seed.\n"
    )

    return run_dir
