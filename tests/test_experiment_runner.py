import json
from datetime import datetime, timedelta

from volbit.experiments.runner import (
    ExperimentConfig,
    load_experiment_config,
    run_experiment,
)


def _write_sample_market_csv(path):
    start = datetime(2023, 1, 1, 0, 0, 0)
    lines = ["timestamp,open,high,low,close,volume"]
    close = 16520.0
    for i in range(130):
        ts = (start + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S")
        open_ = close
        high = close + 30.0
        low = close - 30.0
        close = close + 20.0
        volume = 100.0 + i
        lines.append(f"{ts},{open_},{high},{low},{close},{volume}")
    path.write_text("\n".join(lines) + "\n")


def test_load_experiment_config_parses_required_fields(tmp_path):
    market_csv = tmp_path / "market.csv"
    _write_sample_market_csv(market_csv)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "data_path": str(market_csv),
                "output_dir": str(tmp_path / "artifacts"),
                "run_name": "smoke",
                "seed": 123,
                "simulation": {"horizon_years": 0.25, "n_steps": 8, "n_sims": 5},
            }
        )
    )

    config = load_experiment_config(config_path)
    assert isinstance(config, ExperimentConfig)
    assert config.seed == 123
    assert config.n_steps == 8
    assert config.n_sims == 5


def test_run_experiment_writes_standardized_artifacts_and_metadata(tmp_path):
    market_csv = tmp_path / "market.csv"
    _write_sample_market_csv(market_csv)
    config = ExperimentConfig(
        data_path=market_csv,
        output_dir=tmp_path / "artifacts",
        run_name="test_run",
        seed=42,
        horizon_years=0.25,
        n_steps=6,
        n_sims=4,
    )

    run_dir = run_experiment(config)

    params_path = run_dir / "params" / "heston_parameters.json"
    metrics_path = run_dir / "metrics" / "summary_metrics.json"
    metadata_path = run_dir / "metadata.json"
    figure_placeholder_path = run_dir / "figures" / "README.txt"

    assert params_path.exists()
    assert metrics_path.exists()
    assert metadata_path.exists()
    assert figure_placeholder_path.exists()

    metadata = json.loads(metadata_path.read_text())
    assert metadata["seed"] == 42
    assert metadata["run_name"] == "test_run"
    assert metadata["artifacts"]["params"] == "params/heston_parameters.json"
