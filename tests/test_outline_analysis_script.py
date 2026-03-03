from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from volbit.calibration.heston import HestonCalibrator, HestonParameters
from volbit.simulation.heston import simulate_heston

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "outline_analysis.py"
SPEC = importlib.util.spec_from_file_location("outline_analysis", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Unable to load scripts/outline_analysis.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

analyze_feller_and_zero_hits = MODULE.analyze_feller_and_zero_hits
build_smile_from_heston = MODULE.build_smile_from_heston
compare_stylized_facts = MODULE.compare_stylized_facts


def test_analyze_feller_and_zero_hits_returns_expected_keys() -> None:
    params = HestonParameters(kappa=1.5, theta=0.04, xi=0.8, rho=-0.4, v0=0.04)
    variance_paths = np.array(
        [
            [0.04, 0.05],
            [0.01, 0.00],
            [0.00, 0.03],
        ],
        dtype=float,
    )

    result = analyze_feller_and_zero_hits(params=params, variance_paths=variance_paths)

    assert set(result) == {"feller_ratio", "feller_pass", "zero_hits", "zero_hit_rate"}
    assert isinstance(result["feller_pass"], bool)
    assert result["zero_hits"] == 2


def test_compare_stylized_facts_returns_expected_metrics() -> None:
    empirical_returns = np.array([0.01, -0.03, 0.02, -0.04, 0.03, -0.05, 0.04])
    simulated_returns = np.array([0.01, -0.02, 0.01, -0.03, 0.02, -0.03, 0.02])

    summary = compare_stylized_facts(
        empirical_returns=empirical_returns,
        simulated_returns=simulated_returns,
        lags=2,
    )

    assert isinstance(summary, pd.DataFrame)
    assert list(summary.columns) == ["metric", "empirical", "simulated", "abs_diff"]
    assert set(summary["metric"]) == {
        "excess_kurtosis",
        "tail_event_ratio",
        "volatility_clustering_acf_lag1",
    }


def test_build_smile_from_heston_contains_all_requested_grid_points() -> None:
    params = HestonParameters(kappa=2.0, theta=0.04, xi=0.4, rho=-0.5, v0=0.04)

    dataset = build_smile_from_heston(
        params=params,
        spot=100.0,
        rate=0.0,
        maturities=np.array([0.25, 0.5]),
        strikes=np.array([90.0, 100.0, 110.0]),
        n_steps_per_year=120,
        n_sims=3_000,
        seed=123,
    )

    assert isinstance(dataset, pd.DataFrame)
    assert set(dataset.columns) == {
        "maturity",
        "strike",
        "call_price",
        "implied_volatility",
    }
    assert len(dataset) == 6
    assert np.all(np.isfinite(dataset["implied_volatility"]))


def test_prepare_market_frame_normalizes_ohlcv_columns() -> None:
    raw = pd.DataFrame(
        {
            "Timestamp": ["2024-01-01 00:00:00", "2024-01-01 00:01:00"],
            "Open": [100.0, 101.0],
            "High": [101.0, 102.0],
            "Low": [99.0, 100.0],
            "Close": [100.5, 101.5],
            "Volume": [10.0, 12.0],
        }
    )

    normalized = MODULE._prepare_market_frame(raw)

    assert list(normalized.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(normalized.index, pd.DatetimeIndex)
    assert len(normalized) == 2


def test_load_empirical_returns_supports_kaggle_download(monkeypatch, tmp_path) -> None:
    dataset_dir = tmp_path / "btc_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dataset_dir / "bitcoin_1min.csv"
    pd.DataFrame(
        {
            "Timestamp": [
                "2024-01-01 00:00:00",
                "2024-01-01 00:01:00",
                "2024-01-01 00:02:00",
            ],
            "Open": [100.0, 101.0, 103.0],
            "High": [101.0, 103.0, 104.0],
            "Low": [99.0, 100.0, 102.0],
            "Close": [100.5, 102.0, 103.5],
            "Volume": [10.0, 12.0, 13.0],
        }
    ).to_csv(csv_path, index=False)

    def _fake_download(_: str) -> Path:
        return dataset_dir

    monkeypatch.setattr(MODULE, "_download_kaggle_dataset", _fake_download)

    returns = MODULE._load_empirical_returns(
        data_path=None,
        use_kaggle_data=True,
        kaggle_dataset="aklimarimi/bitcoin-historical-data-1min-interval",
        resample_rule=None,
    )

    assert isinstance(returns, pd.Series)
    assert returns.name == "returns"
    assert len(returns) == 2


def test_prepare_market_frame_accepts_date_without_volume() -> None:
    raw = pd.DataFrame(
        {
            "Date": ["2024-01-01 00:00:00", "2024-01-01 00:01:00"],
            "Open": [100.0, 101.0],
            "High": [101.0, 102.0],
            "Low": [99.0, 100.0],
            "Close": [100.5, 101.5],
        }
    )

    normalized = MODULE._prepare_market_frame(raw)

    assert list(normalized.columns) == ["open", "high", "low", "close", "volume"]
    assert float(normalized["volume"].iloc[0]) == 0.0


def test_prepare_market_frame_coerces_string_prices_to_numeric() -> None:
    raw = pd.DataFrame(
        {
            "Date": ["2024-01-01 00:00:00", "2024-01-01 00:01:00"],
            "Open": ["100.0", "101.0"],
            "High": ["101.0", "102.0"],
            "Low": ["99.0", "100.0"],
            "Close": ["100.5", "101.5"],
        }
    )

    normalized = MODULE._prepare_market_frame(raw)
    assert normalized["close"].dtype.kind in {"f", "i"}


def test_annualized_variance_params_preserve_return_scale() -> None:
    rng = np.random.default_rng(12)
    empirical = pd.Series(rng.normal(0.0, 0.01, size=1_000), name="returns")
    base = HestonCalibrator().calibrate(empirical)
    annualized = MODULE._annualize_variance_params(base, n_steps_per_year=252)

    prices, _ = simulate_heston(
        params=annualized,
        T=len(empirical) / 252,
        n_steps=len(empirical),
        n_sims=3_000,
        seed=42,
    )
    simulated = MODULE._compute_simulated_returns(prices)

    ratio = float(np.std(simulated) / np.std(empirical.to_numpy()))
    assert 0.8 <= ratio <= 1.2


def test_tune_xi_from_stylized_facts_raises_xi_for_heavy_tails() -> None:
    rng = np.random.default_rng(123)
    empirical = pd.Series(rng.standard_t(df=3, size=800) * 0.01, name="returns")
    base = HestonCalibrator().calibrate(empirical)
    annualized = MODULE._annualize_variance_params(base, n_steps_per_year=252)

    tuned = MODULE._tune_xi_from_stylized_facts(
        annualized,
        empirical_returns=empirical.to_numpy(),
        n_steps_per_year=252,
        seed=9,
    )

    assert tuned.xi >= annualized.xi
