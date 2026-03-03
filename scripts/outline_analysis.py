from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from math import exp
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from volbit.analytics.diagnostics import (
        calculate_zero_hits,
        check_feller,
        feller_ratio,
    )
    from volbit.analytics.option_pricing import (
        implied_volatility_call,
        mc_european_call_price,
    )
    from volbit.analytics.regime_analysis import (
        regime_summary,
        rolling_return_vol_correlation,
        segment_regimes,
    )
    from volbit.analytics.stylized_facts import (
        stylized_facts_summary,
        volatility_clustering_acf,
    )
    from volbit.calibration.heston import HestonCalibrator, HestonParameters
    from volbit.data.loader import DataLoader
    from volbit.data.preprocessing import calculate_log_returns
    from volbit.simulation.heston import simulate_heston
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from volbit.analytics.diagnostics import (
        calculate_zero_hits,
        check_feller,
        feller_ratio,
    )
    from volbit.analytics.option_pricing import (
        implied_volatility_call,
        mc_european_call_price,
    )
    from volbit.analytics.regime_analysis import (
        regime_summary,
        rolling_return_vol_correlation,
        segment_regimes,
    )
    from volbit.analytics.stylized_facts import (
        stylized_facts_summary,
        volatility_clustering_acf,
    )
    from volbit.calibration.heston import HestonCalibrator, HestonParameters
    from volbit.data.loader import DataLoader
    from volbit.data.preprocessing import calculate_log_returns
    from volbit.simulation.heston import simulate_heston


@dataclass
class OutlineAnalysisConfig:
    output_dir: Path
    data_path: Path | None = None
    use_kaggle_data: bool = False
    kaggle_dataset: str = "aklimarimi/bitcoin-historical-data-1min-interval"
    resample_rule: str | None = "1h"
    n_sims: int = 100_000
    n_steps_per_year: int = 365 * 24
    smile_sims: int = 40_000
    seed: int = 7
    spot: float = 100.0
    rate: float = 0.0
    lags: int = 40


def analyze_feller_and_zero_hits(
    params: HestonParameters,
    variance_paths: np.ndarray,
    tolerance: float = 1e-10,
) -> dict[str, float | bool | int]:
    hits = calculate_zero_hits(variance_paths, tolerance=tolerance)
    total_points = int(variance_paths.size)
    ratio = float(feller_ratio(params))
    passes = bool(check_feller(params))
    return {
        "feller_ratio": ratio,
        "feller_pass": passes,
        "zero_hits": int(hits),
        "zero_hit_rate": float(hits / total_points if total_points else 0.0),
    }


def compare_stylized_facts(
    empirical_returns: np.ndarray,
    simulated_returns: np.ndarray,
    lags: int = 40,
) -> pd.DataFrame:
    return stylized_facts_summary(
        empirical_returns=empirical_returns,
        simulated_returns=simulated_returns,
        lags=lags,
    )


def build_smile_from_heston(
    params: HestonParameters,
    spot: float,
    rate: float,
    maturities: np.ndarray,
    strikes: np.ndarray,
    n_steps_per_year: int,
    n_sims: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[tuple[float, float, float, float]] = []
    for maturity in maturities:
        maturity_float = float(maturity)
        n_steps = max(int(np.ceil(maturity_float * n_steps_per_year)), 1)
        prices, _ = simulate_heston(
            params=params,
            T=maturity_float,
            n_steps=n_steps,
            n_sims=n_sims,
            seed=seed + n_steps,
            S0=spot,
            mu=rate,
        )
        terminal_prices = prices[-1, :]
        discount_factor = exp(-rate * maturity_float)

        for strike in strikes:
            strike_float = float(strike)
            call_price = mc_european_call_price(
                terminal_prices=terminal_prices,
                strike=strike_float,
                discount_factor=discount_factor,
            )
            lower_bound = max(0.0, spot - strike_float * discount_factor)
            upper_bound = spot
            clipped_price = min(max(call_price, lower_bound), upper_bound)
            implied_vol = implied_volatility_call(
                call_price=clipped_price,
                spot=spot,
                strike=strike_float,
                maturity=maturity_float,
                rate=rate,
            )
            rows.append((maturity_float, strike_float, clipped_price, implied_vol))

    return pd.DataFrame(
        rows,
        columns=["maturity", "strike", "call_price", "implied_volatility"],
    )


def _download_kaggle_dataset(dataset: str) -> Path:
    import kagglehub

    return Path(kagglehub.dataset_download(dataset))


def _prepare_market_frame(raw_frame: pd.DataFrame) -> pd.DataFrame:
    frame = raw_frame.copy()
    frame.columns = [str(col).strip().lower() for col in frame.columns]

    if "timestamp" not in frame.columns:
        if "date" in frame.columns:
            frame = frame.rename(columns={"date": "timestamp"})
        elif "time" in frame.columns:
            frame = frame.rename(columns={"time": "timestamp"})
        else:
            raise ValueError("Could not find timestamp column in market dataset.")

    if "volume" not in frame.columns:
        frame["volume"] = 0.0

    required = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required market columns: {missing}")

    timestamp_series = frame["timestamp"]
    if pd.api.types.is_numeric_dtype(timestamp_series):
        timestamp = pd.to_datetime(timestamp_series, unit="s", utc=False)
    else:
        timestamp = pd.to_datetime(timestamp_series, utc=False)

    prepared = frame[["open", "high", "low", "close", "volume"]].copy()
    for column in ["open", "high", "low", "close", "volume"]:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    prepared.index = timestamp
    prepared = prepared[~prepared.index.isna()].sort_index()
    prepared = prepared.dropna(subset=["open", "high", "low", "close"])
    return prepared[~prepared.index.duplicated(keep="first")]


def _find_market_csv(dataset_dir: Path) -> Path:
    csv_files = sorted(dataset_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in dataset directory: {dataset_dir}"
        )

    for csv_file in csv_files:
        try:
            sample = pd.read_csv(csv_file, nrows=20)
            _prepare_market_frame(sample)
            return csv_file
        except Exception:
            continue

    raise FileNotFoundError(
        "No CSV in dataset matched required market schema "
        "(timestamp/date/time + open/high/low/close/volume)."
    )


def _load_empirical_returns(
    data_path: Path | None,
    use_kaggle_data: bool = False,
    kaggle_dataset: str = "aklimarimi/bitcoin-historical-data-1min-interval",
    resample_rule: str | None = "1h",
) -> pd.Series:
    if use_kaggle_data:
        dataset_dir = _download_kaggle_dataset(kaggle_dataset)
        csv_path = _find_market_csv(dataset_dir)
        raw_frame = pd.read_csv(csv_path)
        frame = _prepare_market_frame(raw_frame)
        if resample_rule:
            close_price = frame["close"].resample(resample_rule).last().dropna()
            frame = pd.DataFrame({"close": close_price})
        returns = calculate_log_returns(frame).dropna()
        return returns.rename("returns")

    if data_path is None:
        rng = np.random.default_rng(123)
        synthetic = rng.standard_t(df=5, size=2_500) * 0.015
        return pd.Series(synthetic, name="returns")

    loader = DataLoader()
    frame = loader.load_csv(data_path)
    if resample_rule:
        close_price = frame["close"].resample(resample_rule).last().dropna()
        frame = pd.DataFrame({"close": close_price})
    returns = calculate_log_returns(frame).dropna()
    return returns.rename("returns")


def _plot_return_distribution(
    empirical_returns: np.ndarray,
    simulated_returns: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(empirical_returns, bins=80, density=True, alpha=0.45, label="Empirical")
    ax.hist(simulated_returns, bins=80, density=True, alpha=0.45, label="Heston")

    mean_emp = float(np.mean(empirical_returns))
    std_emp = float(np.std(empirical_returns))
    x = np.linspace(mean_emp - 5 * std_emp, mean_emp + 5 * std_emp, 400)
    gaussian_pdf = np.exp(-0.5 * ((x - mean_emp) / std_emp) ** 2) / (
        std_emp * np.sqrt(2 * np.pi)
    )
    ax.plot(x, gaussian_pdf, "k--", linewidth=2.0, label="Gaussian benchmark")
    ax.set_title("Return Distribution: Fat Tails vs Gaussian")
    ax.set_xlabel("Log Return")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_volatility_clustering(
    empirical_returns: np.ndarray,
    simulated_returns: np.ndarray,
    lags: int,
    output_path: Path,
) -> None:
    empirical_acf = volatility_clustering_acf(empirical_returns, lags=lags)
    simulated_acf = volatility_clustering_acf(simulated_returns, lags=lags)
    lag_index = np.arange(lags + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(lag_index, empirical_acf, label="Empirical |r| ACF", linewidth=2)
    ax.plot(lag_index, simulated_acf, label="Heston |r| ACF", linewidth=2)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Volatility Clustering (Absolute Return ACF)")
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_variance_paths(variance_paths: np.ndarray, output_path: Path) -> None:
    n_paths = min(30, variance_paths.shape[1])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(variance_paths[:, :n_paths], alpha=0.65)
    ax.set_title("Sample Variance Paths (Heston Langevin Dynamics)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Variance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_leverage_regime(
    rolling_corr: pd.Series,
    output_path: Path,
) -> None:
    clean_corr = rolling_corr.dropna()
    x_values = clean_corr.index.to_numpy()
    y_values = clean_corr.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_values, y_values, linewidth=1.8)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Rolling Return-Volatility Correlation (Leverage Effect)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Rolling Corr")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_volatility_smile(smile_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for maturity, subset in smile_df.groupby("maturity"):
        ordered = subset.sort_values("strike")
        ax.plot(
            ordered["strike"],
            ordered["implied_volatility"],
            marker="o",
            linewidth=1.5,
            label=f"T={maturity:.2f}",
        )
    ax.set_title("Implied Volatility Smile from Heston Monte Carlo")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied Volatility")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_outline_analysis(config: OutlineAnalysisConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    empirical_returns_series = _load_empirical_returns(
        config.data_path,
        use_kaggle_data=config.use_kaggle_data,
        kaggle_dataset=config.kaggle_dataset,
        resample_rule=config.resample_rule,
    )
    empirical_returns = empirical_returns_series.to_numpy(dtype=float)

    calibrator = HestonCalibrator()
    params = calibrator.calibrate(empirical_returns_series)

    n_steps = max(len(empirical_returns), 2)
    horizon_years = n_steps / config.n_steps_per_year

    prices, variance_paths = simulate_heston(
        params=params,
        T=horizon_years,
        n_steps=n_steps,
        n_sims=config.n_sims,
        seed=config.seed,
        S0=config.spot,
        mu=config.rate,
    )

    simulated_returns = np.diff(np.log(prices[:, 0]))
    feller_stats = analyze_feller_and_zero_hits(
        params=params,
        variance_paths=variance_paths,
    )
    stylized_summary = compare_stylized_facts(
        empirical_returns=empirical_returns,
        simulated_returns=simulated_returns,
        lags=min(config.lags, len(empirical_returns) - 1, len(simulated_returns) - 1),
    )

    rolling_window = min(120, len(empirical_returns_series) // 4)
    rolling_window = max(rolling_window, 20)
    vol_proxy = empirical_returns_series.abs()
    rolling_corr = rolling_return_vol_correlation(
        returns=empirical_returns_series,
        vol_proxy=vol_proxy,
        window=rolling_window,
    )
    regimes = segment_regimes(
        returns=empirical_returns_series,
        vol_proxy=vol_proxy,
        vol_quantile=0.8,
    )
    regime_stats = regime_summary(regime_labels=regimes, rho_estimates=rolling_corr)

    strikes = config.spot * np.array([0.8, 0.9, 1.0, 1.1, 1.2], dtype=float)
    maturities = np.array([0.25, 0.5, 1.0], dtype=float)
    smile_df = build_smile_from_heston(
        params=params,
        spot=config.spot,
        rate=config.rate,
        maturities=maturities,
        strikes=strikes,
        n_steps_per_year=config.n_steps_per_year,
        n_sims=config.smile_sims,
        seed=config.seed + 100,
    )

    stylized_summary.to_csv(
        config.output_dir / "stylized_facts_summary.csv",
        index=False,
    )
    regime_stats.to_csv(config.output_dir / "regime_summary.csv", index=False)
    smile_df.to_csv(config.output_dir / "volatility_smile.csv", index=False)

    with (config.output_dir / "feller_diagnostics.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(feller_stats, file, indent=2)

    with (config.output_dir / "calibrated_parameters.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(asdict(params), file, indent=2)

    _plot_return_distribution(
        empirical_returns=empirical_returns,
        simulated_returns=simulated_returns,
        output_path=config.output_dir / "returns_fat_tails.png",
    )
    _plot_volatility_clustering(
        empirical_returns=empirical_returns,
        simulated_returns=simulated_returns,
        lags=min(config.lags, len(empirical_returns) - 1, len(simulated_returns) - 1),
        output_path=config.output_dir / "volatility_clustering_acf.png",
    )
    _plot_variance_paths(
        variance_paths=variance_paths,
        output_path=config.output_dir / "variance_paths.png",
    )
    _plot_leverage_regime(
        rolling_corr=rolling_corr,
        output_path=config.output_dir / "leverage_effect_rolling_corr.png",
    )
    _plot_volatility_smile(
        smile_df=smile_df,
        output_path=config.output_dir / "volatility_smile.png",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the project-outline Heston/Langevin analyses "
            "with visualizations."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/outline_analysis"),
        help="Directory to write plots and analysis tables.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help=(
            "Optional path to BTC OHLCV CSV "
            "(with timestamp, open, high, low, close, volume)."
        ),
    )
    parser.add_argument(
        "--use-kaggle-data",
        action="store_true",
        help="Download and use Bitcoin minute data via kagglehub.",
    )
    parser.add_argument(
        "--kaggle-dataset",
        type=str,
        default="aklimarimi/bitcoin-historical-data-1min-interval",
        help="Kaggle dataset identifier for kagglehub download.",
    )
    parser.add_argument(
        "--resample-rule",
        type=str,
        default="1h",
        help="Pandas resample rule for close prices before returns (e.g. 1h, 4h, 1d).",
    )
    parser.add_argument("--n-sims", type=int, default=100_000)
    parser.add_argument("--smile-sims", type=int, default=40_000)
    parser.add_argument("--n-steps-per-year", type=int, default=365 * 24)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    config = OutlineAnalysisConfig(
        output_dir=args.output_dir,
        data_path=args.data_path,
        use_kaggle_data=args.use_kaggle_data,
        kaggle_dataset=args.kaggle_dataset,
        resample_rule=args.resample_rule,
        n_sims=args.n_sims,
        n_steps_per_year=args.n_steps_per_year,
        smile_sims=args.smile_sims,
        seed=args.seed,
    )
    run_outline_analysis(config)


if __name__ == "__main__":
    main()
