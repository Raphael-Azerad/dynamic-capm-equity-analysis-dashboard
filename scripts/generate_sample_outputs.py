"""Generate sample result tables and chart images for the project README."""

from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
os.environ.setdefault("MPLBACKEND", "Agg")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capm import build_capm_table, build_projection_table
from src.data_loader import calculate_daily_returns, fetch_risk_free_rate, load_price_data
from src.risk_metrics import annualized_return, build_risk_metrics_table, correlation_matrix
from src.utils import DEFAULT_BENCHMARK, DEFAULT_TICKERS, default_date_range, ensure_directory
from src.visualizations import save_static_charts


def main() -> None:
    """Run the default WMT/JPM/ABNB/LMT analysis and save sample outputs."""
    start_date, end_date = default_date_range()
    price_data = load_price_data(
        tickers=DEFAULT_TICKERS,
        benchmark=DEFAULT_BENCHMARK,
        start_date=start_date,
        end_date=end_date,
    )
    prices = price_data.prices
    returns = calculate_daily_returns(prices)
    stock_returns = returns[price_data.stock_tickers]
    benchmark_returns = returns[price_data.benchmark]

    risk_free = fetch_risk_free_rate()
    market_return = annualized_return(benchmark_returns)

    capm_results = build_capm_table(
        stock_returns=stock_returns,
        benchmark_returns=benchmark_returns,
        risk_free_rate=risk_free.rate,
        market_return=market_return,
    )
    risk_metrics = build_risk_metrics_table(
        stock_returns=stock_returns,
        benchmark_returns=benchmark_returns,
        risk_free_rate=risk_free.rate,
    )
    correlations = correlation_matrix(returns)
    projections = build_projection_table(
        latest_prices=prices.iloc[-1],
        capm_results=capm_results,
        horizon_months=12,
    )

    results_dir = ensure_directory(ROOT / "outputs" / "sample_results")
    charts_dir = ensure_directory(ROOT / "outputs" / "charts")

    capm_results.to_csv(results_dir / "capm_results.csv", index=False)
    risk_metrics.to_csv(results_dir / "risk_metrics.csv", index=False)
    projections.to_csv(results_dir / "capm_projection.csv", index=False)
    saved_charts = save_static_charts(
        prices=prices,
        returns=returns,
        capm_results=capm_results,
        risk_metrics=risk_metrics,
        correlation=correlations,
        output_dir=charts_dir,
    )

    print("Generated sample outputs:")
    print(f"- Risk-free rate: {risk_free.rate:.4%} ({risk_free.source}, as of {risk_free.as_of})")
    for path in [
        results_dir / "capm_results.csv",
        results_dir / "risk_metrics.csv",
        results_dir / "capm_projection.csv",
        *saved_charts,
    ]:
        print(f"- {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
