"""CAPM, beta, regression, and projection calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .risk_metrics import annualized_return
from .utils import TRADING_DAYS_PER_YEAR


def calculate_beta_covariance(
    stock_returns: pd.Series, benchmark_returns: pd.Series
) -> float:
    """Calculate beta as covariance(stock, market) divided by variance(market)."""
    aligned = pd.concat([stock_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return np.nan
    stock = aligned.iloc[:, 0]
    benchmark = aligned.iloc[:, 1]
    benchmark_variance = benchmark.var(ddof=1)
    if benchmark_variance == 0 or pd.isna(benchmark_variance):
        return np.nan
    return float(stock.cov(benchmark) / benchmark_variance)


def regression_statistics(
    stock_returns: pd.Series,
    benchmark_returns: pd.Series,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    """Estimate market-model beta, alpha, and R-squared using daily returns."""
    aligned = pd.concat([stock_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 3:
        return {"Regression Beta": np.nan, "Annualized Alpha": np.nan, "R-Squared": np.nan}

    y = aligned.iloc[:, 0].to_numpy(dtype=float)
    x = aligned.iloc[:, 1].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, deg=1)
    fitted = intercept + slope * x
    residual_sum_squares = np.sum((y - fitted) ** 2)
    total_sum_squares = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - residual_sum_squares / total_sum_squares if total_sum_squares else np.nan

    return {
        "Regression Beta": float(slope),
        "Annualized Alpha": float(intercept * trading_days),
        "R-Squared": float(r_squared),
    }


def calculate_capm_expected_return(
    beta: float, risk_free_rate: float, market_return: float
) -> float:
    """Calculate expected return using the Capital Asset Pricing Model."""
    return float(risk_free_rate + beta * (market_return - risk_free_rate))


def build_capm_table(
    stock_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    risk_free_rate: float,
    market_return: float,
) -> pd.DataFrame:
    """Create a CAPM results table for a set of stocks."""
    rows = []
    market_risk_premium = market_return - risk_free_rate

    for ticker in stock_returns.columns:
        beta = calculate_beta_covariance(stock_returns[ticker], benchmark_returns)
        regression = regression_statistics(stock_returns[ticker], benchmark_returns)
        capm_expected = calculate_capm_expected_return(
            beta=beta,
            risk_free_rate=risk_free_rate,
            market_return=market_return,
        )
        actual_return = annualized_return(stock_returns[ticker])

        rows.append(
            {
                "Ticker": ticker,
                "Beta": beta,
                "Regression Beta": regression["Regression Beta"],
                "Annualized Alpha": regression["Annualized Alpha"],
                "R-Squared": regression["R-Squared"],
                "Risk-Free Rate": risk_free_rate,
                "Market Return": market_return,
                "Market Risk Premium": market_risk_premium,
                "CAPM Expected Return": capm_expected,
                "Actual Annualized Return": actual_return,
                "Actual Minus CAPM": actual_return - capm_expected,
            }
        )

    return pd.DataFrame(rows).sort_values("Ticker").reset_index(drop=True)


def rolling_beta(
    stock_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 126,
) -> pd.Series:
    """Calculate rolling covariance/variance beta over a moving window."""
    aligned = pd.concat([stock_returns, benchmark_returns], axis=1).dropna()
    stock = aligned.iloc[:, 0]
    benchmark = aligned.iloc[:, 1]
    rolling_covariance = stock.rolling(window).cov(benchmark)
    rolling_variance = benchmark.rolling(window).var()
    return rolling_covariance / rolling_variance


def build_projection_table(
    latest_prices: pd.Series,
    capm_results: pd.DataFrame,
    horizon_months: int,
) -> pd.DataFrame:
    """Project theoretical future value using annualized CAPM expected return."""
    fraction_of_year = horizon_months / 12
    rows = []

    for _, row in capm_results.iterrows():
        ticker = row["Ticker"]
        current_price = float(latest_prices.loc[ticker])
        capm_return = float(row["CAPM Expected Return"])
        projected_price = current_price * ((1 + capm_return) ** fraction_of_year)
        expected_change = projected_price / current_price - 1
        rows.append(
            {
                "Ticker": ticker,
                "Current Price": current_price,
                "CAPM Expected Annual Return": capm_return,
                "Projection Horizon": f"{horizon_months} month"
                if horizon_months == 1
                else f"{horizon_months} months",
                "CAPM-Implied Projected Price": projected_price,
                "Expected Percent Change": expected_change,
            }
        )

    return pd.DataFrame(rows)

