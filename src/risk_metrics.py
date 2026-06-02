"""Risk, return, and drawdown metrics for equity analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import TRADING_DAYS_PER_YEAR


def annualized_return(
    returns: pd.Series, trading_days: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Calculate geometric annualized return from periodic returns."""
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    cumulative_growth = (1 + clean).prod()
    return float(cumulative_growth ** (trading_days / len(clean)) - 1)


def annualized_volatility(
    returns: pd.Series, trading_days: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Calculate annualized volatility from daily returns."""
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    return float(clean.std(ddof=1) * np.sqrt(trading_days))


def cumulative_return(returns: pd.Series) -> float:
    """Calculate total cumulative return over the analysis window."""
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    return float((1 + clean).prod() - 1)


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Calculate a simple annualized Sharpe ratio."""
    volatility = annualized_volatility(returns, trading_days)
    if pd.isna(volatility) or volatility == 0:
        return np.nan
    return float((annualized_return(returns, trading_days) - risk_free_rate) / volatility)


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Return the drawdown path from a return series."""
    clean = returns.dropna()
    wealth_index = (1 + clean).cumprod()
    running_peak = wealth_index.cummax()
    return wealth_index / running_peak - 1


def maximum_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown as the minimum drawdown value."""
    drawdowns = drawdown_series(returns)
    if drawdowns.empty:
        return np.nan
    return float(drawdowns.min())


def rolling_return(
    returns: pd.Series,
    window: int = 63,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Calculate rolling annualized return over a specified daily window."""
    return (1 + returns).rolling(window).apply(
        lambda values: values.prod() ** (trading_days / window) - 1,
        raw=True,
    )


def build_risk_metrics_table(
    stock_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    risk_free_rate: float,
) -> pd.DataFrame:
    """Create a table of core risk and return metrics for each stock."""
    rows = []
    for ticker in stock_returns.columns:
        aligned = pd.concat([stock_returns[ticker], benchmark_returns], axis=1).dropna()
        stock = aligned.iloc[:, 0]
        benchmark = aligned.iloc[:, 1]
        rows.append(
            {
                "Ticker": ticker,
                "Annualized Return": annualized_return(stock),
                "Annualized Volatility": annualized_volatility(stock),
                "Sharpe Ratio": sharpe_ratio(stock, risk_free_rate),
                "Maximum Drawdown": maximum_drawdown(stock),
                "Cumulative Return": cumulative_return(stock),
                "Benchmark Correlation": float(stock.corr(benchmark)),
            }
        )

    metrics = pd.DataFrame(rows)
    metrics["Risk-Adjusted Rank"] = (
        metrics["Sharpe Ratio"].rank(ascending=False, method="dense").astype("Int64")
    )
    return metrics.sort_values("Risk-Adjusted Rank").reset_index(drop=True)


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Return a correlation matrix for stock and benchmark returns."""
    return returns.dropna().corr()

