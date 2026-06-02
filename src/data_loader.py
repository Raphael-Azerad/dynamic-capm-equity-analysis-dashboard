"""Market data retrieval and cleaning utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable

import pandas as pd
import yfinance as yf

from .utils import DEFAULT_MANUAL_RISK_FREE_RATE, parse_tickers


@dataclass(frozen=True)
class PriceData:
    """Container for aligned stock and benchmark price data."""

    prices: pd.DataFrame
    stock_tickers: list[str]
    benchmark: str
    missing_tickers: list[str]


@dataclass(frozen=True)
class RiskFreeRate:
    """Risk-free-rate assumption with source metadata."""

    rate: float
    source: str
    as_of: str
    used_fallback: bool = False


def _to_yfinance_end_date(end_date: date | datetime | str) -> str:
    """Convert a user end date to yfinance's exclusive end-date convention."""
    parsed = pd.to_datetime(end_date).date()
    return (parsed + timedelta(days=1)).isoformat()


def _extract_close_prices(raw: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """Extract adjusted close-equivalent prices from a yfinance result."""
    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        first_level = raw.columns.get_level_values(0)
        second_level = raw.columns.get_level_values(1)
        if "Close" in first_level:
            close = raw["Close"]
        elif "Adj Close" in first_level:
            close = raw["Adj Close"]
        elif "Close" in second_level:
            close = raw.xs("Close", axis=1, level=1)
        elif "Adj Close" in second_level:
            close = raw.xs("Adj Close", axis=1, level=1)
        else:
            raise ValueError("Could not locate close price columns in market data.")
    elif "Close" in raw.columns:
        close = raw["Close"]
    elif "Adj Close" in raw.columns:
        close = raw["Adj Close"]
    else:
        raise ValueError("Could not locate close price columns in market data.")

    if isinstance(close, pd.Series):
        close = close.to_frame(symbols[0])

    close.columns = [str(column).upper() for column in close.columns]
    return close.apply(pd.to_numeric, errors="coerce")


def load_price_data(
    tickers: Iterable[str],
    benchmark: str,
    start_date: date | datetime | str,
    end_date: date | datetime | str,
) -> PriceData:
    """Download, clean, and align daily close prices for stocks and a benchmark."""
    stock_tickers = parse_tickers(tickers)
    benchmark = benchmark.strip().upper()
    symbols = list(dict.fromkeys([*stock_tickers, benchmark]))
    if not stock_tickers:
        raise ValueError("Please provide at least one stock ticker.")

    raw = yf.download(
        symbols,
        start=pd.to_datetime(start_date).date().isoformat(),
        end=_to_yfinance_end_date(end_date),
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="column",
    )
    close = _extract_close_prices(raw, symbols)
    if close.empty:
        raise ValueError("No price data was returned. Try a different date range.")

    available = [symbol for symbol in symbols if symbol in close.columns]
    missing = [symbol for symbol in symbols if symbol not in available]
    if benchmark not in available:
        raise ValueError(f"Benchmark '{benchmark}' was not returned by the data source.")

    selected = close[available].sort_index()
    selected = selected.loc[~selected.index.duplicated(keep="last")]
    selected = selected.ffill().dropna(how="all")
    selected = selected.dropna(axis=1, how="all")

    usable_stocks = [ticker for ticker in stock_tickers if ticker in selected.columns]
    missing.extend([ticker for ticker in stock_tickers if ticker not in usable_stocks])
    selected_columns = [*usable_stocks, benchmark]
    selected = selected[selected_columns].dropna(how="any")

    if selected.empty or len(selected) < 30:
        raise ValueError(
            "Not enough aligned observations were available after cleaning the data."
        )

    return PriceData(
        prices=selected,
        stock_tickers=usable_stocks,
        benchmark=benchmark,
        missing_tickers=sorted(set(missing)),
    )


def calculate_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate aligned daily simple returns."""
    returns = prices.pct_change(fill_method=None).replace([float("inf"), -float("inf")], pd.NA)
    return returns.dropna(how="any")


def fetch_risk_free_rate(default_rate: float = DEFAULT_MANUAL_RISK_FREE_RATE) -> RiskFreeRate:
    """Fetch a short-term Treasury yield proxy, falling back to a manual assumption."""
    try:
        treasury = yf.download("^IRX", period="10d", auto_adjust=False, progress=False)
        close = treasury["Close"].dropna()
        if isinstance(close, pd.DataFrame):
            close = close.squeeze("columns")
        latest_value = float(close.iloc[-1])
        if latest_value <= 0:
            raise ValueError("Treasury proxy returned a non-positive value.")
        rate = latest_value / 100 if latest_value > 1 else latest_value
        as_of = close.index[-1].strftime("%Y-%m-%d")
        return RiskFreeRate(
            rate=rate,
            source="13-week Treasury bill yield proxy (^IRX via Yahoo Finance)",
            as_of=as_of,
            used_fallback=False,
        )
    except Exception:
        return RiskFreeRate(
            rate=default_rate,
            source="Manual fallback assumption",
            as_of=datetime.utcnow().strftime("%Y-%m-%d"),
            used_fallback=True,
        )
