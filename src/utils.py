"""Small formatting and validation helpers used across the project."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import re
from typing import Iterable

import pandas as pd


DEFAULT_TICKERS = ["WMT", "JPM", "ABNB", "LMT"]
DEFAULT_BENCHMARK = "SPY"
TRADING_DAYS_PER_YEAR = 252
DEFAULT_MANUAL_RISK_FREE_RATE = 0.045
DEFAULT_MANUAL_MARKET_RETURN = 0.085


def default_date_range(years: int = 5) -> tuple[date, date]:
    """Return a conservative default date range ending today."""
    end = date.today()
    start = end - timedelta(days=365 * years + 2)
    return start, end


def parse_tickers(raw: str | Iterable[str]) -> list[str]:
    """Normalize a ticker list from sidebar text or an iterable."""
    if isinstance(raw, str):
        parts = re.split(r"[\s,;]+", raw)
    else:
        parts = list(raw)

    tickers: list[str] = []
    for part in parts:
        ticker = str(part).strip().upper()
        if ticker and ticker not in tickers:
            tickers.append(ticker)
    return tickers


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def format_percent(value: float | int | None, digits: int = 2) -> str:
    """Format a decimal return as a human-readable percentage."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * 100:.{digits}f}%"


def format_number(value: float | int | None, digits: int = 2) -> str:
    """Format a numeric value with a fixed number of decimals."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.{digits}f}"


def format_currency(value: float | int | None, digits: int = 2) -> str:
    """Format a value as a US dollar amount."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"${value:,.{digits}f}"


def percent_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Return a display copy where selected decimal columns are in percent units."""
    display = df.copy()
    for column in columns:
        if column in display.columns:
            display[column] = display[column] * 100
    return display

