"""Focused tests for the core financial calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.capm import (
    build_projection_table,
    calculate_beta_covariance,
    calculate_capm_expected_return,
)
from src.risk_metrics import annualized_return, maximum_drawdown, sharpe_ratio


def test_beta_covariance_matches_known_relationship() -> None:
    benchmark = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01])
    stock = benchmark * 1.5

    beta = calculate_beta_covariance(stock, benchmark)

    assert beta == pytest_approx(1.5)


def test_capm_expected_return_formula() -> None:
    expected_return = calculate_capm_expected_return(
        beta=1.2,
        risk_free_rate=0.04,
        market_return=0.10,
    )

    assert expected_return == pytest_approx(0.112)


def test_annualized_return_uses_geometric_growth() -> None:
    returns = pd.Series([0.01] * 252)

    result = annualized_return(returns)

    assert result == pytest_approx((1.01**252) - 1)


def test_maximum_drawdown_detects_peak_to_trough_loss() -> None:
    returns = pd.Series([0.10, -0.20, 0.05])

    result = maximum_drawdown(returns)

    assert result == pytest_approx(-0.20)


def test_projection_compounds_capm_expected_return() -> None:
    latest_prices = pd.Series({"ABC": 100.0})
    capm_results = pd.DataFrame(
        [{"Ticker": "ABC", "CAPM Expected Return": 0.12}]
    )

    projection = build_projection_table(latest_prices, capm_results, horizon_months=6)

    assert projection.loc[0, "CAPM-Implied Projected Price"] == pytest_approx(
        100 * (1.12**0.5)
    )


def test_sharpe_ratio_is_finite_for_nonzero_volatility() -> None:
    returns = pd.Series([0.01, -0.005, 0.003, 0.012, -0.004] * 20)

    result = sharpe_ratio(returns, risk_free_rate=0.02)

    assert np.isfinite(result)


def pytest_approx(value: float):
    """Tiny wrapper to keep pytest imported only where assertions need it."""
    import pytest

    return pytest.approx(value)

