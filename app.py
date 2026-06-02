"""Streamlit dashboard for dynamic CAPM equity analysis."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.capm import build_capm_table, build_projection_table
from src.data_loader import calculate_daily_returns, fetch_risk_free_rate, load_price_data
from src.risk_metrics import (
    annualized_return,
    build_risk_metrics_table,
    correlation_matrix,
)
from src.utils import (
    DEFAULT_BENCHMARK,
    DEFAULT_MANUAL_MARKET_RETURN,
    DEFAULT_MANUAL_RISK_FREE_RATE,
    DEFAULT_TICKERS,
    default_date_range,
    format_currency,
    format_number,
    format_percent,
    parse_tickers,
    percent_columns,
)
from src.visualizations import (
    beta_bar_figure,
    correlation_heatmap_figure,
    drawdown_figure,
    expected_vs_actual_figure,
    normalized_price_figure,
    risk_return_scatter_figure,
    rolling_beta_figure,
)


st.set_page_config(
    page_title="Equity CAPM Research Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1320px;
    }
    h1, h2, h3 {
        letter-spacing: 0;
    }
    .metric-card {
        background: #111827;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem 1rem 0.75rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        min-height: 132px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .metric-label {
        color: #cbd5e1;
        font-size: 0.86rem;
        font-weight: 650;
        line-height: 1.2;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 1.7rem;
        font-weight: 750;
        line-height: 1.05;
        white-space: nowrap;
    }
    .metric-detail {
        background: rgba(16, 185, 129, 0.16);
        color: #86efac;
        border-radius: 999px;
        display: inline-block;
        font-size: 0.82rem;
        font-weight: 650;
        padding: 0.22rem 0.55rem;
        width: fit-content;
        max-width: 100%;
        white-space: normal;
    }
    div[data-testid="stMetricDelta"] {
        color: #86efac;
    }
    .small-muted {
        color: #64748b;
        font-size: 0.92rem;
        line-height: 1.45;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False, ttl=60 * 60)
def get_market_data(
    tickers: tuple[str, ...],
    benchmark: str,
    start_date,
    end_date,
):
    """Cache market data so the dashboard stays responsive."""
    return load_price_data(tickers, benchmark, start_date, end_date)


@st.cache_data(show_spinner=False, ttl=60 * 60)
def get_risk_free_rate():
    """Cache the risk-free-rate lookup."""
    return fetch_risk_free_rate()


def display_percentage_dataframe(df: pd.DataFrame, percent_cols: list[str]) -> pd.DataFrame:
    """Prepare a display dataframe with selected columns in percent points."""
    display = percent_columns(df, percent_cols)
    numeric_columns = display.select_dtypes(include="number").columns
    display[numeric_columns] = display[numeric_columns].round(3)
    return display


def metric_card(label: str, value: str, detail: str) -> str:
    """Return compact HTML for an executive summary metric."""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-detail">{detail}</div>
    </div>
    """


start_default, end_default = default_date_range()

with st.sidebar:
    st.header("Analysis Inputs")
    raw_tickers = st.text_area(
        "Stock tickers",
        value=", ".join(DEFAULT_TICKERS),
        help="Enter comma-separated public company tickers.",
    )
    benchmark = st.text_input("Benchmark", value=DEFAULT_BENCHMARK)
    start_date = st.date_input("Start date", value=start_default)
    end_date = st.date_input("End date", value=end_default)

    st.divider()
    st.subheader("Assumptions")
    risk_free_mode = st.radio(
        "Risk-free rate",
        ["Auto Treasury proxy", "Manual assumption"],
        help="Auto mode attempts to use the 13-week Treasury bill proxy (^IRX).",
    )
    manual_risk_free_rate = (
        st.number_input(
            "Manual risk-free rate (%)",
            min_value=0.0,
            max_value=20.0,
            value=DEFAULT_MANUAL_RISK_FREE_RATE * 100,
            step=0.10,
        )
        / 100
    )

    market_return_mode = st.radio(
        "Market return assumption",
        ["Use benchmark historical return", "Manual market return"],
        help="CAPM requires an expected market return assumption.",
    )
    manual_market_return = (
        st.number_input(
            "Manual market return (%)",
            min_value=-20.0,
            max_value=40.0,
            value=DEFAULT_MANUAL_MARKET_RETURN * 100,
            step=0.25,
        )
        / 100
    )

    horizon_label = st.selectbox(
        "Projection horizon",
        ["1 month", "3 months", "6 months", "12 months"],
        index=3,
    )
    horizon_months = int(horizon_label.split()[0])


st.title("Equity CAPM Research Dashboard")
st.markdown(
    """
    <p class="small-muted">
    A reusable Python dashboard for estimating CAPM-implied expected returns,
    comparing them with historical performance, and evaluating risk-adjusted
    equity behavior against a market benchmark.
    </p>
    """,
    unsafe_allow_html=True,
)

tickers = parse_tickers(raw_tickers)

if not tickers:
    st.warning("Enter at least one stock ticker to run the analysis.")
    st.stop()

if start_date >= end_date:
    st.error("The start date must be earlier than the end date.")
    st.stop()

try:
    with st.spinner("Loading market data and calculating metrics..."):
        price_data = get_market_data(tuple(tickers), benchmark, start_date, end_date)
        prices = price_data.prices
        returns = calculate_daily_returns(prices)
        stock_returns = returns[price_data.stock_tickers]
        benchmark_returns = returns[price_data.benchmark]

        if risk_free_mode == "Auto Treasury proxy":
            risk_free_info = get_risk_free_rate()
            risk_free_rate = risk_free_info.rate
        else:
            risk_free_info = None
            risk_free_rate = manual_risk_free_rate

        benchmark_market_return = annualized_return(benchmark_returns)
        market_return = (
            benchmark_market_return
            if market_return_mode == "Use benchmark historical return"
            else manual_market_return
        )

        capm_results = build_capm_table(
            stock_returns=stock_returns,
            benchmark_returns=benchmark_returns,
            risk_free_rate=risk_free_rate,
            market_return=market_return,
        )
        risk_metrics = build_risk_metrics_table(
            stock_returns=stock_returns,
            benchmark_returns=benchmark_returns,
            risk_free_rate=risk_free_rate,
        )
        correlations = correlation_matrix(returns)
        projections = build_projection_table(
            latest_prices=prices.iloc[-1],
            capm_results=capm_results,
            horizon_months=horizon_months,
        )
except Exception as exc:
    st.error(f"Analysis could not be completed: {exc}")
    st.stop()

if price_data.missing_tickers:
    st.warning(
        "Some tickers were unavailable or removed during cleaning: "
        + ", ".join(price_data.missing_tickers)
    )

best_sharpe = risk_metrics.sort_values("Sharpe Ratio", ascending=False).iloc[0]
highest_beta = capm_results.sort_values("Beta", ascending=False).iloc[0]
largest_gap = capm_results.assign(abs_gap=capm_results["Actual Minus CAPM"].abs()).sort_values(
    "abs_gap", ascending=False
).iloc[0]

st.subheader("Executive Summary")
summary_cols = st.columns(4)
summary_cols[0].markdown(
    metric_card(
        "Best Sharpe Ratio",
        best_sharpe["Ticker"],
        f"Sharpe {format_number(best_sharpe['Sharpe Ratio'])}",
    ),
    unsafe_allow_html=True,
)
summary_cols[1].markdown(
    metric_card(
        "Highest Beta",
        highest_beta["Ticker"],
        f"Beta {format_number(highest_beta['Beta'])}",
    ),
    unsafe_allow_html=True,
)
summary_cols[2].markdown(
    metric_card(
        "Market Return",
        format_percent(market_return),
        "Historical benchmark" if market_return_mode.startswith("Use") else "Manual input",
    ),
    unsafe_allow_html=True,
)
summary_cols[3].markdown(
    metric_card(
        "Risk-Free Rate",
        format_percent(risk_free_rate),
        "Treasury proxy"
        if risk_free_info and not risk_free_info.used_fallback
        else "Manual assumption",
    ),
    unsafe_allow_html=True,
)

st.caption(
    f"Analysis window: {prices.index.min().date()} to {prices.index.max().date()} "
    f"({len(returns):,} aligned trading days). Largest actual-vs-CAPM gap: "
    f"{largest_gap['Ticker']} at {format_percent(largest_gap['Actual Minus CAPM'])}."
)

if risk_free_info is not None:
    st.caption(f"Risk-free-rate source: {risk_free_info.source}; as of {risk_free_info.as_of}.")

st.subheader("CAPM Results")
capm_percent_cols = [
    "Annualized Alpha",
    "Risk-Free Rate",
    "Market Return",
    "Market Risk Premium",
    "CAPM Expected Return",
    "Actual Annualized Return",
    "Actual Minus CAPM",
]
st.dataframe(
    display_percentage_dataframe(capm_results, capm_percent_cols),
    use_container_width=True,
    hide_index=True,
)
st.download_button(
    "Download CAPM results CSV",
    capm_results.to_csv(index=False).encode("utf-8"),
    file_name="capm_results.csv",
    mime="text/csv",
)

return_tab, risk_tab, projection_tab, methodology_tab = st.tabs(
    ["Returns & Beta", "Risk Profile", "Projection", "Methodology"]
)

with return_tab:
    left, right = st.columns(2)
    with left:
        st.plotly_chart(expected_vs_actual_figure(capm_results), use_container_width=True)
    with right:
        st.plotly_chart(beta_bar_figure(capm_results), use_container_width=True)
    st.plotly_chart(normalized_price_figure(prices), use_container_width=True)
    st.plotly_chart(
        rolling_beta_figure(stock_returns, benchmark_returns),
        use_container_width=True,
    )

with risk_tab:
    risk_percent_cols = [
        "Annualized Return",
        "Annualized Volatility",
        "Maximum Drawdown",
        "Cumulative Return",
        "Benchmark Correlation",
    ]
    st.dataframe(
        display_percentage_dataframe(risk_metrics, risk_percent_cols),
        use_container_width=True,
        hide_index=True,
    )
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            risk_return_scatter_figure(risk_metrics, capm_results),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(correlation_heatmap_figure(correlations), use_container_width=True)
    st.plotly_chart(drawdown_figure(returns), use_container_width=True)

with projection_tab:
    st.markdown(
        """
        <p class="small-muted">
        This table compounds each stock's annual CAPM expected return over the selected
        horizon. It is a theoretical expected-return projection, not a price prediction.
        </p>
        """,
        unsafe_allow_html=True,
    )
    projection_percent_cols = [
        "CAPM Expected Annual Return",
        "Expected Percent Change",
    ]
    projection_display = display_percentage_dataframe(projections, projection_percent_cols)
    projection_display["Current Price"] = projections["Current Price"].map(format_currency)
    projection_display["CAPM-Implied Projected Price"] = projections[
        "CAPM-Implied Projected Price"
    ].map(format_currency)
    st.dataframe(projection_display, use_container_width=True, hide_index=True)
    st.download_button(
        "Download projection CSV",
        projections.to_csv(index=False).encode("utf-8"),
        file_name="capm_projection.csv",
        mime="text/csv",
    )
    st.info(
        "CAPM is not a price prediction model. These projections are theoretical "
        "estimates based on expected return assumptions and should not be interpreted "
        "as investment advice."
    )

with methodology_tab:
    st.markdown(
        """
        ### Methodology

        Beta is calculated manually from historical daily returns as covariance between
        a stock and the benchmark divided by benchmark variance. I also include a
        simple market-model regression beta, annualized alpha, and R-squared to show
        how much of each stock's return variation is explained by the benchmark.

        CAPM expected return is calculated as:

        `Risk-Free Rate + Beta x (Market Return - Risk-Free Rate)`

        The dashboard compares that theoretical required return with realized
        annualized historical return, volatility, drawdown, Sharpe ratio, benchmark
        correlation, and relative ranking. Market data is sourced through Yahoo Finance
        via `yfinance`.

        ### Disclaimer

        This project is for research, education, and portfolio demonstration only. It
        is not investment advice. CAPM is a simplified expected-return framework and
        should not be interpreted as a guarantee of future returns or prices.
        """
    )
