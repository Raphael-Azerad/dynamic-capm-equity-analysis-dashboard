"""Plotly dashboard charts and static chart export helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

from .capm import rolling_beta
from .risk_metrics import drawdown_series
from .utils import ensure_directory


COLOR_SEQUENCE = ["#2563EB", "#0F766E", "#F59E0B", "#DC2626", "#7C3AED", "#475569"]
PLOTLY_TEMPLATE = "plotly_white"


def _apply_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        template=PLOTLY_TEMPLATE,
        colorway=COLOR_SEQUENCE,
        hovermode="x unified",
        margin={"l": 20, "r": 20, "t": 60, "b": 40},
        legend={"orientation": "h", "y": -0.18},
        font={"family": "Inter, Arial, sans-serif", "size": 13},
    )
    return fig


def normalized_price_figure(prices: pd.DataFrame) -> go.Figure:
    normalized = prices.divide(prices.iloc[0]).multiply(100)
    fig = px.line(normalized, x=normalized.index, y=normalized.columns)
    fig.update_yaxes(title="Growth of $100")
    fig.update_xaxes(title="")
    return _apply_layout(fig, "Normalized Price Performance")


def expected_vs_actual_figure(capm_results: pd.DataFrame) -> go.Figure:
    plot_data = capm_results[["Ticker", "CAPM Expected Return", "Actual Annualized Return"]].melt(
        id_vars="Ticker", var_name="Return Type", value_name="Return"
    )
    plot_data["Return"] = plot_data["Return"] * 100
    fig = px.bar(plot_data, x="Ticker", y="Return", color="Return Type", barmode="group", text_auto=".1f")
    fig.update_yaxes(title="Annualized Return (%)")
    fig.update_xaxes(title="")
    return _apply_layout(fig, "CAPM Expected Return vs Historical Return")


def beta_bar_figure(capm_results: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        capm_results.sort_values("Beta"),
        x="Ticker",
        y="Beta",
        color="Beta",
        color_continuous_scale=["#0F766E", "#F59E0B", "#DC2626"],
        text_auto=".2f",
    )
    fig.add_hline(y=1, line_dash="dash", line_color="#334155", annotation_text="Market beta = 1.0")
    fig.update_xaxes(title="")
    return _apply_layout(fig, "Beta Compared With the Market")


def risk_return_scatter_figure(risk_metrics: pd.DataFrame, capm_results: pd.DataFrame) -> go.Figure:
    plot_data = risk_metrics.merge(capm_results[["Ticker", "Beta"]], on="Ticker")
    plot_data["Annualized Return"] *= 100
    plot_data["Annualized Volatility"] *= 100
    fig = px.scatter(
        plot_data,
        x="Annualized Volatility",
        y="Annualized Return",
        color="Ticker",
        size=plot_data["Beta"].abs().clip(lower=0.2),
        hover_data={"Beta": ":.2f", "Sharpe Ratio": ":.2f"},
    )
    fig.update_xaxes(title="Annualized Volatility (%)")
    fig.update_yaxes(title="Annualized Return (%)")
    return _apply_layout(fig, "Risk vs Return Profile")


def correlation_heatmap_figure(correlation: pd.DataFrame) -> go.Figure:
    fig = px.imshow(correlation, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig.update_xaxes(title="")
    fig.update_yaxes(title="")
    return _apply_layout(fig, "Return Correlation Heatmap")


def drawdown_figure(returns: pd.DataFrame) -> go.Figure:
    drawdowns = pd.DataFrame({column: drawdown_series(returns[column]) * 100 for column in returns.columns})
    fig = px.line(drawdowns, x=drawdowns.index, y=drawdowns.columns)
    fig.update_yaxes(title="Drawdown (%)")
    fig.update_xaxes(title="")
    return _apply_layout(fig, "Historical Drawdown")


def rolling_beta_figure(stock_returns: pd.DataFrame, benchmark_returns: pd.Series, window: int = 126) -> go.Figure:
    betas = pd.DataFrame({ticker: rolling_beta(stock_returns[ticker], benchmark_returns, window=window) for ticker in stock_returns.columns})
    fig = px.line(betas, x=betas.index, y=betas.columns)
    fig.update_yaxes(title=f"Rolling Beta ({window} trading days)")
    fig.update_xaxes(title="")
    return _apply_layout(fig, "Rolling Beta")


def save_static_charts(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    capm_results: pd.DataFrame,
    risk_metrics: pd.DataFrame,
    correlation: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    output_path = ensure_directory(output_dir)
    saved: list[Path] = []
    sns.set_theme(style="whitegrid", context="talk")

    normalized = prices.divide(prices.iloc[0]).multiply(100)
    ax = normalized.plot(figsize=(12, 6), linewidth=2.2)
    ax.set_title("Normalized Price Performance", loc="left", weight="bold")
    ax.set_ylabel("Growth of $100")
    ax.set_xlabel("")
    file_path = output_path / "normalized_price_performance.png"
    plt.tight_layout(); plt.savefig(file_path, dpi=180, bbox_inches="tight"); plt.close(); saved.append(file_path)

    comparison = capm_results[["Ticker", "CAPM Expected Return", "Actual Annualized Return"]].set_index("Ticker") * 100
    ax = comparison.plot(kind="bar", figsize=(10, 6))
    ax.set_title("CAPM Expected Return vs Historical Return", loc="left", weight="bold")
    ax.set_ylabel("Annualized Return (%)")
    file_path = output_path / "expected_vs_actual_returns.png"
    plt.tight_layout(); plt.savefig(file_path, dpi=180, bbox_inches="tight"); plt.close(); saved.append(file_path)

    ax = sns.barplot(data=capm_results.sort_values("Beta"), x="Ticker", y="Beta", hue="Ticker", legend=False)
    ax.axhline(1, color="#334155", linestyle="--", linewidth=1.4)
    ax.set_title("Beta Compared With the Market", loc="left", weight="bold")
    file_path = output_path / "beta_comparison.png"
    plt.tight_layout(); plt.savefig(file_path, dpi=180, bbox_inches="tight"); plt.close(); saved.append(file_path)

    scatter = risk_metrics.merge(capm_results[["Ticker", "Beta"]], on="Ticker")
    scatter["Annualized Return"] *= 100
    scatter["Annualized Volatility"] *= 100
    ax = sns.scatterplot(data=scatter, x="Annualized Volatility", y="Annualized Return", hue="Ticker", size="Beta", sizes=(120, 380))
    ax.set_title("Risk vs Return Profile", loc="left", weight="bold")
    file_path = output_path / "risk_return_scatter.png"
    plt.tight_layout(); plt.savefig(file_path, dpi=180, bbox_inches="tight"); plt.close(); saved.append(file_path)

    ax = sns.heatmap(correlation, annot=True, cmap="RdBu_r", vmin=-1, vmax=1, linewidths=0.5)
    ax.set_title("Return Correlation Heatmap", loc="left", weight="bold")
    file_path = output_path / "correlation_heatmap.png"
    plt.tight_layout(); plt.savefig(file_path, dpi=180, bbox_inches="tight"); plt.close(); saved.append(file_path)

    drawdowns = pd.DataFrame({column: drawdown_series(returns[column]) * 100 for column in returns.columns})
    ax = drawdowns.plot(figsize=(12, 6), linewidth=2)
    ax.set_title("Historical Drawdown", loc="left", weight="bold")
    ax.set_ylabel("Drawdown (%)")
    file_path = output_path / "drawdown.png"
    plt.tight_layout(); plt.savefig(file_path, dpi=180, bbox_inches="tight"); plt.close(); saved.append(file_path)

    return saved
