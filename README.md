# Equity CAPM Research Dashboard

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Finance](https://img.shields.io/badge/Finance-CAPM%20%7C%20Beta%20%7C%20Risk-brightgreen)
![Tests](https://github.com/Raphael-Azerad/EquityCAPM/actions/workflows/tests.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A Streamlit dashboard for comparing public equities through CAPM, beta, realized returns, volatility, drawdown, Sharpe ratio, and benchmark correlation. I built it to make the assumptions behind expected-return analysis visible instead of hiding them inside a spreadsheet.

<p align="center">
  <img src="assets/screenshots/dashboard_overview.svg" alt="Equity CAPM Research Dashboard screenshot" width="900">
</p>

## Key Findings

Using the default analysis window and ticker set, the dashboard highlights four different risk profiles:

- **Walmart (WMT)** had the strongest risk-adjusted result in the sample, with a Sharpe ratio of **0.82** and realized annualized return of about **21.3%**.
- **Airbnb (ABNB)** had the highest beta at **1.55**, but its realized annualized return was about **-1.6%**, leaving the largest negative gap versus its CAPM-implied expected return.
- **JPMorgan Chase (JPM)** tracked the market more closely than the other names, with beta near **0.90** and the highest benchmark correlation in the group.
- **Lockheed Martin (LMT)** had the lowest beta at **0.24**, showing a more defensive relationship to the benchmark over the sample period.

These findings are not investment recommendations. They show how the same CAPM framework can separate market sensitivity, realized performance, and risk-adjusted behavior across different companies.

## What The Dashboard Does

The app downloads market data, aligns stock and benchmark returns, calculates beta, and compares CAPM expected return with what actually happened over the selected period. The default benchmark is `SPY`, used as a practical S&P 500 proxy.

The dashboard includes:

- CAPM expected return and actual annualized return
- Manual beta from covariance divided by benchmark variance
- Regression beta, annualized alpha, and R-squared
- Volatility, Sharpe ratio, maximum drawdown, and cumulative return
- Correlation analysis against the benchmark
- CAPM-implied projection table for 1, 3, 6, or 12 months
- Downloadable CSV outputs for the main result tables

## Default Companies

| Company | Ticker |
| --- | --- |
| Walmart | WMT |
| JPMorgan Chase | JPM |
| Airbnb | ABNB |
| Lockheed Martin | LMT |

The ticker list, benchmark, date range, risk-free rate, market return assumption, and projection horizon can all be changed from the sidebar.

## Methodology

The dashboard uses daily price data from Yahoo Finance through `yfinance`. It converts prices into daily returns, aligns each stock with the benchmark, and then calculates beta as:

```text
Beta = Covariance(stock returns, benchmark returns) / Variance(benchmark returns)
```

CAPM expected return is calculated as:

```text
Expected Return = Risk-Free Rate + Beta x (Market Return - Risk-Free Rate)
```

The risk-free rate can be pulled from the 13-week Treasury bill proxy (`^IRX`) or entered manually. The market return can use the benchmark's historical annualized return or a manual assumption.

## Limitations

CAPM is useful, but it is intentionally simplified:

- It relies heavily on beta, which only measures sensitivity to the chosen benchmark.
- Historical returns, volatility, and correlations may not hold in the future.
- The expected market return and risk-free rate are assumptions, not facts.
- The model does not directly account for company fundamentals, valuation, liquidity, sector effects, or changing macro conditions.
- CAPM-implied projections are expected-return illustrations, not price forecasts.

## Project Structure

```text
.
|-- app.py
|-- src/
|   |-- data_loader.py
|   |-- capm.py
|   |-- risk_metrics.py
|   |-- visualizations.py
|   `-- utils.py
|-- notebooks/
|   `-- capm_analysis.ipynb
|-- scripts/
|   `-- generate_sample_outputs.py
|-- tests/
|   `-- test_capm.py
|-- outputs/
|   |-- charts/
|   `-- sample_results/
|-- assets/
|   `-- screenshots/
|-- requirements.txt
|-- LICENSE
`-- README.md
```

## Installation

```bash
git clone https://github.com/Raphael-Azerad/EquityCAPM.git
cd EquityCAPM
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run The Dashboard

```bash
streamlit run app.py
```

## Generate Sample Outputs

```bash
python scripts/generate_sample_outputs.py
```

Sample CSV outputs are saved under `outputs/sample_results/`. Static charts can be regenerated under `outputs/charts/`.

## Disclaimer

This project is for research and education only. It is not investment advice. CAPM is a simplified expected-return framework and should not be interpreted as a guarantee of future returns or prices.

## Author

Raphael Azerad
