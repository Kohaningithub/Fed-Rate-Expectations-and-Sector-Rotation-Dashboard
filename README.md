# Fed Rate Expectations and Sector Rotation Dashboard

An investment research dashboard that connects Federal Reserve rate expectations, yield-curve conditions, and live US sector ETF performance.

The application is designed as a client-facing market research tool. It sources live macro and ETF market data, engineers policy and rotation indicators, classifies market regimes, and presents the output as an executive dashboard for investment discussions.

## What It Shows

- Whether market rates imply easing, tightening, or neutral policy conditions.
- Which sectors are leading on 1M, 3M, 6M, and YTD returns.
- How sector performance changes across policy regimes.
- Whether current sector rotation is consistent with rate expectations and yield-curve pressure.

## Data Sources

- FRED public CSV endpoints: Effective Fed Funds, Fed target upper bound, 2Y Treasury, 10Y Treasury, and SOFR.
- Yahoo Finance chart API: Daily adjusted price data for SPY and SPDR sector ETFs.

The dashboard uses live data by default. If either source is unavailable, the app shows a data connection error so the user can trust that displayed analysis is sourced from current market data.

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- FRED public CSV endpoints
- Yahoo Finance chart API for ETF price data

## Run Locally

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The dashboard opens in live mode by default using FRED and Yahoo Finance chart data.

## Deploy on Streamlit Community Cloud

- Repository: `Kohaningithub/Fed-Rate-Expectations-and-Sector-Rotation-Dashboard`
- Branch: `main`
- Main file path: `app.py`

## Market Interpretation

The `Market Interpretation` view generates a client-facing explanation from the current dashboard state without calling an external AI API. It uses transparent rules based on:

- Current policy-rate signals.
- Top and bottom sector rotation rankings.
- 1M, 3M, 6M, and YTD sector returns.
- Regime-level sector performance.
- Data-through dates for FRED and Yahoo Finance.

It does not make buy, sell, hold, price-target, or allocation recommendations. The goal is to help users understand what the dashboard is showing and what questions to monitor next.

## Run Tests

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests
```

## Analytical Method

The dashboard uses a transparent expectations proxy:

```text
Policy expectations gap = 2Y Treasury yield - Effective Fed Funds Rate
```

Interpretation:

- Negative gap: the market is pricing easier policy relative to the current policy rate.
- Positive gap: the market is pricing higher-for-longer or additional tightening risk.
- Flat or inverted 10Y-2Y curve: restrictive or late-cycle conditions may be present.

Sector rotation is scored with a weighted blend of:

- 3M return
- 6M return
- 3M annualized volatility
- 6M max drawdown

## Resume Bullets

- Built a Streamlit investment research dashboard linking Fed policy expectations, Treasury yield-curve signals, and US sector ETF rotation.
- Engineered macro-financial indicators from FRED and ETF price data from Yahoo Finance, including a 2Y-Fed Funds policy expectations proxy and 10Y-2Y curve regime signal.
- Developed a sector rotation scoring model combining momentum, volatility, and drawdown to rank sector leadership under different monetary policy regimes.

## Interview Talking Points

- Why the 2Y Treasury yield can be used as a practical proxy for near-term policy expectations.
- How sector leadership often changes under tightening, easing, and inverted-curve regimes.
- Why the dashboard separates market screening from investment recommendation.
- How the project could be extended with Fed funds futures, macro surprises, or earnings revisions.

## Data Caveats

This project is an analytical screen, not investment advice. FRED and Yahoo Finance are public data sources, but production investment workflows should include formal data validation, adjusted-return checks, and source licensing review.
