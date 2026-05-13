from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


DEFAULT_START_DATE = date(2018, 1, 1)
REQUEST_TIMEOUT_SECONDS = 8

FRED_SERIES = {
    "DFF": "fed_funds_effective",
    "DFEDTARU": "fed_target_upper",
    "DGS2": "ust_2y",
    "DGS10": "ust_10y",
    "SOFR": "sofr",
}

SECTOR_ETFS = {
    "XLC": "Communication Services",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLK": "Technology",
    "XLU": "Utilities",
}

BENCHMARKS = {
    "SPY": "S&P 500",
}

PRICE_TICKERS = {**BENCHMARKS, **SECTOR_ETFS}


@dataclass(frozen=True)
class DataLoadResult:
    macro: pd.DataFrame
    prices: pd.DataFrame
    metadata: dict[str, str]


def _read_remote_csv(url: str) -> pd.DataFrame:
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/csv,application/json,*/*",
        },
    )
    with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return pd.read_csv(response)


def _read_remote_json(url: str) -> dict:
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json,text/plain,*/*",
        },
    )
    with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def load_fred_series(series_id: str, start_date: date) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    frame = _read_remote_csv(url)
    date_column = "DATE" if "DATE" in frame.columns else "observation_date"
    if date_column not in frame.columns or series_id not in frame.columns:
        raise ValueError(f"Unexpected FRED response for {series_id}.")

    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    frame[series_id] = pd.to_numeric(frame[series_id].replace(".", np.nan), errors="coerce")
    frame = frame.dropna(subset=[date_column]).set_index(date_column).sort_index()
    frame = frame.loc[frame.index >= pd.Timestamp(start_date)]
    return frame[[series_id]]


def load_macro_data(start_date: date) -> pd.DataFrame:
    frames = []
    for series_id, column_name in FRED_SERIES.items():
        series = load_fred_series(series_id, start_date).rename(columns={series_id: column_name})
        frames.append(series)

    macro = pd.concat(frames, axis=1).sort_index().ffill()
    macro = macro.dropna(how="all")
    macro["curve_10y_2y"] = macro["ust_10y"] - macro["ust_2y"]
    macro["policy_expectations_gap"] = macro["ust_2y"] - macro["fed_funds_effective"]
    macro["policy_gap_6m_change"] = macro["policy_expectations_gap"].diff(126)
    macro["fed_funds_3m_change"] = macro["fed_funds_effective"].diff(63)
    return macro


def load_yahoo_chart_close(ticker: str, start_date: date) -> pd.Series:
    period1 = int(pd.Timestamp(start_date).timestamp())
    period2 = int(pd.Timestamp(date.today() + timedelta(days=1)).timestamp())
    query = urlencode(
        {
            "period1": period1,
            "period2": period2,
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true",
        }
    )
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?{query}"
    payload = _read_remote_json(url)
    chart = payload.get("chart", {})
    error = chart.get("error")
    if error:
        raise ValueError(f"Yahoo chart error for {ticker}: {error}")

    results = chart.get("result") or []
    if not results:
        raise ValueError(f"No Yahoo chart result for {ticker}.")

    result = results[0]
    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators", {})
    adjclose = indicators.get("adjclose") or []
    quote = indicators.get("quote") or []
    close_values = None
    if adjclose and adjclose[0].get("adjclose"):
        close_values = adjclose[0]["adjclose"]
    elif quote and quote[0].get("close"):
        close_values = quote[0]["close"]
    if not timestamps or not close_values:
        raise ValueError(f"No close history returned for {ticker}.")

    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(timestamps, unit="s").normalize(),
            "Close": pd.to_numeric(close_values, errors="coerce"),
        }
    )
    frame = frame.dropna(subset=["Date", "Close"]).drop_duplicates(subset=["Date"])
    frame = frame.set_index("Date").sort_index()
    if frame.empty:
        raise ValueError(f"No usable Yahoo price history returned for {ticker}.")
    return frame["Close"].rename(ticker)


def load_price_data(start_date: date) -> pd.DataFrame:
    frames = []
    failures = []
    for ticker in PRICE_TICKERS:
        try:
            frames.append(load_yahoo_chart_close(ticker, start_date))
        except Exception as exc:
            failures.append(f"{ticker}: {exc}")

    if len(frames) < 8:
        joined_failures = "; ".join(failures[:4])
        raise RuntimeError(f"Too few live price series loaded. {joined_failures}")

    prices = pd.concat(frames, axis=1).sort_index().ffill()
    prices.attrs["warnings"] = "; ".join(failures)
    return prices.dropna(how="all")


def load_dashboard_data(start_date: date = DEFAULT_START_DATE) -> DataLoadResult:
    macro = load_macro_data(start_date)
    prices = load_price_data(start_date)
    warnings = prices.attrs.get("warnings", "")
    return DataLoadResult(
        macro=macro,
        prices=prices,
        metadata={
            "source": "FRED macro series and Yahoo Finance chart API ETF prices",
            "warning": warnings,
        },
    )
