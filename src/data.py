from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from io import BytesIO
import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


DEFAULT_START_DATE = date(2018, 1, 1)
REQUEST_TIMEOUT_SECONDS = 6
REQUEST_RETRIES = 2
SNAPSHOT_DIR = Path(__file__).resolve().parents[1] / "data"
SNAPSHOT_MACRO_FILE = SNAPSHOT_DIR / "live_macro_snapshot.csv"
SNAPSHOT_PRICES_FILE = SNAPSHOT_DIR / "live_prices_snapshot.csv"
SNAPSHOT_METADATA_FILE = SNAPSHOT_DIR / "live_snapshot_metadata.json"

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
    payload = _read_remote_bytes(url, "text/csv,application/json,*/*")
    return pd.read_csv(BytesIO(payload))


def _read_remote_json(url: str) -> dict:
    payload = _read_remote_bytes(url, "application/json,text/plain,*/*")
    return json.loads(payload.decode("utf-8"))


def _read_remote_bytes(url: str, accept: str) -> bytes:
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": accept,
            "Connection": "close",
        },
    )

    last_error: Exception | None = None
    for attempt in range(REQUEST_RETRIES + 1):
        try:
            with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                return response.read()
        except Exception as exc:
            last_error = exc
            if attempt < REQUEST_RETRIES:
                time.sleep(0.35 * (attempt + 1))
    raise RuntimeError(f"Request failed after {REQUEST_RETRIES + 1} attempts: {last_error}")


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
    try:
        series_query = ",".join(FRED_SERIES)
        frame = _read_remote_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_query}")
        date_column = "DATE" if "DATE" in frame.columns else "observation_date"
        if date_column not in frame.columns:
            raise ValueError("Unexpected FRED bulk response: missing date column.")
        frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
        frame = frame.dropna(subset=[date_column]).set_index(date_column).sort_index()
        frame = frame.loc[frame.index >= pd.Timestamp(start_date)]
        macro = frame.rename(columns=FRED_SERIES)
        macro = macro[list(FRED_SERIES.values())]
        for column in macro.columns:
            macro[column] = pd.to_numeric(macro[column].replace(".", np.nan), errors="coerce")
    except Exception:
        frames = []
        for series_id, column_name in FRED_SERIES.items():
            series = load_fred_series(series_id, start_date).rename(columns={series_id: column_name})
            frames.append(series)
        macro = pd.concat(frames, axis=1).sort_index()

    macro = macro.ffill()
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


def load_stooq_close(ticker: str, start_date: date) -> pd.Series:
    start = pd.Timestamp(start_date).strftime("%Y%m%d")
    end = pd.Timestamp(date.today() + timedelta(days=1)).strftime("%Y%m%d")
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d&d1={start}&d2={end}"
    frame = _read_remote_csv(url)
    if "Date" not in frame.columns or "Close" not in frame.columns:
        raise ValueError(f"Unexpected Stooq response for {ticker}.")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
    frame = frame.dropna(subset=["Date", "Close"]).drop_duplicates(subset=["Date"])
    frame = frame.set_index("Date").sort_index()
    if frame.empty:
        raise ValueError(f"No usable Stooq price history returned for {ticker}.")
    return frame["Close"].rename(ticker)


def load_live_price_close(ticker: str, start_date: date) -> tuple[str, pd.Series, str]:
    try:
        return ticker, load_yahoo_chart_close(ticker, start_date), "Yahoo Finance"
    except Exception as yahoo_error:
        try:
            return ticker, load_stooq_close(ticker, start_date), "Stooq"
        except Exception as stooq_error:
            raise RuntimeError(f"Yahoo failed: {yahoo_error}; Stooq failed: {stooq_error}") from stooq_error


def load_price_data(start_date: date) -> pd.DataFrame:
    frames = []
    failures = []
    source_by_ticker = {}

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(load_live_price_close, ticker, start_date): ticker for ticker in PRICE_TICKERS}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                loaded_ticker, series, source = future.result()
                frames.append(series)
                source_by_ticker[loaded_ticker] = source
            except Exception as exc:
                failures.append(f"{ticker}: {exc}")

    if len(frames) < 8:
        joined_failures = "; ".join(failures[:4])
        raise RuntimeError(f"Too few live price series loaded. {joined_failures}")

    prices = pd.concat(frames, axis=1).sort_index().ffill()
    prices.attrs["warnings"] = "; ".join(failures)
    prices.attrs["source_by_ticker"] = source_by_ticker
    return prices.dropna(how="all")


def build_live_dashboard_data(start_date: date = DEFAULT_START_DATE) -> DataLoadResult:
    macro = load_macro_data(start_date)
    prices = load_price_data(start_date)
    warnings = prices.attrs.get("warnings", "")
    source_by_ticker = prices.attrs.get("source_by_ticker", {})
    price_sources = sorted(set(source_by_ticker.values()))
    return DataLoadResult(
        macro=macro,
        prices=prices,
        metadata={
            "source": f"FRED macro series and {'/'.join(price_sources)} ETF prices",
            "source_mode": "live",
            "warning": warnings,
            "price_sources": ", ".join(f"{ticker}: {source}" for ticker, source in sorted(source_by_ticker.items())),
        },
    )


def _read_snapshot_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["Date"])
    if "Date" not in frame.columns:
        raise ValueError(f"Snapshot file is missing Date column: {path.name}")
    return frame.set_index("Date").sort_index()


def load_snapshot_dashboard_data(start_date: date, live_error: Exception) -> DataLoadResult:
    missing = [
        path.name
        for path in (SNAPSHOT_MACRO_FILE, SNAPSHOT_PRICES_FILE, SNAPSHOT_METADATA_FILE)
        if not path.exists()
    ]
    if missing:
        raise RuntimeError(f"Live data failed and snapshot files are missing: {', '.join(missing)}") from live_error

    macro = _read_snapshot_frame(SNAPSHOT_MACRO_FILE)
    prices = _read_snapshot_frame(SNAPSHOT_PRICES_FILE)
    macro = macro.loc[macro.index >= pd.Timestamp(start_date)]
    prices = prices.loc[prices.index >= pd.Timestamp(start_date)]
    if macro.empty or prices.empty:
        raise RuntimeError("Snapshot data is empty for the selected start date.") from live_error

    with SNAPSHOT_METADATA_FILE.open("r", encoding="utf-8") as file:
        snapshot_metadata = json.load(file)

    return DataLoadResult(
        macro=macro,
        prices=prices,
        metadata={
            "source": "Committed live-data snapshot from FRED and public ETF price feeds",
            "source_mode": "snapshot",
            "warning": (
                "Live public data timed out in this environment; using the latest committed live-data snapshot."
            ),
            "price_sources": str(snapshot_metadata.get("price_sources", "Snapshot source detail unavailable.")),
            "snapshot_created_at": str(snapshot_metadata.get("snapshot_created_at", "unknown")),
            "live_error": str(live_error)[:400],
        },
    )


def load_dashboard_data(start_date: date = DEFAULT_START_DATE) -> DataLoadResult:
    try:
        return build_live_dashboard_data(start_date=start_date)
    except Exception as exc:
        return load_snapshot_dashboard_data(start_date=start_date, live_error=exc)
