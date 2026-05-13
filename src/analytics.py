from __future__ import annotations

import numpy as np
import pandas as pd

from .data import SECTOR_ETFS


TRADING_WINDOWS = {
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "12M": 252,
}


def sector_columns(prices: pd.DataFrame) -> list[str]:
    return [ticker for ticker in SECTOR_ETFS if ticker in prices.columns]


def zscore(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def classify_policy_regime(macro: pd.DataFrame) -> pd.DataFrame:
    frame = macro.copy()
    conditions = [
        frame["policy_expectations_gap"] > 0.50,
        frame["policy_expectations_gap"] < -0.50,
        frame["curve_10y_2y"] < -0.25,
        frame["fed_funds_3m_change"] > 0.25,
    ]
    labels = [
        "Hike Expectations",
        "Cut Expectations",
        "Restrictive / Inverted Curve",
        "Recent Tightening",
    ]
    frame["policy_regime"] = np.select(conditions, labels, default="Neutral / Transition")
    return frame


def latest_policy_read(macro: pd.DataFrame) -> dict[str, str | float]:
    latest = macro.dropna(subset=["fed_funds_effective", "ust_2y", "ust_10y"]).iloc[-1]
    gap = float(latest["policy_expectations_gap"])
    curve = float(latest["curve_10y_2y"])

    if gap <= -0.50:
        bias = "Market rates imply an easing bias versus the current policy rate."
    elif gap >= 0.50:
        bias = "Market rates imply a higher-for-longer or additional tightening bias."
    else:
        bias = "Market rates are close to the current policy rate."

    if curve < -0.25:
        curve_note = "The 10Y-2Y curve is inverted, which often coincides with restrictive policy conditions."
    elif curve > 0.75:
        curve_note = "The 10Y-2Y curve is positively sloped, often consistent with easier future conditions."
    else:
        curve_note = "The 10Y-2Y curve is close to flat."

    return {
        "as_of": latest.name.strftime("%Y-%m-%d"),
        "fed_funds": float(latest["fed_funds_effective"]),
        "two_year": float(latest["ust_2y"]),
        "ten_year": float(latest["ust_10y"]),
        "policy_gap": gap,
        "curve": curve,
        "bias": bias,
        "curve_note": curve_note,
    }


def build_return_table(prices: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    clean_prices = prices[columns].dropna(how="all").ffill()
    rows = []
    latest_date = clean_prices.dropna(how="all").index.max()

    for ticker in columns:
        series = clean_prices[ticker].dropna()
        if series.empty:
            continue
        row = {
            "Ticker": ticker,
            "Sector": SECTOR_ETFS.get(ticker, ticker),
        }
        for label, window in TRADING_WINDOWS.items():
            row[label] = series.pct_change(window).iloc[-1]

        year_start = series[series.index.year == latest_date.year]
        row["YTD"] = (series.iloc[-1] / year_start.iloc[0]) - 1 if not year_start.empty else np.nan
        rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values("3M", ascending=False).reset_index(drop=True)


def score_sector_rotation(prices: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    clean_prices = prices[columns].dropna(how="all").ffill()
    returns = clean_prices.pct_change()
    rows = []

    for ticker in columns:
        series = clean_prices[ticker].dropna()
        if len(series) < 130:
            continue
        daily = returns[ticker].dropna()
        trailing = series.tail(126)
        drawdown = trailing / trailing.cummax() - 1
        rows.append(
            {
                "Ticker": ticker,
                "Sector": SECTOR_ETFS.get(ticker, ticker),
                "return_1m": series.pct_change(21).iloc[-1],
                "return_3m": series.pct_change(63).iloc[-1],
                "return_6m": series.pct_change(126).iloc[-1],
                "volatility_3m": daily.tail(63).std() * np.sqrt(252),
                "max_drawdown_6m": drawdown.min(),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    frame["rotation_score"] = (
        0.45 * zscore(frame["return_3m"])
        + 0.35 * zscore(frame["return_6m"])
        - 0.15 * zscore(frame["volatility_3m"])
        - 0.05 * zscore(frame["max_drawdown_6m"].abs())
    )
    frame["rank"] = frame["rotation_score"].rank(ascending=False, method="dense").astype(int)
    return frame.sort_values("rotation_score", ascending=False).reset_index(drop=True)


def regime_performance(macro: pd.DataFrame, prices: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if "policy_regime" not in macro.columns:
        macro = classify_policy_regime(macro)

    returns = prices[columns].pct_change().dropna(how="all")
    regimes = macro[["policy_regime"]].reindex(returns.index).ffill()
    joined = returns.join(regimes).dropna(subset=["policy_regime"])
    long = joined.melt(id_vars="policy_regime", var_name="Ticker", value_name="daily_return").dropna()
    if long.empty:
        return long

    grouped = long.groupby(["policy_regime", "Ticker"])["daily_return"]
    summary = grouped.agg(["mean", "std", "count"]).reset_index()
    summary["Sector"] = summary["Ticker"].map(SECTOR_ETFS)
    summary["annualized_return"] = summary["mean"] * 252
    summary["annualized_volatility"] = summary["std"] * np.sqrt(252)
    summary["sharpe_proxy"] = summary["annualized_return"] / summary["annualized_volatility"]
    return summary.replace([np.inf, -np.inf], np.nan)


def normalize_prices(prices: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = prices[columns].dropna(how="all").ffill().dropna()
    if frame.empty:
        return frame
    return frame / frame.iloc[0] * 100
