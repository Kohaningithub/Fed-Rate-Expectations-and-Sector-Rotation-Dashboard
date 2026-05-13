from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from src.analytics import (
    build_return_table,
    classify_policy_regime,
    latest_policy_read,
    normalize_prices,
    regime_performance,
    score_sector_rotation,
    sector_columns,
)
from src.charts import (
    plot_normalized_prices,
    plot_policy_gap,
    plot_policy_rates,
    plot_regime_bars,
    plot_return_heatmap,
    plot_rotation_scatter,
)
from src.data import DEFAULT_START_DATE, SECTOR_ETFS, load_dashboard_data
from src.interpretation import build_rule_based_interpretation


st.set_page_config(
    page_title="Fed Rate Expectations and Sector Rotation",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.7rem; padding-bottom: 2.5rem;}
      [data-testid="stMetricValue"] {font-size: 1.45rem;}
      .small-note {font-size: 0.88rem; color: #5c665f;}
      .info-strip {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
        margin: 1rem 0 1.15rem;
      }
      .info-tile {
        border: 1px solid #dfe6df;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        background: #ffffff;
        min-height: 112px;
      }
      .info-tile strong {
        display: block;
        margin-bottom: 0.35rem;
        color: #20342e;
      }
      .info-tile span {
        color: #5c665f;
        font-size: 0.9rem;
        line-height: 1.45;
      }
      .executive-card {
        border: 1px solid #dfe6df;
        border-radius: 8px;
        padding: 1rem 1.1rem;
        background: #ffffff;
      }
      .ai-note {
        border-left: 4px solid #2f6f5e;
        padding: 0.85rem 1rem;
        background: #f4faf6;
        color: #2c3f38;
        margin: 0.75rem 0 1rem;
      }
      @media (max-width: 900px) {
        .info-strip {grid-template-columns: 1fr;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.1%}"


def rate(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.2f}%"


def point_gap(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:+.2f} pp"


@st.cache_data(ttl=60 * 60)
def get_data(start_date: date):
    return load_dashboard_data(start_date=start_date)


with st.sidebar:
    st.header("Controls")
    start_date = st.date_input(
        "Start date",
        value=DEFAULT_START_DATE,
        min_value=date(2005, 1, 1),
        max_value=date.today(),
    )
    view = st.radio(
        "Dashboard view",
        options=[
            "Executive View",
            "Policy Signals",
            "Sector Rotation",
            "Regime Analytics",
            "Market Interpretation",
            "Methodology",
        ],
        index=0,
    )

try:
    data = get_data(start_date)
except Exception as exc:
    st.title("Fed Rate Expectations and Sector Rotation Dashboard")
    st.error("Live data connection failed. Please refresh the page or try again later.")
    st.info(
        "This dashboard depends on FRED public CSV endpoints for rates and Yahoo Finance chart data for ETF prices."
    )
    st.caption(f"Technical detail: {exc}")
    st.stop()

macro = classify_policy_regime(data.macro)
columns = sector_columns(data.prices)

with st.sidebar:
    st.success("Live FRED/Yahoo data active.")
    st.caption(f"Macro through {macro.index.max().date()}")
    st.caption(f"ETF prices through {data.prices.index.max().date()}")
    if data.metadata.get("warning"):
        st.warning(data.metadata["warning"])
    selected_sectors = st.multiselect(
        "Sectors",
        options=columns,
        default=columns,
        format_func=lambda ticker: f"{ticker} - {SECTOR_ETFS[ticker]}",
    )

if not selected_sectors:
    st.warning("Select at least one sector to display the dashboard.")
    st.stop()

policy_read = latest_policy_read(macro)
return_table = build_return_table(data.prices, selected_sectors)
rotation = score_sector_rotation(data.prices, selected_sectors)
regime_summary = regime_performance(macro, data.prices, selected_sectors)

if return_table.empty or rotation.empty:
    st.warning("Not enough price history for the selected start date. Choose an earlier start date.")
    st.stop()

st.title("Fed Rate Expectations and Sector Rotation Dashboard")
st.caption(
    "A market research dashboard linking policy-rate expectations, yield-curve conditions, "
    "and US sector ETF performance. This is an analytical tool, not investment advice."
)
st.markdown(
    f"""
    <div class="info-strip">
      <div class="info-tile">
        <strong>Purpose</strong>
        <span>Help investors and research teams connect Fed-rate expectations with current US sector leadership.</span>
      </div>
      <div class="info-tile">
        <strong>Live Data</strong>
        <span>Rates from FRED through {macro.index.max().date()}. ETF prices from Yahoo Finance through {data.prices.index.max().date()}.</span>
      </div>
      <div class="info-tile">
        <strong>Decision Lens</strong>
        <span>Use the policy gap, yield curve, and sector rotation score to frame market regime discussions.</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_cols = st.columns(4)
metric_cols[0].metric("Effective Fed Funds", rate(policy_read["fed_funds"]), help="FRED DFF series.")
metric_cols[1].metric("2Y - Fed Funds Gap", point_gap(policy_read["policy_gap"]))
metric_cols[2].metric("10Y - 2Y Curve", point_gap(policy_read["curve"]))
if not rotation.empty:
    leader = rotation.iloc[0]
    metric_cols[3].metric("Top Rotation Sector", leader["Ticker"], delta=pct(leader["return_3m"]))
else:
    metric_cols[3].metric("Top Rotation Sector", "n/a")

st.markdown("---")

if view == "Executive View":
    left, right = st.columns([1.05, 1])
    with left:
        st.subheader("Current policy read")
        st.markdown(
            f"""
            <div class="executive-card">
              <strong>As of {policy_read["as_of"]}</strong><br>
              {policy_read["bias"]}<br><br>
              {policy_read["curve_note"]}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        st.subheader("Rotation ranking")
        st.dataframe(
            rotation[
                [
                    "rank",
                    "Ticker",
                    "Sector",
                    "return_1m",
                    "return_3m",
                    "return_6m",
                    "volatility_3m",
                    "rotation_score",
                ]
            ].style.format(
                {
                    "return_1m": "{:.1%}",
                    "return_3m": "{:.1%}",
                    "return_6m": "{:.1%}",
                    "volatility_3m": "{:.1%}",
                    "rotation_score": "{:.2f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
    with right:
        st.subheader("Sector rotation map")
        st.plotly_chart(plot_rotation_scatter(rotation, title=None), width="stretch")

elif view == "Policy Signals":
    st.subheader("Rates and expectations proxy")
    st.markdown("**Policy rate and Treasury yields**")
    st.plotly_chart(plot_policy_rates(macro), width="stretch")
    st.markdown("**Rate expectations proxy and yield-curve slope**")
    st.plotly_chart(plot_policy_gap(macro), width="stretch")
    st.markdown(
        """
        <p class="small-note">
        The 2Y Treasury minus effective Fed Funds is used as a transparent proxy for market-implied
        policy direction. A negative gap suggests the market expects easier policy over the policy-sensitive
        horizon, while a positive gap suggests tighter or higher-for-longer conditions.
        </p>
        """,
        unsafe_allow_html=True,
    )

elif view == "Sector Rotation":
    st.subheader("Sector return heatmap")
    st.plotly_chart(plot_return_heatmap(return_table), width="stretch")
    st.subheader("Normalized sector ETF price paths")
    indexed = normalize_prices(data.prices, selected_sectors)
    st.plotly_chart(plot_normalized_prices(indexed), width="stretch")

elif view == "Regime Analytics":
    regimes = list(regime_summary["policy_regime"].dropna().unique())
    if not regimes:
        st.warning("Not enough data to calculate regime performance.")
    else:
        control_cols = st.columns([1, 1, 2])
        selected_regime = control_cols[0].selectbox("Policy regime", options=regimes)
        selected_metric = control_cols[1].selectbox(
            "Metric",
            options=["annualized_return", "annualized_volatility", "sharpe_proxy"],
            format_func=lambda value: value.replace("_", " ").title(),
        )
        st.subheader(f"Sector performance during {selected_regime}")
        st.plotly_chart(
            plot_regime_bars(regime_summary, selected_regime, selected_metric),
            width="stretch",
        )
        table = regime_summary.loc[regime_summary["policy_regime"] == selected_regime]
        st.dataframe(
            table[
                [
                    "policy_regime",
                    "Ticker",
                    "Sector",
                    "annualized_return",
                    "annualized_volatility",
                    "sharpe_proxy",
                    "count",
                ]
            ].style.format(
                {
                    "annualized_return": "{:.1%}",
                    "annualized_volatility": "{:.1%}",
                    "sharpe_proxy": "{:.2f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

elif view == "Market Interpretation":
    st.subheader("Market interpretation")
    st.markdown(
        """
        <div class="ai-note">
          This interpretation is generated locally from transparent dashboard rules. It uses the current
          policy signals, sector rotation ranking, return table, and regime analytics without calling any
          external AI API. It is not investment advice.
        </div>
        """,
        unsafe_allow_html=True,
    )
    interpretation = build_rule_based_interpretation(
        policy_read=policy_read,
        rotation=rotation,
        return_table=return_table,
        regime_summary=regime_summary,
        macro_through=str(macro.index.max().date()),
        prices_through=str(data.prices.index.max().date()),
    )
    st.markdown(f"### {interpretation['headline']}")

    section_labels = [
        ("Executive read", "executive_read"),
        ("Policy signal", "policy_signal"),
        ("Sector rotation", "sector_rotation"),
        ("Questions to monitor", "watch_items"),
    ]
    for label, key in section_labels:
        st.markdown(f"**{label}**")
        for item in interpretation[key]:
            st.markdown(f"- {item}")

    st.caption(str(interpretation["data_note"]))

elif view == "Methodology":
    st.subheader("Methodology")
    st.markdown(
        """
        **Research question:** How do policy-rate expectations and yield-curve conditions relate to
        short-term US sector leadership?

        **Data inputs**

        - FRED: Effective Fed Funds, Fed target upper bound, 2Y Treasury, 10Y Treasury, and SOFR.
        - Yahoo Finance chart API: Daily adjusted ETF price histories for SPY and SPDR sector ETFs.

        **Core metrics**

        - Policy expectations proxy: `2Y Treasury yield - Effective Fed Funds`.
        - Curve slope: `10Y Treasury yield - 2Y Treasury yield`.
        - Sector rotation score: weighted blend of 3M return, 6M return, 3M volatility, and 6M drawdown.
        - Regime performance: annualized sector returns and volatility grouped by policy-rate regime.
        - Market interpretation: local rule-based narrative generated from the displayed dashboard metrics.

        **Client use case**

        This dashboard is designed as an investment research screen: it does not make a trading call by
        itself, but it helps an analyst explain sector leadership in the context of changing monetary policy.
        """
    )
    st.caption(f"Data source mode: {data.metadata.get('source', 'Unknown')}")
