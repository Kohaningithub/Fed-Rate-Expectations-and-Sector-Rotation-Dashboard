from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


POLICY_LABELS = {
    "fed_funds_effective": "Effective Fed Funds",
    "fed_target_upper": "Fed Target Upper",
    "ust_2y": "2Y Treasury",
    "ust_10y": "10Y Treasury",
    "sofr": "SOFR",
    "policy_expectations_gap": "2Y Treasury - Fed Funds",
    "curve_10y_2y": "10Y - 2Y Curve",
}


def apply_theme(fig: go.Figure, height: int = 420, show_legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        title=None,
        height=height,
        margin=dict(l=64, r=32, t=28, b=64),
        hovermode="x unified",
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
    )
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def plot_policy_rates(macro: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for column, color in [
        ("fed_funds_effective", "#2f6f5e"),
        ("ust_2y", "#2f4f7f"),
        ("ust_10y", "#b45f06"),
    ]:
        if column in macro.columns:
            fig.add_trace(
                go.Scatter(
                    x=macro.index,
                    y=macro[column],
                    mode="lines",
                    name=POLICY_LABELS[column],
                    line=dict(width=2.4, color=color),
                )
            )
    fig.update_yaxes(title_text="Rate (%)")
    return apply_theme(fig, height=430)


def plot_policy_gap(macro: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for column, color in [
        ("policy_expectations_gap", "#2f6f5e"),
        ("curve_10y_2y", "#7f3f98"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=macro.index,
                y=macro[column],
                mode="lines",
                name=POLICY_LABELS[column],
                line=dict(width=2.2, color=color),
            )
        )
    fig.add_hline(y=0, line_dash="dash", line_color="#777777")
    fig.update_yaxes(title_text="Percentage points")
    return apply_theme(fig, height=430)


def plot_return_heatmap(return_table: pd.DataFrame) -> go.Figure:
    heatmap = return_table.set_index("Sector")[["1M", "3M", "6M", "YTD"]] * 100
    fig = px.imshow(
        heatmap,
        text_auto=".1f",
        aspect="auto",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        labels=dict(color="Return (%)"),
    )
    fig.update_layout(coloraxis_colorbar=dict(title="Return (%)", thickness=12, len=0.72))
    fig.update_xaxes(side="top")
    return apply_theme(fig, height=470, show_legend=False)


def plot_rotation_scatter(rotation: pd.DataFrame, title: str | None = "Sector Rotation: Momentum vs Risk") -> go.Figure:
    frame = rotation.copy()
    min_score = frame["rotation_score"].min()
    frame["score_size"] = (frame["rotation_score"] - min_score + 0.5).clip(lower=0.2)
    fig = px.scatter(
        frame,
        x="return_6m",
        y="volatility_3m",
        size="score_size",
        color="rotation_score",
        text="Ticker",
        hover_name="Sector",
        hover_data={
            "Ticker": True,
            "return_3m": ":.1%",
            "return_6m": ":.1%",
            "volatility_3m": ":.1%",
            "rotation_score": ":.2f",
            "score_size": False,
        },
        labels={
            "return_6m": "6M return",
            "volatility_3m": "3M annualized volatility",
        },
        color_continuous_scale="Tealrose",
    )
    fig.update_traces(
        marker=dict(line=dict(width=1, color="#ffffff"), opacity=0.88),
        textposition="top center",
        textfont=dict(size=11, color="#35423c"),
    )
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(
        title=title,
        showlegend=False,
        coloraxis_colorbar=dict(title="Score", thickness=10, len=0.65, x=1.02),
    )
    fig = apply_theme(fig, height=430, show_legend=False)
    fig.update_layout(
        hovermode="closest",
        margin=dict(l=64, r=54, t=30, b=64),
    )
    return fig


def plot_normalized_prices(indexed_prices: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for column in indexed_prices.columns:
        fig.add_trace(
            go.Scatter(
                x=indexed_prices.index,
                y=indexed_prices[column],
                mode="lines",
                name=column,
                line=dict(width=1.8),
            )
        )
    fig.update_yaxes(title_text="Indexed price, start = 100")
    return apply_theme(fig, height=470)


def plot_regime_bars(regime_summary: pd.DataFrame, regime: str, metric: str) -> go.Figure:
    frame = regime_summary.loc[regime_summary["policy_regime"] == regime].copy()
    frame = frame.sort_values(metric, ascending=False)
    fig = px.bar(
        frame,
        x="Sector",
        y=metric,
        color=metric,
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        labels={
            "annualized_return": "Annualized return",
            "annualized_volatility": "Annualized volatility",
            "sharpe_proxy": "Sharpe proxy",
        },
    )
    if metric != "sharpe_proxy":
        fig.update_yaxes(tickformat=".0%")
    fig.update_layout(coloraxis_colorbar=dict(title="Value", thickness=12, len=0.72))
    return apply_theme(fig, height=470, show_legend=False)
