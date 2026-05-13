from __future__ import annotations

import pandas as pd


def _pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.1%}"


def _pp(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:+.2f} pp"


def _sector_phrase(row: pd.Series) -> str:
    return (
        f"{row['Ticker']} ({row['Sector']}): 3M {_pct(row.get('return_3m'))}, "
        f"6M {_pct(row.get('return_6m'))}, score {row.get('rotation_score', 0):.2f}"
    )


def policy_signal_label(policy_gap: float, curve: float) -> str:
    if policy_gap <= -0.50:
        rate_signal = "easing bias"
    elif policy_gap >= 0.50:
        rate_signal = "higher-for-longer bias"
    else:
        rate_signal = "near-neutral policy pricing"

    if curve < -0.25:
        curve_signal = "restrictive inverted-curve backdrop"
    elif curve > 0.75:
        curve_signal = "positively sloped curve backdrop"
    else:
        curve_signal = "flat-to-moderate curve backdrop"

    return f"{rate_signal} with a {curve_signal}"


def build_dynamic_watch_items(
    *,
    policy_gap: float,
    curve: float,
    leaders: pd.DataFrame,
    laggards: pd.DataFrame,
    return_table: pd.DataFrame,
) -> list[str]:
    questions: list[str] = []

    top = leaders.iloc[0] if not leaders.empty else None
    second = leaders.iloc[1] if len(leaders) > 1 else None
    bottom = laggards.iloc[0] if not laggards.empty else None

    if policy_gap <= -0.50:
        questions.append(
            f"The 2Y - Fed Funds gap is {_pp(policy_gap)}. Is the market moving toward a deeper easing cycle, "
            "or is this mainly a short-term growth scare?"
        )
    elif policy_gap >= 0.50:
        questions.append(
            f"The 2Y - Fed Funds gap is {_pp(policy_gap)}. Are sector winners resilient to higher-for-longer rates, "
            "or is leadership vulnerable to a financing-cost shock?"
        )
    else:
        questions.append(
            f"The 2Y - Fed Funds gap is {_pp(policy_gap)}. If policy pricing is near neutral, what non-rate driver "
            "is explaining current sector leadership?"
        )

    if curve < -0.25:
        questions.append(
            f"The 10Y - 2Y curve is {_pp(curve)}. Does sector leadership confirm late-cycle caution, or is the market "
            "rotating toward growth despite an inverted curve?"
        )
    elif curve > 0.75:
        questions.append(
            f"The 10Y - 2Y curve is {_pp(curve)}. Is steepening being driven by healthier growth expectations or by "
            "higher long-end inflation and term-premium risk?"
        )
    else:
        questions.append(
            f"The 10Y - 2Y curve is {_pp(curve)}. What would push the curve out of this range, and which sectors would "
            "be most exposed?"
        )

    if top is not None and second is not None:
        score_gap = float(top["rotation_score"] - second["rotation_score"])
        if score_gap >= 0.75:
            questions.append(
                f"{top['Ticker']} ({top['Sector']}) has a clear rotation-score lead over {second['Ticker']}. "
                "Is this broad sector confirmation or a narrow leadership signal?"
            )
        else:
            questions.append(
                f"Leadership between {top['Ticker']} ({top['Sector']}) and {second['Ticker']} ({second['Sector']}) is close. "
                "Which incoming data point would break the tie?"
            )

    if top is not None:
        top_sector = str(top["Sector"])
        if top_sector in {"Technology", "Communication Services", "Consumer Discretionary", "Real Estate"}:
            questions.append(
                f"{top['Ticker']} ({top_sector}) is leading. Is the market rewarding duration-sensitive growth, "
                "and would that leadership survive a rebound in yields?"
            )
        elif top_sector in {"Consumer Staples", "Health Care", "Utilities"}:
            questions.append(
                f"{top['Ticker']} ({top_sector}) is leading. Is this defensive positioning, or a sector-specific "
                "earnings and valuation story?"
            )
        elif top_sector in {"Energy", "Materials", "Industrials"}:
            questions.append(
                f"{top['Ticker']} ({top_sector}) is leading. Is cyclicality being supported by real demand, commodities, "
                "or policy-sensitive capital spending?"
            )
        elif top_sector == "Financials":
            questions.append(
                "Financials are near the top of the rotation screen. Are curve dynamics, credit quality, and deposit costs "
                "supporting that signal?"
            )

    if bottom is not None:
        questions.append(
            f"{bottom['Ticker']} ({bottom['Sector']}) is one of the weakest rotation signals. Is the weakness idiosyncratic, "
            "or does it challenge the broader market narrative?"
        )

    if not return_table.empty:
        positive_count = int((return_table["3M"] > 0).sum())
        breadth = positive_count / len(return_table)
        if breadth >= 0.75:
            questions.append(
                f"{positive_count} of {len(return_table)} selected sectors have positive 3M returns. Is broad participation "
                "confirming the rotation, or are correlations masking sector-specific risk?"
            )
        elif breadth <= 0.35:
            questions.append(
                f"Only {positive_count} of {len(return_table)} selected sectors have positive 3M returns. Is leadership too narrow "
                "to support a durable market regime?"
            )
        else:
            questions.append(
                f"{positive_count} of {len(return_table)} selected sectors have positive 3M returns. Which sectors need to improve "
                "for the rotation to broaden?"
            )

    return questions[:6]


def build_rule_based_interpretation(
    *,
    policy_read: dict[str, str | float],
    rotation: pd.DataFrame,
    return_table: pd.DataFrame,
    regime_summary: pd.DataFrame,
    macro_through: str,
    prices_through: str,
) -> dict[str, list[str] | str]:
    leaders = rotation.sort_values("rotation_score", ascending=False).head(3)
    laggards = rotation.sort_values("rotation_score", ascending=True).head(3)
    returns = return_table.sort_values("3M", ascending=False)
    top_3m = returns.head(3)
    bottom_3m = returns.tail(3).sort_values("3M")

    policy_gap = float(policy_read["policy_gap"])
    curve = float(policy_read["curve"])
    signal = policy_signal_label(policy_gap, curve)

    if not leaders.empty:
        top_leader = leaders.iloc[0]
        leadership_note = (
            f"{top_leader['Ticker']} ({top_leader['Sector']}) currently has the strongest rotation score, "
            f"with a 3M return of {_pct(top_leader.get('return_3m'))}."
        )
    else:
        leadership_note = "No sector leadership signal is available for the selected universe."

    if policy_gap < -0.50 and not leaders.empty:
        regime_note = (
            "The policy gap is negative enough to suggest easing expectations. In this setting, leadership in "
            "rate-sensitive or growth-oriented sectors may indicate the market is looking through current policy restraint."
        )
    elif policy_gap > 0.50 and not leaders.empty:
        regime_note = (
            "The policy gap is positive enough to suggest higher-for-longer pricing. Sector leadership should be checked "
            "against balance-sheet sensitivity, financing costs, and earnings durability."
        )
    else:
        regime_note = (
            "Policy pricing is close to the current Fed Funds rate. Sector leadership may be driven more by earnings, "
            "valuation, or idiosyncratic industry factors than by a single rates narrative."
        )

    if curve < -0.25:
        curve_note = (
            "The curve is inverted, so the dashboard should be read with late-cycle risk in mind. Defensive leadership "
            "or narrow growth leadership would both be important signals to monitor."
        )
    else:
        curve_note = (
            "The curve is not deeply inverted. That reduces one recession-signal pressure point, but it does not remove "
            "the need to monitor growth, inflation, and earnings revisions."
        )

    return {
        "headline": f"As of {policy_read['as_of']}, the dashboard shows {signal}.",
        "executive_read": [
            leadership_note,
            regime_note,
            curve_note,
        ],
        "policy_signal": [
            f"Effective Fed Funds is {policy_read['fed_funds']:.2f}%, while the 2Y Treasury is {policy_read['two_year']:.2f}%.",
            f"The 2Y - Fed Funds gap is {_pp(policy_gap)}, and the 10Y - 2Y curve is {_pp(curve)}.",
            str(policy_read["bias"]),
        ],
        "sector_rotation": [
            "Rotation leaders: " + "; ".join(_sector_phrase(row) for _, row in leaders.iterrows()),
            "Rotation laggards: " + "; ".join(_sector_phrase(row) for _, row in laggards.iterrows()),
            "3M return leaders: "
            + "; ".join(
                f"{row['Ticker']} ({row['Sector']}): {_pct(row.get('3M'))}" for _, row in top_3m.iterrows()
            ),
            "3M return laggards: "
            + "; ".join(
                f"{row['Ticker']} ({row['Sector']}): {_pct(row.get('3M'))}" for _, row in bottom_3m.iterrows()
            ),
        ],
        "watch_items": build_dynamic_watch_items(
            policy_gap=policy_gap,
            curve=curve,
            leaders=leaders,
            laggards=laggards,
            return_table=return_table,
        ),
        "data_note": f"Rates through {macro_through}; ETF prices through {prices_through}. This is a research interpretation, not investment advice.",
    }
