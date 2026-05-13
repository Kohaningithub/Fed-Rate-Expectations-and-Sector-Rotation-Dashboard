from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analytics import classify_policy_regime, score_sector_rotation, sector_columns
from src.interpretation import build_dynamic_watch_items, policy_signal_label


def test_classify_policy_regime_labels_cut_expectations():
    index = pd.bdate_range("2024-01-01", periods=5)
    macro = pd.DataFrame(
        {
            "fed_funds_effective": [5.25] * 5,
            "ust_2y": [4.40] * 5,
            "ust_10y": [4.10] * 5,
            "curve_10y_2y": [-0.30] * 5,
            "policy_expectations_gap": [-0.85] * 5,
            "fed_funds_3m_change": [0.00] * 5,
        },
        index=index,
    )

    result = classify_policy_regime(macro)

    assert result["policy_regime"].iloc[-1] == "Cut Expectations"


def test_score_sector_rotation_returns_ranked_rows():
    index = pd.bdate_range("2024-01-01", periods=160)
    prices = pd.DataFrame(
        {
            "XLK": 100 * np.exp(np.linspace(0, 0.20, len(index))),
            "XLF": 100 * np.exp(np.linspace(0, 0.08, len(index))),
            "XLU": 100 * np.exp(np.linspace(0, 0.02, len(index))),
        },
        index=index,
    )

    result = score_sector_rotation(prices, sector_columns(prices))

    assert list(result["Ticker"]) == ["XLK", "XLF", "XLU"]
    assert result["rank"].tolist() == [1, 2, 3]


def test_policy_signal_label_combines_rate_and_curve_signals():
    assert policy_signal_label(-0.75, -0.40) == "easing bias with a restrictive inverted-curve backdrop"


def test_dynamic_watch_items_change_with_policy_and_leadership():
    leaders = pd.DataFrame(
        [
            {
                "Ticker": "XLK",
                "Sector": "Technology",
                "return_3m": 0.18,
                "return_6m": 0.24,
                "rotation_score": 1.60,
            },
            {
                "Ticker": "XLV",
                "Sector": "Health Care",
                "return_3m": 0.07,
                "return_6m": 0.11,
                "rotation_score": 0.65,
            },
        ]
    )
    laggards = pd.DataFrame(
        [
            {
                "Ticker": "XLU",
                "Sector": "Utilities",
                "return_3m": -0.04,
                "return_6m": -0.02,
                "rotation_score": -1.10,
            }
        ]
    )
    return_table = pd.DataFrame(
        {
            "Ticker": ["XLK", "XLV", "XLU"],
            "Sector": ["Technology", "Health Care", "Utilities"],
            "3M": [0.18, 0.07, -0.04],
        }
    )

    easing_questions = build_dynamic_watch_items(
        policy_gap=-0.80,
        curve=-0.35,
        leaders=leaders,
        laggards=laggards,
        return_table=return_table,
    )
    tightening_questions = build_dynamic_watch_items(
        policy_gap=0.80,
        curve=0.90,
        leaders=leaders,
        laggards=laggards,
        return_table=return_table,
    )

    assert easing_questions != tightening_questions
    assert any("deeper easing cycle" in question for question in easing_questions)
    assert any("higher-for-longer" in question for question in tightening_questions)
    assert any("XLK" in question for question in easing_questions)
