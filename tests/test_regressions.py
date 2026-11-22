import numpy as np
import pandas as pd
import pytest

from etf_analysis.backtest import momentum_signal
from etf_analysis.regressions import (
    fixed_effects_by_dummies,
    momentum_panel,
    momentum_panel_regression,
    style_regressions,
)


def test_demeaning_equals_month_dummies(monthly_returns):
    """Frisch-Waugh-Lovell: within-month demeaning gives the dummy-regression coefficient."""
    panel = momentum_panel(monthly_returns, momentum_signal(monthly_returns, 12, 1))
    res = momentum_panel_regression(panel)
    assert res["Coefficient (monthly)"] == pytest.approx(fixed_effects_by_dummies(panel))


def test_panel_uses_next_month_return(monthly_returns):
    panel = momentum_panel(monthly_returns, momentum_signal(monthly_returns, 12, 1))
    date, sector = panel.index[0]
    nxt = monthly_returns.index[monthly_returns.index.get_loc(date) + 1]
    assert panel.iloc[0]["ret_next"] == monthly_returns.loc[nxt, sector]


def test_style_regression_recovers_known_betas():
    rng = np.random.default_rng(5)
    idx = pd.date_range("2000-01-31", periods=400, freq="ME")
    spy = rng.normal(0.007, 0.045, 400)
    iwm = spy + rng.normal(0, 0.02, 400)
    iwd, iwf = rng.normal(0.006, 0.04, 400), rng.normal(0.008, 0.045, 400)
    rf = pd.Series(0.001, index=idx)
    size, value = iwm - spy, iwd - iwf
    sec = rf.values + 1.2 * (spy - rf.values) + 0.5 * size - 0.3 * value + rng.normal(0, 0.005, 400)
    monthly = pd.DataFrame({"SPY": spy, "IWM": iwm, "IWD": iwd, "IWF": iwf, "XLK": sec}, index=idx)
    out = style_regressions(monthly, rf, {"XLK": "Tech"})
    assert out.loc["XLK", "Beta MKT"] == pytest.approx(1.2, abs=0.03)
    assert out.loc["XLK", "Beta SIZE"] == pytest.approx(0.5, abs=0.05)
    assert out.loc["XLK", "Beta VALUE"] == pytest.approx(-0.3, abs=0.03)
    assert out.loc["XLK", "p(no style tilt)"] < 0.001
