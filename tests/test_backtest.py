import numpy as np
import pandas as pd
import pytest

from etf_analysis.backtest import momentum_signal, run_backtest, top_n_weights


def test_signal_is_12_minus_1_product(monthly_returns):
    sig = momentum_signal(monthly_returns, 12, 1)
    t = 20
    manual = (1 + monthly_returns.iloc[t - 11 : t]).prod() - 1  # months t-11 ... t-1
    np.testing.assert_allclose(sig.iloc[t].values, manual.values)


def test_no_look_ahead(monthly_returns):
    """Changing the return of month t must not change the signal known at t."""
    sig = momentum_signal(monthly_returns, 12, 1)
    shocked = monthly_returns.copy()
    shocked.iloc[50] += 0.5
    sig2 = momentum_signal(shocked, 12, 1)
    pd.testing.assert_frame_equal(sig.iloc[:51], sig2.iloc[:51])


@pytest.mark.parametrize("n", [1, 3, 5])
def test_weights_sum_to_one_and_pick_top(monthly_returns, n):
    sig = momentum_signal(monthly_returns, 12, 1).dropna()
    w = top_n_weights(sig, n)
    np.testing.assert_allclose(w.sum(axis=1), 1.0)
    row = sig.iloc[0]
    assert set(w.columns[w.iloc[0] > 0]) == set(row.nlargest(n).index)


def test_buy_and_hold_one_asset_has_no_cost(monthly_returns):
    w = pd.DataFrame(0.0, index=monthly_returns.index, columns=monthly_returns.columns)
    w["S0"] = 1.0
    net, to = run_backtest(w, monthly_returns, cost_bps=10)
    np.testing.assert_allclose(net.values, monthly_returns["S0"].iloc[1:].values)
    assert to.abs().max() < 1e-12


def test_cost_equals_bps_times_turnover(monthly_returns):
    w = pd.DataFrame(1 / 9, index=monthly_returns.index, columns=monthly_returns.columns)
    gross, _ = run_backtest(w, monthly_returns, cost_bps=0)
    net, to = run_backtest(w, monthly_returns, cost_bps=25)
    np.testing.assert_allclose(gross - net, 0.0025 * to)
