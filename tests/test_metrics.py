import numpy as np
import pandas as pd
import pytest

from etf_analysis.metrics import cagr, max_drawdown


def test_max_drawdown_known_path():
    # wealth: 1.0 -> 1.1 -> 0.55 -> 0.605 ; worst fall is 1.1 -> 0.55 = -50%
    r = pd.Series([0.10, -0.50, 0.10])
    assert max_drawdown(r) == pytest.approx(-0.5)


def test_no_drawdown_when_always_up():
    assert max_drawdown(pd.Series([0.01, 0.02, 0.03])) == 0


def test_cagr_constant_return():
    r = pd.Series([0.01] * 24)  # 1% a month for two years
    assert cagr(r, 12) == pytest.approx(1.01**12 - 1)


def test_cagr_matches_final_wealth():
    r = pd.Series(np.random.default_rng(0).normal(0.01, 0.05, 36))
    g = cagr(r, 12)
    assert (1 + g) ** 3 == pytest.approx((1 + r).prod())
