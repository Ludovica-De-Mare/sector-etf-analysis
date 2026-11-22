import numpy as np
import pandas as pd
import pytest

from etf_analysis.inference import (
    block_bootstrap_indices,
    bootstrap_sharpes,
    sharpe_inference,
    sharpe_with_cov,
)


def test_delta_se_matches_lo_formula_for_iid_normal():
    """For iid normal returns the delta-method SE of the (per-period) Sharpe ratio
    is sqrt((1 + SR^2 / 2) / T)  (Lo, 2002)."""
    rng = np.random.default_rng(1)
    T, sr_true = 200_000, 0.2
    x = rng.normal(sr_true, 1.0, T)
    sr, cov = sharpe_with_cov(x, periods=1, lags=0)
    assert np.sqrt(cov[0, 0]) == pytest.approx(np.sqrt((1 + sr[0] ** 2 / 2) / T), rel=0.02)


def test_delta_and_bootstrap_agree():
    rng = np.random.default_rng(2)
    x = rng.normal(0.01, 0.04, (600, 2))
    _, cov = sharpe_with_cov(x, periods=12, lags=3)
    boot = bootstrap_sharpes(x, 12, block_len=6, n_boot=4000, rng=rng)
    np.testing.assert_allclose(boot.std(axis=0), np.sqrt(np.diag(cov)), rtol=0.15)


def test_bootstrap_indices_are_consecutive_blocks():
    idx = block_bootstrap_indices(T=50, block_len=5, n_boot=3, rng=np.random.default_rng(0))
    assert idx.shape == (3, 50)
    first_block = idx[0, :5]
    assert np.all(np.diff(first_block) % 50 == 1)


def test_seed_makes_bootstrap_reproducible():
    x = np.random.default_rng(3).normal(size=(100, 2))
    a = bootstrap_sharpes(x, 12, 6, 200, np.random.default_rng(42))
    b = bootstrap_sharpes(x, 12, 6, 200, np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)


def test_difference_with_itself_is_zero():
    rng = np.random.default_rng(4)
    x = rng.normal(0.01, 0.04, 300)
    df = pd.DataFrame({"A": x, "B": x})
    out = sharpe_inference(df, "A", 12, 3, 6, 500, rng)
    assert out.loc["B", "Diff vs A"] == pytest.approx(0)
