"""How precise is a Sharpe ratio? Two independent answers.

1. Delta method: the Sharpe ratio mean/std is a smooth
   function of two moments. Its standard error follows from the gradient of
   that function and the (Newey-West) covariance of the moment estimates.
2. Moving-block bootstrap: resample blocks of consecutive months,
   recompute the Sharpe ratio thousands of times, read off the spread.

If the two agree, the uncertainty estimate is trustworthy; if not, the
normal approximation behind the delta method is doubtful.
"""

import numpy as np
from scipy import stats


def newey_west(u: np.ndarray, lags: int) -> np.ndarray:
    """Long-run covariance of the rows of u (T x k, mean zero), Bartlett weights."""
    T = u.shape[0]
    S = u.T @ u / T
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        G = u[lag:].T @ u[:-lag] / T
        S += w * (G + G.T)
    return S


def sharpe_with_cov(X: np.ndarray, periods: int, lags: int):
    """Annualized Sharpe ratios of the columns of X (excess returns) and their
    joint covariance matrix from the delta method.

    theta = (mu_1..mu_n, var_1..var_n), SR_j = mu_j / sqrt(var_j)
    Var(SR_hat) ~= G S G' / T, with G = dSR/dtheta' and S the long-run
    covariance of the moment conditions (x - mu, (x - mu)^2 - var).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    T, n = X.shape
    mu = X.mean(axis=0)
    dev = X - mu
    var = (dev**2).mean(axis=0)
    sd = np.sqrt(var)

    moments = np.hstack([dev, dev**2 - var])  # T x 2n
    S = newey_west(moments, lags)

    G = np.zeros((n, 2 * n))
    G[np.arange(n), np.arange(n)] = 1 / sd  # dSR/dmu
    G[np.arange(n), n + np.arange(n)] = -mu / (2 * sd**3)  # dSR/dvar

    cov = G @ S @ G.T / T
    sr = X.mean(axis=0) / X.std(axis=0, ddof=1)  # same estimator as the tables
    return sr * np.sqrt(periods), cov * periods


def block_bootstrap_indices(T: int, block_len: int, n_boot: int, rng) -> np.ndarray:
    """Circular moving-block bootstrap: n_boot index paths of length T."""
    n_blocks = int(np.ceil(T / block_len))
    starts = rng.integers(0, T, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block_len)) % T
    return idx.reshape(n_boot, -1)[:, :T]


def bootstrap_sharpes(X: np.ndarray, periods: int, block_len: int, n_boot: int, rng):
    """n_boot x n matrix of annualized Sharpe ratios on resampled paths.
    All columns use the same resampled months, so their correlation is kept."""
    X = np.asarray(X, dtype=float)
    idx = block_bootstrap_indices(X.shape[0], block_len, n_boot, rng)
    Xb = X[idx]  # n_boot x T x n
    return Xb.mean(axis=1) / Xb.std(axis=1, ddof=1) * np.sqrt(periods)


def sharpe_inference(excess, base: str, periods: int, lags: int, block_len: int, n_boot: int, rng):
    """Table of Sharpe ratios with delta-method and bootstrap 95% intervals,
    plus the difference of every strategy against `base` (e.g. SPY).

    excess: DataFrame of excess returns, one column per strategy.
    """
    import pandas as pd

    names = list(excess.columns)
    X = excess.to_numpy()
    sr, cov = sharpe_with_cov(X, periods, lags)
    se = np.sqrt(np.diag(cov))
    boot = bootstrap_sharpes(X, periods, block_len, n_boot, rng)
    z = stats.norm.ppf(0.975)
    b = names.index(base)

    rows = {}
    for j, name in enumerate(names):
        row = {
            "Sharpe": sr[j],
            "SE (delta)": se[j],
            "CI low (delta)": sr[j] - z * se[j],
            "CI high (delta)": sr[j] + z * se[j],
            "SE (bootstrap)": boot[:, j].std(ddof=1),
            "CI low (bootstrap)": np.percentile(boot[:, j], 2.5),
            "CI high (bootstrap)": np.percentile(boot[:, j], 97.5),
        }
        if j != b:
            a = np.zeros(len(names))
            a[j], a[b] = 1, -1
            diff = sr[j] - sr[b]
            se_diff = float(np.sqrt(a @ cov @ a))
            d_boot = boot[:, j] - boot[:, b]
            row.update(
                {
                    f"Diff vs {base}": diff,
                    "p-value (delta)": 2 * stats.norm.sf(abs(diff) / se_diff) if se_diff > 0 else np.nan,
                    "Diff CI low (bootstrap)": np.percentile(d_boot, 2.5),
                    "Diff CI high (bootstrap)": np.percentile(d_boot, 97.5),
                }
            )
        rows[name] = row
    return pd.DataFrame(rows).T
