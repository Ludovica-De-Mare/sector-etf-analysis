"""Performance and risk statistics for a single return series."""

import numpy as np
import pandas as pd


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Percentage below the previous peak of cumulative wealth, at each date."""
    wealth = (1 + returns).cumprod()
    return wealth / wealth.cummax() - 1


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough fall of cumulative wealth (a negative number)."""
    return float(drawdown_series(returns).min())


def cagr(returns: pd.Series, periods: int) -> float:
    """Compound annual growth rate: the constant yearly return giving the same final wealth."""
    return float((1 + returns).prod() ** (periods / len(returns)) - 1)


def perf_stats(r, rf, bench, periods) -> dict:
    """Annualized summary statistics. periods = 252 for daily, 12 for monthly data."""
    excess = r - rf
    active = r - bench
    cov = np.cov(r, bench)
    te = active.std() * np.sqrt(periods)
    return {
        "CAGR": cagr(r, periods),
        "Volatility": r.std() * np.sqrt(periods),
        "Sharpe": excess.mean() / excess.std() * np.sqrt(periods),
        "Max drawdown": max_drawdown(r),
        "Beta vs SPY": cov[0, 1] / cov[1, 1],
        "Corr vs SPY": np.corrcoef(r, bench)[0, 1],
        "Tracking error": te,
        "Info ratio": active.mean() * periods / te if te > 0 else np.nan,
    }
