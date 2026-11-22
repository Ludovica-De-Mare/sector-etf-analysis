"""Sector momentum signal, monthly backtest engine and robustness grid."""

import numpy as np
import pandas as pd

from .metrics import perf_stats


def momentum_signal(sector_ret: pd.DataFrame, lookback: int, skip: int) -> pd.DataFrame:
    """Cumulative return over months t-lookback+1 ... t-skip, known at the end of month t.

    With lookback=12, skip=1 this is the standard "12-1" momentum. Computed with
    log returns so the rolling product becomes a vectorized rolling sum.
    """
    log_r = np.log1p(sector_ret).shift(skip)
    return np.expm1(log_r.rolling(lookback - skip).sum())


def top_n_weights(signal: pd.DataFrame, n: int) -> pd.DataFrame:
    """Equal weight 1/n in the n sectors with the highest signal each month."""
    ranks = signal.rank(axis=1, ascending=False, method="first")
    return (ranks <= n).astype(float) / n


def run_backtest(target_weights: pd.DataFrame, returns: pd.DataFrame, cost_bps: float):
    """Monthly backtest without look-ahead.

    Weights chosen at the end of month t (row t) earn the returns of month t+1.
    Between rebalances weights drift with prices; the cost is charged on
    turnover = sum |new target - drifted weights|.
    Returns (net monthly returns, turnover), both indexed by the earning month.
    """
    cost = cost_bps / 10_000
    W = target_weights.to_numpy()
    R = returns.loc[target_weights.index].to_numpy()
    held = W[0]
    net, turnover = [], []
    for t in range(1, len(W)):
        gross = float(held @ R[t])
        drifted = held * (1 + R[t]) / (1 + gross)
        to = float(np.abs(W[t] - drifted).sum())
        net.append(gross - cost * to)
        turnover.append(to)
        held = W[t]
    idx = target_weights.index[1:]
    return pd.Series(net, index=idx), pd.Series(turnover, index=idx)


def robustness_grid(sector_ret, rf, lookbacks, top_ns, skip, cost_bps, start):
    """Sharpe ratio and CAGR of the momentum strategy for every (lookback, top_n).

    All cells use the same months (from `start`), so they are comparable.
    """
    out = []
    for lb in lookbacks:
        signal = momentum_signal(sector_ret, lb, skip).loc[start:]
        for n in top_ns:
            r, to = run_backtest(top_n_weights(signal, n), sector_ret, cost_bps)
            s = perf_stats(r, rf.loc[r.index], r, 12)  # benchmark irrelevant here
            out.append(
                {
                    "Lookback": lb,
                    "Top N": n,
                    "Sharpe": s["Sharpe"],
                    "CAGR": s["CAGR"],
                    "Annual turnover": to.mean() * 12,
                }
            )
    return pd.DataFrame(out)
