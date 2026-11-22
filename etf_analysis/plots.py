"""All charts. Each function saves one PNG into the output folder."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from .metrics import drawdown_series

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})
PCT = PercentFormatter(1.0)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def risk_return(etf_stats, cfg, path):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for t in etf_stats.index:
        x, y = etf_stats.loc[t, "Volatility"], etf_stats.loc[t, "CAGR"]
        color = "black" if t == cfg.benchmark else ("tab:blue" if t in cfg.sectors else "tab:orange")
        ax.scatter(x, y, color=color, s=40)
        ax.annotate(t, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set(xlabel="Annualized volatility", ylabel="Annualized return (CAGR)")
    ax.xaxis.set_major_formatter(PCT)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_title("Risk and return: sectors (blue), style ETFs (orange), SPY (black)")
    _save(fig, path)


def correlations(corr, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45)
    ax.set_yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation of monthly returns")
    ax.grid(False)
    _save(fig, path)


def style_exposures(style, cfg, path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    style[["Beta MKT", "Beta SIZE", "Beta VALUE"]].plot.bar(ax=ax)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticklabels([f"{t}\n{cfg.sectors[t]}" for t in style.index], rotation=0, fontsize=7)
    ax.set_title("Sector style exposures (monthly regression on ETF factor proxies)")
    _save(fig, path)


def rolling_volatility(daily, cfg, path, tickers=("XLK", "XLU", "XLE")):
    cols = [cfg.benchmark, *[t for t in tickers if t in daily]]
    vol = daily[cols].rolling(252).std() * np.sqrt(252)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    vol.plot(ax=ax, lw=1)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_title("Rolling 1-year annualized volatility")
    _save(fig, path)


def backtest(strategies, cost_bps, path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    (1 + strategies).cumprod().plot(ax=axes[0], lw=1.3)
    axes[0].set(title=f"Backtest: growth of $1 (net of {cost_bps:g} bps costs)", ylabel="Wealth")
    strategies.apply(drawdown_series).plot(ax=axes[1], lw=1, legend=False)
    axes[1].yaxis.set_major_formatter(PCT)
    axes[1].set_ylabel("Drawdown")
    _save(fig, path)


def sharpe_intervals(inf, path):
    """Point estimate with delta-method and bootstrap 95% intervals, side by side."""
    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(inf))
    for lab, off, col in (
        ("delta", -0.12, "tab:blue"),
        ("bootstrap", 0.12, "tab:orange"),
    ):
        lo, hi = inf[f"CI low ({lab})"], inf[f"CI high ({lab})"]
        ax.errorbar(
            inf["Sharpe"],
            y + off,
            xerr=[inf["Sharpe"] - lo, hi - inf["Sharpe"]],
            fmt="o",
            color=col,
            capsize=4,
            label=f"95% CI ({lab})",
        )
    ax.axvline(0, color="black", lw=0.7)
    ax.set_yticks(y, inf.index)
    ax.set_xlabel("Annualized Sharpe ratio")
    ax.set_title("How precisely are the Sharpe ratios estimated?")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8, frameon=False)
    _save(fig, path)


def robustness_heatmap(grid, baseline, path):
    tab = grid.pivot(index="Lookback", columns="Top N", values="Sharpe")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    im = ax.imshow(tab.values, cmap="viridis", aspect="auto")
    mid = np.nanmean(tab.values)
    ax.set_xticks(range(tab.shape[1]), tab.columns)
    ax.set_yticks(range(tab.shape[0]), tab.index)
    for i, lb in enumerate(tab.index):
        for j, n in enumerate(tab.columns):
            bold = (lb, n) == baseline
            ax.text(
                j,
                i,
                f"{tab.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="black" if tab.iloc[i, j] > mid else "white",
                fontsize=9,
                fontweight="bold" if bold else "normal",
            )
    ax.set(xlabel="Number of sectors held (Top N)", ylabel="Look-back (months)")
    ax.set_title("Momentum Sharpe ratio across settings (bold = baseline)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    _save(fig, path)
