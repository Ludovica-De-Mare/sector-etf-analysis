"""The whole analysis, from prices to tables, charts and report.

run() takes the data as arguments (it does not download anything), so the
tests can run it end to end on synthetic prices.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .backtest import momentum_signal, robustness_grid, run_backtest, top_n_weights
from .data import TRADING_DAYS, to_returns
from .inference import sharpe_inference
from .metrics import perf_stats
from .regressions import momentum_panel, momentum_panel_regression, style_regressions
from .report import write_report


def run(cfg, prices: pd.DataFrame, tbill: pd.Series) -> dict:
    out = Path(cfg.out_dir)
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(cfg.seed)  # the only source of randomness

    d = to_returns(prices, tbill)
    daily, monthly = d["daily"], d["monthly"]
    rf_d, rf_m = d["rf_daily"], d["rf_monthly"]
    bench = cfg.benchmark
    sectors = list(cfg.sectors)
    sector_ret = monthly[sectors]

    # 1. Risk and return of each ETF (daily data) ---------------------
    etf_stats = pd.DataFrame(
        {t: perf_stats(daily[t], rf_d, daily[bench], TRADING_DAYS) for t in cfg.tickers}
    ).T.drop(columns=["Tracking error", "Info ratio"])
    etf_stats.insert(0, "Name", [cfg.labels[t] for t in etf_stats.index])
    corr = monthly[sectors + [bench]].corr()

    # 2. Style exposures with robust inference ------------------------
    style = style_regressions(monthly, rf_m, cfg.sectors, bench, cfg.hac_lags)

    # 3. Backtest ------------------------------------------------------
    # Common sample: starts when the LONGEST look-back in the grid has a
    # signal, so baseline and grid use exactly the same months.
    longest = momentum_signal(sector_ret, max(cfg.grid_lookbacks + (cfg.lookback,)), cfg.skip)
    start = longest.dropna().index[0]

    signal = momentum_signal(sector_ret, cfg.lookback, cfg.skip).loc[start:]
    w_mom = top_n_weights(signal, cfg.top_n)
    w_ew = pd.DataFrame(1 / len(sectors), index=signal.index, columns=sectors)
    mom_ret, mom_to = run_backtest(w_mom, sector_ret, cfg.cost_bps)
    ew_ret, ew_to = run_backtest(w_ew, sector_ret, cfg.cost_bps)
    spy_ret = monthly[bench].loc[mom_ret.index]
    rf_bt = rf_m.loc[mom_ret.index]

    mom_name = f"Sector momentum (top {cfg.top_n})"
    strategies = pd.DataFrame(
        {"SPY buy-and-hold": spy_ret, "Equal-weight sectors": ew_ret, mom_name: mom_ret}
    )
    bt_stats = pd.DataFrame({k: perf_stats(v, rf_bt, spy_ret, 12) for k, v in strategies.items()}).T
    bt_stats["Months beating SPY"] = [(v > spy_ret).mean() for _, v in strategies.items()]
    bt_stats["Annual turnover"] = [0.0, ew_to.mean() * 12, mom_to.mean() * 12]
    # comparing SPY with itself is meaningless: leave those cells empty
    bt_stats.loc["SPY buy-and-hold", ["Tracking error", "Info ratio", "Months beating SPY"]] = np.nan
    yearly = (1 + strategies).groupby(strategies.index.year).prod() - 1
    holding = (w_mom.iloc[:-1] > 0).mean().sort_values(ascending=False)
    holding.index = [f"{t} ({cfg.sectors[t]})" for t in holding.index]

    # 4. Is the Sharpe ratio difference real? --------------------------
    excess = strategies.sub(rf_bt, axis=0)
    sharpe_inf = sharpe_inference(
        excess, "SPY buy-and-hold", 12, cfg.hac_lags, cfg.block_len, cfg.n_boot, rng
    )

    # 5. Does momentum predict next month's relative return? -----------
    panel = momentum_panel(sector_ret, momentum_signal(sector_ret, cfg.lookback, cfg.skip))
    panel_res = momentum_panel_regression(panel)

    # 6. Robustness grid -------------------------------------------------
    grid = robustness_grid(
        sector_ret, rf_m, cfg.grid_lookbacks, cfg.grid_top_n, cfg.skip, cfg.cost_bps, start
    )

    # Save tables ---------------------------------------------------------
    etf_stats.to_csv(out / "etf_risk_return.csv")
    corr.to_csv(out / "sector_correlations.csv")
    style.to_csv(out / "style_exposures.csv")
    bt_stats.to_csv(out / "backtest_summary.csv")
    strategies.to_csv(out / "backtest_monthly_returns.csv")
    yearly.to_csv(out / "backtest_calendar_years.csv")
    sharpe_inf.to_csv(out / "sharpe_inference.csv")
    pd.Series(panel_res).to_csv(out / "momentum_panel_regression.csv", header=["value"])
    grid.to_csv(out / "robustness_grid.csv", index=False)

    # Charts ----------------------------------------------------------------
    plots.risk_return(etf_stats, cfg, out / "1_risk_return.png")
    plots.correlations(corr, out / "2_correlations.png")
    plots.style_exposures(style, cfg, out / "3_style_exposures.png")
    plots.rolling_volatility(daily, cfg, out / "4_rolling_volatility.png")
    plots.backtest(strategies, cfg.cost_bps, out / "5_backtest.png")
    plots.sharpe_intervals(sharpe_inf, out / "6_sharpe_intervals.png")
    plots.robustness_heatmap(grid, (cfg.lookback, cfg.top_n), out / "7_robustness_grid.png")

    results = {
        "cfg": cfg,
        "prices": prices,
        "etf_stats": etf_stats,
        "style": style,
        "bt_stats": bt_stats,
        "strategies": strategies,
        "holding": holding,
        "sharpe_inf": sharpe_inf,
        "panel": panel_res,
        "grid": grid,
        "mom_name": mom_name,
    }
    write_report(results, out / "report.md")
    return results
