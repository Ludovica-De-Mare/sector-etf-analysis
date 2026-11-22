"""End-to-end run on synthetic prices: no network, checks every output is produced."""

from dataclasses import replace

from etf_analysis.config import Config
from etf_analysis.data import synthetic_prices
from etf_analysis.pipeline import run

EXPECTED = [
    "report.md",
    "backtest_summary.csv",
    "sharpe_inference.csv",
    "momentum_panel_regression.csv",
    "robustness_grid.csv",
    "style_exposures.csv",
    "1_risk_return.png",
    "5_backtest.png",
    "6_sharpe_intervals.png",
    "7_robustness_grid.png",
]


def test_full_pipeline(tmp_path):
    cfg = replace(Config(), out_dir=str(tmp_path), n_boot=200)
    prices, tbill = synthetic_prices(cfg.tickers)
    res = run(cfg, prices, tbill)
    for f in EXPECTED:
        assert (tmp_path / f).exists(), f
    assert len(res["grid"]) == len(cfg.grid_lookbacks) * len(cfg.grid_top_n)
    assert res["strategies"].notna().all().all()
