"""U.S. Sector ETF Analysis: risk, style exposures and a sector momentum backtest.

Independent personal project using public market data.
NOT investment advice and NOT professional investment experience.

Usage:
    pip install -r requirements.txt
    python run_analysis.py            # uses data/ cache if present
    python run_analysis.py --refresh  # re-download prices and T-bill rate
"""

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from etf_analysis.config import Config
from etf_analysis.data import load_data
from etf_analysis.pipeline import run

PACKAGES = ["numpy", "pandas", "scipy", "statsmodels", "matplotlib", "yfinance"]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "(not a git repo)"


def run_info(cfg, prices) -> dict:
    """Everything needed to reproduce this run (fd12: reproducibility header)."""
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = version(p)
        except PackageNotFoundError:
            versions[p] = "not installed"
    return {
        "run_at": f"{pd.Timestamp.now():%Y-%m-%d %H:%M}",
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()}",
        "git_commit": git_commit(),
        "packages": versions,
        "data_first_day": f"{prices.index[0]:%Y-%m-%d}",
        "data_last_day": f"{prices.index[-1]:%Y-%m-%d}",
        "config": asdict(cfg),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="re-download the data")
    args = parser.parse_args()

    cfg = Config()
    prices, tbill = load_data(cfg, refresh=args.refresh)
    info = run_info(cfg, prices)
    print("=== reproducibility header ===")
    for k in ["run_at", "python", "git_commit", "data_first_day", "data_last_day"]:
        print(f"{k:15s} {info[k]}")

    res = run(cfg, prices, tbill)

    Path(cfg.out_dir, "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    pd.set_option("display.width", 160)
    print("\n=== Backtest summary ===")
    print(res["bt_stats"].round(3).to_string())
    print("\n=== Sharpe ratio inference ===")
    print(res["sharpe_inf"].round(3).to_string())
    print("\n=== Momentum panel regression ===")
    for k, v in res["panel"].items():
        print(f"{k:28s} {v:.4f}" if isinstance(v, float) else f"{k:28s} {v}")
    print(f"\nDone. Tables, charts, report.md and run_info.json are in ./{cfg.out_dir}/")


if __name__ == "__main__":
    main()
