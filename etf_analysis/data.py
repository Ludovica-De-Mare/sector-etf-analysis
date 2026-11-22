"""Download, cache and convert the raw data.

Raw downloads are cached as CSV in data/ (git-ignored), so a second run uses
exactly the same numbers even if Yahoo later revises its history.
Use --refresh to download again.
"""

from pathlib import Path

import numpy as np
import pandas as pd

TRADING_DAYS = 252
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"


def download_prices(tickers, start, end) -> pd.DataFrame:
    """Daily adjusted close prices (splits and dividends reinvested)."""
    import yfinance as yf  # imported here so tests run without it

    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
    prices = prices[tickers].dropna(how="all")
    missing = [t for t in tickers if prices[t].isna().all()]
    if missing:
        raise RuntimeError(f"No data downloaded for: {missing}")
    return prices.dropna()  # keep dates where every ETF has a price


def download_tbill(start, end) -> pd.Series:
    """3-month T-bill rate (FRED series DTB3), annualized, in decimals."""
    df = pd.read_csv(FRED_CSV.format("DTB3"))
    s = pd.Series(
        pd.to_numeric(df.iloc[:, 1], errors="coerce").values,
        index=pd.to_datetime(df.iloc[:, 0]),
        name="rf",
    )
    return (s / 100).loc[start:end].ffill().dropna()


def load_data(cfg, refresh=False):
    """Return (prices, tbill), from the local cache when available."""
    folder = Path(cfg.data_dir)
    folder.mkdir(exist_ok=True)
    p_file, r_file = folder / "prices.csv", folder / "tbill.csv"

    if p_file.exists() and r_file.exists() and not refresh:
        prices = pd.read_csv(p_file, index_col=0, parse_dates=True)
        tbill = pd.read_csv(r_file, index_col=0, parse_dates=True).iloc[:, 0]
        if set(cfg.tickers) <= set(prices.columns):
            print(f"Using cached data in {folder}/ (run with --refresh to re-download)")
            return prices[cfg.tickers], tbill

    prices = download_prices(cfg.tickers, cfg.start, cfg.end)
    try:
        tbill = download_tbill(cfg.start, cfg.end)
    except Exception as e:  # keep the project running if FRED is unreachable
        print(f"Could not download the T-bill rate ({e}); using rf = 0.")
        tbill = pd.Series(0.0, index=prices.index, name="rf")
    prices.to_csv(p_file)
    tbill.to_csv(r_file)
    return prices, tbill


def month_end(df):
    """Last observation of each calendar month (old and new pandas)."""
    try:
        return df.resample("ME").last()
    except ValueError:
        return df.resample("M").last()


def month_mean(df):
    try:
        return df.resample("ME").mean()
    except ValueError:
        return df.resample("M").mean()


def simple_returns(prices):
    """r_t = P_t / P_{t-1} - 1, first row dropped."""
    return (prices / prices.shift(1) - 1).iloc[1:]


def to_returns(prices, tbill) -> dict:
    """Daily and monthly simple returns plus matching risk-free rates."""
    daily = simple_returns(prices)
    monthly = simple_returns(month_end(prices))

    rf_annual = tbill.reindex(prices.index).ffill().bfill()
    rf_daily = ((1 + rf_annual) ** (1 / TRADING_DAYS) - 1).reindex(daily.index)
    # average annual rate during each month, converted to a monthly rate
    rf_monthly = ((1 + month_mean(rf_annual)) ** (1 / 12) - 1).reindex(monthly.index)

    return {
        "daily": daily,
        "monthly": monthly,
        "rf_daily": rf_daily.astype(float),
        "rf_monthly": rf_monthly.astype(float),
    }


def synthetic_prices(tickers, start="2006-01-01", end="2015-12-31", seed=0):
    """Fake prices from a one-factor model. Used ONLY by the tests (no network)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    market = rng.normal(0.0003, 0.011, len(idx))
    betas = rng.uniform(0.6, 1.4, len(tickers))
    noise = rng.normal(0, 0.008, (len(idx), len(tickers)))
    rets = 0.0001 + market[:, None] * betas + noise
    rets[:, 0] = market  # first ticker is the benchmark
    prices = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=tickers)
    tbill = pd.Series(0.02, index=idx, name="rf")
    return prices, tbill
