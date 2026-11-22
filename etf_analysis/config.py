"""All analysis settings in one place. Change them here (or pass a new Config)."""

from dataclasses import dataclass, field

SECTORS = {  # the nine original SPDR sector ETFs (all trade since Dec 1998)
    "XLB": "Materials",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLK": "Technology",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
}

STYLE = {  # used to build the style-factor proxies
    "IWF": "Russell 1000 Growth",
    "IWD": "Russell 1000 Value",
    "IWM": "Russell 2000 (small cap)",
}


@dataclass(frozen=True)
class Config:
    # sample and universe
    start: str = "2006-01-01"  # includes the 2008 financial crisis
    end: str = "2026-10-31"  # fixed end date so results are reproducible
    benchmark: str = "SPY"
    sectors: dict = field(default_factory=lambda: dict(SECTORS))
    style: dict = field(default_factory=lambda: dict(STYLE))

    # momentum strategy (baseline)
    top_n: int = 3  # sectors held
    lookback: int = 12  # months in the look-back window
    skip: int = 1  # skip the most recent month ("12-1 momentum")
    cost_bps: float = 10.0  # cost per unit of turnover, in basis points

    # inference
    hac_lags: int = 3  # Newey-West lags (monthly data)
    block_len: int = 6  # months per bootstrap block
    n_boot: int = 5000  # bootstrap replications
    seed: int = 42  # one seed for everything random

    # robustness grid (Sharpe ratio of every combination is reported)
    grid_lookbacks: tuple = (3, 6, 9, 12)
    grid_top_n: tuple = (1, 2, 3, 4, 5)

    # folders (both are git-ignored)
    data_dir: str = "data"
    out_dir: str = "output"

    @property
    def tickers(self) -> list:
        return [self.benchmark, *self.sectors, *self.style]

    @property
    def labels(self) -> dict:
        return {self.benchmark: "S&P 500", **self.sectors, **self.style}
