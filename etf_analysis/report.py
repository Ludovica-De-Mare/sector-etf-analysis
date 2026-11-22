"""Write output/report.md from the actual results (no hand-typed numbers)."""

import pandas as pd

PCT, NUM, P3 = "{:.1%}", "{:.2f}", "{:.3f}"

FMT_ETF = {"CAGR": PCT, "Volatility": PCT, "Sharpe": NUM, "Max drawdown": PCT}
FMT_ETF |= {"Beta vs SPY": NUM, "Corr vs SPY": NUM}
FMT_STYLE = {c: NUM for c in ["t(alpha)", "Beta MKT", "Beta SIZE", "Beta VALUE", "t(SIZE)", "t(VALUE)", "R2"]}
FMT_STYLE |= {"Alpha (ann.)": PCT, "p(no style tilt)": P3, "p(beta MKT = 1)": P3}
FMT_BT = FMT_ETF | {
    "Tracking error": PCT,
    "Info ratio": NUM,
    "Months beating SPY": PCT,
    "Annual turnover": NUM,
}


def md_table(df, fmt, index=True):
    """Small Markdown table writer (no extra packages needed). Missing values show as '–'."""
    cols = [str(c) for c in df.columns]
    lines = [("| | " if index else "| ") + " | ".join(cols) + " |", "|---" * (len(cols) + index) + "|"]
    for idx, row in df.iterrows():
        cells = ["–" if pd.isna(v) else fmt.get(c, "{}").format(v) for c, v in row.items()]
        lines.append((f"| {idx} | " if index else "| ") + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(res, path):
    cfg, etf, style, bt = res["cfg"], res["etf_stats"], res["style"], res["bt_stats"]
    inf, panel, grid, strat = res["sharpe_inf"], res["panel"], res["grid"], res["strategies"]
    prices = res["prices"]
    lab = cfg.labels
    best = etf["Sharpe"].astype(float).idxmax()
    hi_vol = etf["Volatility"].astype(float).idxmax()
    lo_vol = etf["Volatility"].astype(float).idxmin()
    base_row = grid[(grid["Lookback"] == cfg.lookback) & (grid["Top N"] == cfg.top_n)]
    rank = int((grid["Sharpe"] > base_row["Sharpe"].iloc[0]).sum()) + 1 if len(base_row) else None

    piv = grid.pivot(index="Lookback", columns="Top N", values="Sharpe")
    piv.columns = [f"Top {c}" for c in piv.columns]
    piv.index = [f"{i} months" for i in piv.index]

    inf_main = inf[["Sharpe", "SE (delta)", "SE (bootstrap)", "CI low (bootstrap)", "CI high (bootstrap)"]]
    diff_cols = [c for c in inf.columns if c.startswith("Diff") or c.startswith("p-value")]
    inf_diff = inf.drop(index="SPY buy-and-hold")[diff_cols]

    period = f"{strat.index[0]:%b %Y} to {strat.index[-1]:%b %Y}, {len(strat)} months"
    panel_size = (
        f"{panel['Sectors']} sectors x {panel['Months (clusters)']} months "
        f"({panel['Observations']} observations)"
    )
    text = f"""# U.S. Sector ETF Analysis — Results

*Independent project using public data. Not investment advice.*
Data: Yahoo Finance (adjusted prices), FRED (3-month T-bill).
Sample: {prices.index[0]:%d %b %Y} to {prices.index[-1]:%d %b %Y}.

## 1. Risk and return (daily data, annualized)

{md_table(etf, FMT_ETF)}

- Highest Sharpe ratio: **{best}** ({lab[best]}).
- Most volatile: **{hi_vol}** ({lab[hi_vol]}); least volatile: **{lo_vol}** ({lab[lo_vol]}).

![Risk vs return](1_risk_return.png)
![Correlations](2_correlations.png)
![Rolling volatility](4_rolling_volatility.png)

## 2. Style exposures (monthly regressions, Newey-West {cfg.hac_lags} lags)

Excess sector return on MKT = SPY - rf, SIZE = IWM - SPY, VALUE = IWD - IWF.
*p(no style tilt)*: robust Wald test of beta_SIZE = beta_VALUE = 0.
*p(beta MKT = 1)*: test that the sector moves one-for-one with the market.

{md_table(style, FMT_STYLE)}

![Style exposures](3_style_exposures.png)

## 3. Backtest ({period}, {cfg.cost_bps:g} bps per unit turnover)

{md_table(bt, FMT_BT)}

Share of months each sector was held by the momentum strategy:

{md_table(res["holding"].to_frame("Share of months"), {"Share of months": PCT})}

![Backtest](5_backtest.png)

## 4. How precise are the Sharpe ratios?

Delta-method standard errors (Newey-West, {cfg.hac_lags} lags) and a moving-block
bootstrap ({cfg.n_boot:,} replications, {cfg.block_len}-month blocks, seed {cfg.seed}).

{md_table(inf_main, dict.fromkeys(inf_main.columns, NUM))}

Difference in Sharpe ratio against SPY:

{md_table(inf_diff, {c: (P3 if c.startswith("p") else NUM) for c in inf_diff.columns})}

![Sharpe intervals](6_sharpe_intervals.png)

## 5. Does past momentum predict next month's relative return?

Panel of {panel_size}.
Next-month sector return on the cross-sectionally standardized {cfg.lookback}-{cfg.skip} momentum,
with month fixed effects; standard errors clustered by month.

| | Value |
|---|---|
| Coefficient (per 1 SD of momentum, monthly) | {panel["Coefficient (monthly)"]:.4f} |
| Annualized | {panel["Coefficient (annualized)"]:.2%} |
| SE clustered by month | {panel["SE clustered by month"]:.4f} |
| t-statistic | {panel["t clustered"]:.2f} |
| p-value | {panel["p clustered"]:.3f} |
| SE if errors were independent (for comparison) | {panel["SE naive (iid)"]:.4f} |

## 6. Robustness grid (same months for every cell)

{md_table(piv, dict.fromkeys(piv.columns, NUM))}

The baseline ({cfg.lookback}-month look-back, top {cfg.top_n}) ranks {rank} of {len(grid)} settings
by Sharpe ratio.

![Robustness](7_robustness_grid.png)

## Caveats

- Historical, hypothetical results; they ignore taxes, market impact and bid-ask
  spreads beyond the assumed cost.
- One market (U.S.), one sample period, ETF proxies for academic factors.
- The grid shows sensitivity to settings; it is not used to pick the best one.
"""
    with open(path, "w") as f:
        f.write(text)
