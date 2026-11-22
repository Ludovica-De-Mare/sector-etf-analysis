"""Regressions: style exposures (time series) and momentum predictability (panel)."""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def style_factors(monthly, rf_monthly, benchmark="SPY") -> pd.DataFrame:
    """ETF proxies for the market, size and value factors (monthly)."""
    return pd.DataFrame(
        {
            "MKT": monthly[benchmark] - rf_monthly,  # market risk premium
            "SIZE": monthly["IWM"] - monthly[benchmark],  # small minus large
            "VALUE": monthly["IWD"] - monthly["IWF"],  # value minus growth
        }
    )


def style_regressions(monthly, rf_monthly, sectors: dict, benchmark="SPY", lags=3):
    """Excess sector return on MKT, SIZE, VALUE with Newey-West (HAC) errors.

    Adds two robust Wald tests per sector:
      * p(no style tilt): H0 beta_SIZE = beta_VALUE = 0 (joint, chi2 with 2 df)
      * p(beta MKT = 1):  H0 the sector moves one-for-one with the market
    """
    X = sm.add_constant(style_factors(monthly, rf_monthly, benchmark))
    rows = {}
    for t, name in sectors.items():
        y = monthly[t] - rf_monthly
        fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        wald = fit.wald_test("SIZE = 0, VALUE = 0", use_f=False, scalar=True)
        rows[t] = {
            "Sector": name,
            "Alpha (ann.)": fit.params["const"] * 12,
            "t(alpha)": fit.tvalues["const"],
            "Beta MKT": fit.params["MKT"],
            "Beta SIZE": fit.params["SIZE"],
            "Beta VALUE": fit.params["VALUE"],
            "t(SIZE)": fit.tvalues["SIZE"],
            "t(VALUE)": fit.tvalues["VALUE"],
            "p(no style tilt)": float(wald.pvalue),
            "p(beta MKT = 1)": float(fit.t_test("MKT = 1").pvalue),
            "R2": fit.rsquared,
        }
    out = pd.DataFrame(rows).T
    num = out.columns.drop("Sector")
    out[num] = out[num].astype(float)
    return out


def momentum_panel(sector_ret: pd.DataFrame, signal: pd.DataFrame) -> pd.DataFrame:
    """Long panel: one row per (month, sector) with the signal at t and the return at t+1."""
    nxt = sector_ret.shift(-1)  # return earned in the month after the signal
    z = signal.sub(signal.mean(axis=1), axis=0).div(signal.std(axis=1), axis=0)
    panel = pd.DataFrame({"ret_next": nxt.stack(), "mom": signal.stack(), "mom_z": z.stack()}).dropna()
    panel.index.names = ["date", "sector"]
    return panel


def momentum_panel_regression(panel: pd.DataFrame) -> dict:
    """ret_next(i, t+1) = a_t + b * mom_z(i, t) + e, with month fixed effects a_t.

    Month fixed effects are removed by demeaning within each month (the
    Frisch-Waugh-Lovell result: same b as adding one dummy per month).
    They absorb everything common to all sectors in a month (market moves),
    so b measures whether RELATIVE past winners earn RELATIVE future returns.

    Standard errors are clustered by month: all sectors share the same shock
    in a given month, so their errors are correlated within a month.
    """
    dates = panel.index.get_level_values("date")
    y = panel["ret_next"] - panel.groupby(level="date")["ret_next"].transform("mean")
    x = panel["mom_z"] - panel.groupby(level="date")["mom_z"].transform("mean")

    naive = sm.OLS(y.values, x.values).fit()
    clustered = sm.OLS(y.values, x.values).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(dates)[0]}
    )
    b = float(clustered.params[0])
    return {
        "Coefficient (monthly)": b,
        "Coefficient (annualized)": b * 12,
        "SE clustered by month": float(clustered.bse[0]),
        "t clustered": float(clustered.tvalues[0]),
        "p clustered": float(clustered.pvalues[0]),
        "SE naive (iid)": float(naive.bse[0]),
        "Observations": int(len(panel)),
        "Months (clusters)": int(dates.nunique()),
        "Sectors": int(panel.index.get_level_values("sector").nunique()),
    }


def fixed_effects_by_dummies(panel: pd.DataFrame) -> float:
    """Same regression with explicit month dummies. Used in the tests to verify
    that demeaning gives the identical coefficient (FWL)."""
    dummies = pd.get_dummies(panel.index.get_level_values("date"), dtype=float)
    X = np.column_stack([panel["mom_z"].values, dummies.values])
    return float(sm.OLS(panel["ret_next"].values, X).fit().params[0])
