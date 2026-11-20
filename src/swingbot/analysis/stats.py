"""Overlap- and cross-section-aware significance tests and FDR control."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

NAN = float("nan")


@dataclass
class TestResult:
    __test__ = False  # not a pytest test class

    mean: float
    se: float
    t: float
    p: float
    n: int
    n_dates: int


def per_date_means(values: pd.Series, dates: pd.Series) -> pd.Series:
    """Drop NaN values, average per calendar date, sort dates ascending."""
    v = np.asarray(values, dtype="float64")
    d = pd.to_datetime(pd.Series(np.asarray(dates)))
    ok = ~np.isnan(v)
    if not ok.any():
        return pd.Series([], dtype="float64")
    return pd.Series(v[ok]).groupby(d.values[ok], sort=True).mean()


def hac_test(daily: pd.Series, horizon: int, n: int) -> TestResult:
    """HAC (Newey-West, maxlags=horizon) test that the mean of a per-date series is zero."""
    n_dates = int(len(daily))
    if n_dates == 0:
        return TestResult(NAN, NAN, NAN, NAN, int(n), 0)
    y = daily.to_numpy(dtype="float64")
    mean = float(y.mean())
    if n_dates < 3 or float(np.ptp(y)) == 0.0:
        return TestResult(mean, NAN, NAN, NAN, int(n), n_dates)
    fit = sm.OLS(y, np.ones((n_dates, 1))).fit(cov_type="HAC", cov_kwds={"maxlags": int(horizon)})
    se = float(fit.bse[0])
    if not np.isfinite(se) or se <= 0:
        return TestResult(mean, NAN, NAN, NAN, int(n), n_dates)
    return TestResult(mean, se, float(fit.tvalues[0]), float(fit.pvalues[0]), int(n), n_dates)


def clustered_mean_test(values: pd.Series, dates: pd.Series, horizon: int) -> TestResult:
    """Two-sided test of mean(values) == 0, robust to same-day and overlap correlation.

    NaN values are dropped. Values are averaged per calendar date (removing
    cross-sectional correlation between assets on the same day), the per-date
    means are sorted by date, and an OLS on a constant is fitted with HAC
    (Newey-West) standard errors with ``maxlags=horizon`` (removing the
    autocorrelation from overlapping ``horizon``-day forward returns).

    ``mean`` is the mean of the per-date means, so it equal-weights *dates*,
    not occurrences: a day with 50 firings counts the same as a day with one.
    ``n`` is the number of non-NaN values, ``n_dates`` the number of distinct
    dates. With fewer than 3 dates, or a constant per-date series, se/t/p are
    NaN (mean is still reported when n > 0). Never raises on degenerate input.
    """
    daily = per_date_means(values, dates)
    n = int(np.count_nonzero(~np.isnan(np.asarray(values, dtype="float64"))))
    return hac_test(daily, horizon, n)


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values. NaN p-values stay NaN and do not count toward m."""
    p = np.asarray(p, dtype="float64")
    q = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    if m == 0:
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q
