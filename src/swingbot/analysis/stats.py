"""Overlap- and cross-section-aware significance tests and FDR control."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

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


def min_dates_for_test(horizon: int) -> int:
    """Fewest dates for which a HAC p-value is reported (small-sample HAC is anti-conservative)."""
    return max(3, 5 * int(horizon))


def effective_df(n_dates: int, horizon: int) -> int:
    """t-distribution df for the HAC test: about the number of non-overlapping blocks."""
    return max(n_dates // (2 * int(horizon)) - 1, 1)


def hac_test(daily: pd.Series, horizon: int, n: int) -> TestResult:
    """Test that the mean of a date-sorted per-date series is zero.

    OLS on a constant with Bartlett HAC (Newey-West) errors, ``maxlags = 2*horizon``.
    The two-sided p uses a t distribution with an *effective* df of
    ``max(n_dates // (2*horizon) - 1, 1)`` (roughly the number of
    non-overlapping blocks), not ``n_dates - 1``: with overlapping outcomes
    the HAC se is itself noisy, and the naive df over-rejects. se/t/p are NaN
    when ``n_dates < 5*horizon`` (or < 3) or the series is constant.

    Calibration (size <= ~6-8% at nominal 5%) was checked by simulation only
    for two regimes: dense iid firings and sparse Markov on/off runs (~10-day
    runs, ~3% occupancy), h=10, ~1500 dates. It is not claimed in general.
    """
    n_dates = int(len(daily))
    if n_dates == 0:
        return TestResult(NAN, NAN, NAN, NAN, int(n), 0)
    y = daily.to_numpy(dtype="float64")
    mean = float(y.mean())
    if n_dates < min_dates_for_test(horizon) or float(np.ptp(y)) == 0.0:
        return TestResult(mean, NAN, NAN, NAN, int(n), n_dates)
    fit = sm.OLS(y, np.ones((n_dates, 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 2 * int(horizon)}, use_t=True)
    se = float(fit.bse[0])
    if not np.isfinite(se) or se <= 0:
        return TestResult(mean, NAN, NAN, NAN, int(n), n_dates)
    t = float(fit.tvalues[0])
    p = float(2.0 * stats.t.sf(abs(t), effective_df(n_dates, horizon)))
    return TestResult(mean, se, t, p, int(n), n_dates)


def clustered_mean_test(values: pd.Series, dates: pd.Series, horizon: int) -> TestResult:
    """Two-sided test of mean(values) == 0, robust to same-day and overlap correlation.

    NaN values are dropped. Values are averaged per calendar date (removing
    cross-sectional correlation between assets on the same day), the per-date
    means are sorted by date, and an OLS on a constant is fitted with HAC
    (Newey-West) standard errors with ``maxlags=2*horizon``, and the p-value
    uses a t distribution with effective df ``max(n_dates // (2*horizon) - 1, 1)``
    (see ``hac_test``, which also lists the regimes where calibration was
    checked). This absorbs the autocorrelation from overlapping
    ``horizon``-day forward returns.

    ``mean`` is the mean of the per-date means, so it equal-weights *dates*,
    not occurrences: a day with 50 firings counts the same as a day with one.
    ``n`` is the number of non-NaN values, ``n_dates`` the number of distinct
    dates. With fewer than ``5*horizon`` dates (or 3), or a constant per-date
    series, se/t/p are NaN (mean is still reported when n > 0). Never raises
    on degenerate input.
    """
    daily = per_date_means(values, dates)
    n = int(np.count_nonzero(~np.isnan(np.asarray(values, dtype="float64"))))
    return hac_test(daily, horizon, n)


def date_matched_edge_test(signal_daily: pd.Series, base_daily: pd.Series,
                           horizon: int, n: int) -> TestResult:
    """Test whether the signal beats the same-day all-rows average.

    ``signal_daily``: per-date mean of the signal's values (date index, sorted).
    ``base_daily``: per-date mean of all rows (signal or not) in the same
    group/split. For each signal date, d = signal mean - all-rows mean; the
    result's ``mean`` is the date-weighted mean of d and t/p come from
    ``hac_test`` on the date-sorted d series.
    """
    sig = signal_daily.sort_index()
    d = (sig - base_daily.reindex(sig.index)).dropna()
    return hac_test(d, horizon, n)


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
