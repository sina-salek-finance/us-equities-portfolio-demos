from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import (
    CustomFactor,
    DailyReturns,
    Returns,
    SimpleMovingAverage,
    AnnualizedVolatility,
)

import numpy as np


def momentum_1yr(window_length, universe):
    return Returns(window_length=window_length, mask=universe).demean().rank().zscore()


def mean_reversion_5day_smoothed(window_length, universe):
    unsmoothed_factor = (
        -Returns(window_length=window_length, mask=universe).demean().rank().zscore()
    )
    return (
        SimpleMovingAverage(inputs=[unsmoothed_factor], window_length=window_length)
        .rank()
        .zscore()
    )


class CTO(Returns):
    """
    Computes the overnight return, per hypothesis from
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2554010
    """

    inputs = [USEquityPricing.open, USEquityPricing.close]

    def compute(self, today, assets, out, opens, closes):
        """
        The opens and closes matrix is 2 rows x N assets, with the most recent at the bottom.
        As such, opens[-1] is the most recent open, and closes[0] is the earlier close
        """
        out[:] = (opens[-1] - closes[0]) / closes[0]


class TrailingOvernightReturns(Returns):
    """
    Sum of trailing 1m O/N returns
    """

    window_safe = True

    def compute(self, today, asset_ids, out, cto):
        out[:] = np.nansum(cto, axis=0)


def overnight_sentiment_smoothed(
    cto_window_length, trail_overnight_returns_window_length, universe
):
    cto_out = CTO(mask=universe, window_length=cto_window_length)
    unsmoothed_factor = (
        TrailingOvernightReturns(
            inputs=[cto_out], window_length=trail_overnight_returns_window_length
        )
        .rank()
        .zscore()
    )
    return (
        SimpleMovingAverage(
            inputs=[unsmoothed_factor],
            window_length=trail_overnight_returns_window_length,
        )
        .rank()
        .zscore()
    )


class MarketDispersion(CustomFactor):
    inputs = [DailyReturns()]
    window_length = 1
    window_safe = True

    def compute(self, today, assets, out, returns):
        # returns are days in rows, assets across columns
        out[:] = np.sqrt(np.nanmean((returns - np.nanmean(returns)) ** 2))


class MarketVolatility(CustomFactor):
    inputs = [DailyReturns()]
    window_length = 1
    window_safe = True

    def compute(self, today, assets, out, returns):
        mkt_returns = np.nanmean(returns, axis=1)
        out[:] = np.sqrt(
            260.0 * np.nanmean((mkt_returns - np.nanmean(mkt_returns)) ** 2)
        )
