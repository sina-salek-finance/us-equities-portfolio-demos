import pandas as pd
from zipline.pipeline.factors import (
    AnnualizedVolatility,
    AverageDollarVolume,
    Returns,
    SimpleMovingAverage,
)

from alphalab.alpha_library.alphas import (
    MarketDispersion,
    MarketVolatility,
    mean_reversion_5day_smoothed,
    momentum_1yr,
    overnight_sentiment_smoothed,
)


def add_factors_to_pipeline(pipeline, universe, all_factors):
    if "Momentum_1YR" in all_factors:
        pipeline.add(momentum_1yr(252, universe), "Momentum_1YR")
    if "Mean_Reversion_Smoothed" in all_factors:
        pipeline.add(
            mean_reversion_5day_smoothed(20, universe),
            "Mean_Reversion_Smoothed",
        )
    if "Overnight_Sentiment_Smoothed" in all_factors:
        pipeline.add(
            overnight_sentiment_smoothed(2, 10, universe),
            "Overnight_Sentiment_Smoothed",
        )
    if "volatility_20d" in all_factors:
        pipeline.add(
            AnnualizedVolatility(window_length=20, mask=universe).rank().zscore(),
            "volatility_20d",
        )
    if "volatility_120d" in all_factors:
        pipeline.add(
            AnnualizedVolatility(window_length=120, mask=universe).rank().zscore(),
            "volatility_120d",
        )
    if "adv_20d" in all_factors:
        pipeline.add(
            AverageDollarVolume(window_length=20, mask=universe).rank().zscore(),
            "adv_20d",
        )
    if "adv_120d" in all_factors:
        pipeline.add(
            AverageDollarVolume(window_length=120, mask=universe).rank().zscore(),
            "adv_120d",
        )
    if "dispersion_20d" in all_factors:
        pipeline.add(
            SimpleMovingAverage(
                inputs=[MarketDispersion(mask=universe)], window_length=20
            ),
            "dispersion_20d",
        )
    if "dispersion_120d" in all_factors:
        pipeline.add(
            SimpleMovingAverage(
                inputs=[MarketDispersion(mask=universe)], window_length=120
            ),
            "dispersion_120d",
        )
    if "market_vol_20d" in all_factors:
        pipeline.add(MarketVolatility(window_length=20), "market_vol_20d")
    if "market_vol_120d" in all_factors:
        pipeline.add(MarketVolatility(window_length=120), "market_vol_120d")

    if "return_5d" in all_factors:
        pipeline.add(Returns(window_length=5, mask=universe).quantiles(2), "return_5d")
    if "return_5d_no_quantile" in all_factors:
        pipeline.add(Returns(window_length=5, mask=universe), "return_5d_no_quantile")
    if "return_5d_p" in all_factors:
        pipeline.add(
        Returns(window_length=5, mask=universe).quantiles(25), "return_5d_p"
    )
    return pipeline

def create_calendar_features(all_factors, factor_start_date, universe_end_date):
    all_factors["is_Janaury"] = all_factors.index.get_level_values(0).month == 1
    all_factors["is_December"] = all_factors.index.get_level_values(0).month == 12
    all_factors["weekday"] = all_factors.index.get_level_values(0).weekday
    all_factors["quarter"] = all_factors.index.get_level_values(0).quarter
    all_factors["qtr_yr"] = (
            all_factors.quarter.astype("str")
            + "_"
            + all_factors.index.get_level_values(0).year.astype("str")
    )
    all_factors["month_end"] = all_factors.index.get_level_values(0).isin(
        pd.date_range(start=factor_start_date, end=universe_end_date, freq="BM")
    )
    all_factors["month_start"] = all_factors.index.get_level_values(0).isin(
        pd.date_range(start=factor_start_date, end=universe_end_date, freq="BMS")
    )
    all_factors["qtr_end"] = all_factors.index.get_level_values(0).isin(
        pd.date_range(start=factor_start_date, end=universe_end_date, freq="BQ")
    )
    all_factors["qtr_start"] = all_factors.index.get_level_values(0).isin(
        pd.date_range(start=factor_start_date, end=universe_end_date, freq="BQS")
    )
    return all_factors
