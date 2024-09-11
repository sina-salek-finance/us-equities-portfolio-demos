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
    """
    Adds specified factors to the given pipeline.

    Parameters:
    pipeline (Pipeline): The pipeline to which factors will be added.
    universe (Filter): The universe of assets to consider.
    all_factors (list): List of factor names to be added to the pipeline.

    Returns:
    Pipeline: The updated pipeline with the specified factors added.
    """
    factor_functions = {
        "Momentum_1YR": lambda: momentum_1yr(252, universe),
        "Mean_Reversion_Smoothed": lambda: mean_reversion_5day_smoothed(20, universe),
        "Overnight_Sentiment_Smoothed": lambda: overnight_sentiment_smoothed(2, 10, universe),
        "volatility_20d": lambda: AnnualizedVolatility(window_length=20, mask=universe).rank().zscore(),
        "volatility_120d": lambda: AnnualizedVolatility(window_length=120, mask=universe).rank().zscore(),
        "adv_20d": lambda: AverageDollarVolume(window_length=20, mask=universe).rank().zscore(),
        "adv_120d": lambda: AverageDollarVolume(window_length=120, mask=universe).rank().zscore(),
        "dispersion_20d": lambda: SimpleMovingAverage(inputs=[MarketDispersion(mask=universe)], window_length=20),
        "dispersion_120d": lambda: SimpleMovingAverage(inputs=[MarketDispersion(mask=universe)], window_length=120),
        "market_vol_20d": lambda: MarketVolatility(window_length=20),
        "market_vol_120d": lambda: MarketVolatility(window_length=120),
        "return_5d": lambda: Returns(window_length=5, mask=universe).quantiles(2),
        "return_5d_no_quantile": lambda: Returns(window_length=5, mask=universe),
        "return_5d_p": lambda: Returns(window_length=5, mask=universe).quantiles(25),
    }

    for factor_name, factor_function in factor_functions.items():
        if factor_name in all_factors:
            pipeline.add(factor_function(), factor_name)

    return pipeline

def create_calendar_features(all_factors, factor_start_date, universe_end_date):
    """
    Creates calendar-based features for the given factors.

    Parameters:
    all_factors (DataFrame): The DataFrame containing the factors.
    factor_start_date (Timestamp): The start date of the factors.
    universe_end_date (Timestamp): The end date of the universe.

    Returns:
    DataFrame: The DataFrame with the calendar-based features added.
    """
    date_index = all_factors.index.get_level_values(0)

    all_factors["is_January"] = date_index.month == 1
    all_factors["is_December"] = date_index.month == 12
    all_factors["weekday"] = date_index.weekday
    all_factors["quarter"] = date_index.quarter
    all_factors["qtr_yr"] = all_factors.quarter.astype("str") + "_" + date_index.year.astype("str")

    frequencies = {
        "month_end": "BM",
        "month_start": "BMS",
        "qtr_end": "BQ",
        "qtr_start": "BQS"
    }

    for feature_name, freq in frequencies.items():
        all_factors[feature_name] = date_index.isin(
            pd.date_range(start=factor_start_date, end=universe_end_date, freq=freq)
        )

    return all_factors
