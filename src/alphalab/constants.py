from enum import Enum

ALL_FEATURES = [
            "Mean_Reversion_Smoothed",
            "Momentum_1YR",
            "Overnight_Sentiment_Smoothed",
            "adv_120d",
            "adv_20d",
            "dispersion_120d",
            "dispersion_20d",
            "market_vol_120d",
            "market_vol_20d",
            "volatility_20d",
            "is_Janaury",
            "is_December",
            "weekday",
            "month_end",
            "month_start",
            "qtr_end",
            "qtr_start",
        ]


class ModelConfig(Enum):
    N_DAYS = 10
    N_STOCKS = 500
    CLF_RANDOM_STATE = 0

CLF_PARAMETERS = {
    "criterion": "entropy",
    "min_samples_leaf": ModelConfig.N_STOCKS.value * ModelConfig.N_DAYS.value,
    "oob_score": True,
    "n_jobs": -1,
    "random_state": ModelConfig.CLF_RANDOM_STATE.value,
}

N_TREES_L = [50, 100, 250, 500, 1000]
