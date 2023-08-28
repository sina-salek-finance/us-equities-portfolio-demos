import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

from zipline.pipeline import Pipeline, domain
from zipline.pipeline.factors import AverageDollarVolume
from zipline import get_calendar
from zipline.data import bundles
from zipline.data.data_portal import DataPortal


from utils.zipline_func_wrappers import (
    build_pipeline_engine,
    get_pricing,
)

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from alpha_library.alphas import (
    momentum_1yr,
    mean_reversion_5day_sector_neutral_smoothed,
    overnight_sentiment_smoothed,
    MarketDispersion,
    MarketVolatility,
)

from utils.tidy_functions import spearman_cor, fit_pca

from utils.alphalens_func_wrappers import show_sample_results

from combining_alphas.utils import train_valid_test_split

from combining_alphas.model_evaluation import rank_features_by_importance
from utils import (
    data_visualisation_utils as dvis,
)

from combining_alphas.alpha_combination_estimators import NoOverlapVoter

from portfolio_optimisation.data_preprocessing import (
    get_factor_betas,
    get_factor_returns,
    get_factor_cov_matrix,
    get_idiosyncratic_var_matrix,
    get_idiosyncratic_var_vector,
)

from markowitz.backtesting.data_processing import get_all_backtest_data

from markowitz.portfolio_optimisation.optimisation_classes import (
    OptimalHoldingsStrictFactor,
)

from zipline import run_algorithm
import pandas_datareader.data as web

import cvxpy as cvx

from zipline.pipeline.factors import (
    CustomFactor,
    DailyReturns,
    Returns,
    SimpleMovingAverage,
    AnnualizedVolatility,
)

from markowitz.backtesting.performance_analysis_funcs import (
    analyze,
    initialize,
    before_trading_start,
)

import pyfolio as pf

import time


# sector = Sector()
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_bundle", type=str, help="Zipline data bundle to use", default="quandl"
    )
    parser.add_argument(
        "--end_date", type=str, help="End date for backtest", default="2016-01-05"
    )
    parser.add_argument(
        "--years_back", type=int, help="Number of years to backtest", default=7
    )

    args = parser.parse_args()

    universe_end_date = pd.to_datetime(args.end_date)

    factor_start_date = universe_end_date - pd.DateOffset(years=args.years_back)

    universe = AverageDollarVolume(window_length=120).top(500)
    trading_calendar = get_calendar("NYSE")

    bundle_data = bundles.load(args.data_bundle)
    engine = build_pipeline_engine(bundle_data, trading_calendar)

    data_portal = DataPortal(
        bundle_data.asset_finder,
        trading_calendar=trading_calendar,
        first_trading_day=bundle_data.equity_daily_bar_reader.first_trading_day,
        equity_minute_reader=None,
        equity_daily_reader=bundle_data.equity_daily_bar_reader,
        adjustment_reader=bundle_data.adjustment_reader,
    )

    universe_tickers = (
        engine.run_pipeline(
            Pipeline(screen=universe, domain=domain.US_EQUITIES),
            universe_end_date,
            universe_end_date,
        )
        .index.get_level_values(1)
        .values.tolist()
    )

    # sector = Sector()

    pipeline = Pipeline(screen=universe)
    pipeline.add(momentum_1yr(252, universe), "Momentum_1YR")  # , sector),
    pipeline.add(
        mean_reversion_5day_sector_neutral_smoothed(20, universe),  # , sector),
        "Mean_Reversion_Sector_Neutral_Smoothed",
    )
    pipeline.add(
        overnight_sentiment_smoothed(2, 10, universe), "Overnight_Sentiment_Smoothed"
    )

    pipeline.add(
        AnnualizedVolatility(window_length=20, mask=universe).rank().zscore(),
        "volatility_20d",
    )
    pipeline.add(
        AnnualizedVolatility(window_length=120, mask=universe).rank().zscore(),
        "volatility_120d",
    )
    pipeline.add(
        AverageDollarVolume(window_length=20, mask=universe).rank().zscore(), "adv_20d"
    )
    pipeline.add(
        AverageDollarVolume(window_length=120, mask=universe).rank().zscore(),
        "adv_120d",
    )

    pipeline.add(
        SimpleMovingAverage(inputs=[MarketDispersion(mask=universe)], window_length=20),
        "dispersion_20d",
    )
    pipeline.add(
        SimpleMovingAverage(
            inputs=[MarketDispersion(mask=universe)], window_length=120
        ),
        "dispersion_120d",
    )

    pipeline.add(MarketVolatility(window_length=20), "market_vol_20d")
    pipeline.add(MarketVolatility(window_length=120), "market_vol_120d")

    pipeline.add(Returns(window_length=5, mask=universe).quantiles(2), "return_5d")
    pipeline.add(Returns(window_length=5, mask=universe), "return_5d_no_quantile")
    pipeline.add(Returns(window_length=5, mask=universe).quantiles(25), "return_5d_p")

    all_factors = engine.run_pipeline(pipeline, factor_start_date, universe_end_date)

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

    # TODO: after implementing sector class, one-hot encode sector

    all_factors["target"] = all_factors.groupby(level=1)["return_5d"].shift(-5)

    # TODO: why some targets are missing (resulting in -1)?
    all_factors = all_factors[all_factors["target"].isin([0, 1])]

    all_factors["target_p"] = all_factors.groupby(level=1)["return_5d_p"].shift(-5)
    all_factors["target_1"] = all_factors.groupby(level=1)["return_5d"].shift(-4)
    all_factors["target_2"] = all_factors.groupby(level=1)["return_5d"].shift(-3)
    all_factors["target_3"] = all_factors.groupby(level=1)["return_5d"].shift(-2)
    all_factors["target_4"] = all_factors.groupby(level=1)["return_5d"].shift(-1)

    # display the rolling auto-correlation of the target
    from matplotlib import pyplot as plt

    g = all_factors.dropna().groupby(level=0)
    for i in range(4):
        label = "target_" + str(i + 1)
        ic = g.apply(spearman_cor, "target", label)
        ic.plot(ylim=(-1, 1), label=label)
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.title("Rolling Autocorrelation of Labels Shifted 1,2,3,4 Days")
    plt.show()

    features = [
        "Mean_Reversion_Sector_Neutral_Smoothed",
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
    ]  # + sector_columns
    target_label = "target"

    temp = all_factors.dropna().copy()
    X = temp[features]
    y = temp[target_label]

    X_train, X_test, y_train, y_test = train_valid_test_split(X, y, 0.8, 0.2)

    n_days = 10
    n_stocks = 500

    clf_random_state = 0

    clf_parameters = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.1,
        "max_depth": 5,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 1000,
        "early_stopping_rounds": 10,
        "seed": clf_random_state,
    }

    all_assets = all_factors.index.levels[1].values.tolist()

    all_pricing = get_pricing(
        data_portal,
        trading_calendar,
        all_assets,
        factor_start_date,
        universe_end_date,
    )

    factor_names = [
        "Mean_Reversion_Sector_Neutral_Smoothed",
        "Momentum_1YR",
        "Overnight_Sentiment_Smoothed",
        "adv_120d",
        "volatility_20d",
    ]

    # train_score = []

    clf = XGBClassifier(**clf_parameters)

    clf_nov = NoOverlapVoter(clf, "soft")
    clf_nov.fit(X_train, y_train, validate=True)

    # TODO: implement an evaluation method

    # train_score.append(clf_nov.score(X_train, y_train.astype(int).values))
    # oob_score.append(clf_nov.oob_score_)

    # dvis.plot(
    #     [n_trees_l] * 3,
    #     [train_score, valid_score, oob_score],
    #     ["train", "validation", "oob"],
    #     "Random Forrest Accuracy",
    #     "Number of Trees",
    # )

    # n_trees = 500
    #
    # clf = XGBClassifier(n_trees, **clf_parameters)
    # clf_nov = NoOverlapVoter(clf, "soft")
    # TODO: make a final fit method that takes in all data
    # clf_nov.fit(pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]))

    # print(
    #     "train: {}, oob: {}, valid: {}".format(
    #         clf_nov.score(X_train, y_train.values),
    #         clf_nov.score(X_valid, y_valid.values),
    #         clf_nov.oob_score_,
    #     )
    # )

    show_sample_results(
        all_factors, X_train, clf_nov, factor_names, pricing=all_pricing
    )

    show_sample_results(all_factors, X_test, clf_nov, factor_names, pricing=all_pricing)

    prob_array = [-1, 1]
    alpha_vectors = pd.DataFrame(
        clf_nov.predict_proba(X_test).dot(np.array(prob_array)), index=X_test.index
    )

    (
        factor_betas_dict,
        risk_factor_cov_matrix_dict,
        risk_idiosyncratic_var_vector_dict,
        lambdas_dict,
    ) = get_all_backtest_data(alpha_vectors, trading_calendar, data_portal)

    initial_time = time.time()

    optimal_weights_dict = {}

    valid_dates = pd.DatetimeIndex(
        set(alpha_vectors.reset_index()["level_0"].unique()).intersection(
            trading_calendar.closes.index
        )
    )

    for idx, end_date in enumerate(valid_dates):
        try:
            if idx == 0:
                previous_weights = pd.DataFrame(
                    np.zeros_like(alpha_vectors.loc[end_date])
                )[0]
            else:
                previous_weights = optimal_weights_dict[last_date][0]
            previous_weights = pd.DataFrame(alpha_vectors.loc[end_date]).join(
                previous_weights.rename("prev")
            )["prev"]
            optimal_weights_dict[end_date] = OptimalHoldingsStrictFactor(
                weights_max=0.02,
                weights_min=-0.02,
                risk_cap=0.0015,
                factor_max=0.015,
                factor_min=-0.015,
            ).find(
                alpha_vectors.loc[end_date],
                factor_betas_dict[end_date],
                risk_factor_cov_matrix_dict[end_date],
                risk_idiosyncratic_var_vector_dict[end_date],
                cvx.ECOS,
                previous_weights,
                lambdas_dict[end_date],
                500,
            )

            last_date = end_date
        except ValueError as e:
            print(e)
            optimal_weights_dict[end_date] = optimal_weights_dict[last_date]

    final_time = time.time() - initial_time

    # with open('optimal_weights_dict.pickle', 'wb') as handle:
    #     pickle.dump(optimal_weights_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    sorted_dates = np.sort(list(optimal_weights_dict.keys()))
    backtest_start = sorted_dates[0]
    backtest_end = sorted_dates[-1]
    capital_base = 1e7

    sp500 = web.DataReader("SP500", "fred", backtest_start, backtest_end).SP500
    benchmark_returns = sp500.pct_change()

    perf = run_algorithm(
        start=backtest_start,
        end=backtest_end,
        initialize=initialize,
        analyze=analyze,
        capital_base=capital_base,
        benchmark_returns=benchmark_returns,
        before_trading_start=before_trading_start,
        bundle="quandl",
    )

    plt.figure(figsize=(10, 6))
    plt.plot(perf.index, perf.portfolio_value, label="Portfolio Value")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.title(
        "Portfolio Backtest with Forecasted Returns, Covariance Matrix, and Transaction Costs"
    )
    plt.show()

    perf.sharpe.plot()

    returns, positions, transactions = pf.utils.extract_rets_pos_txn_from_zipline(perf)

    pf.create_full_tear_sheet(
        returns,
        positions=positions,
        transactions=transactions,
        round_trips=True,
    )


if __name__ == "__main__":
    main()
