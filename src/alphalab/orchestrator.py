import argparse
import os
import pickle
import time

import cvxpy as cvx
import mlflow
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import pyfolio as pf
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from zipline import get_calendar, run_algorithm
from zipline.data import bundles
from zipline.data.data_portal import DataPortal
from zipline.pipeline import Pipeline
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
from alphalab.combining_alphas.alpha_combination_estimators import NoOverlapVoter
from alphalab.combining_alphas.utils import train_valid_test_split
from alphalab.constants import ALL_FEATURES
from alphalab.markowitz.backtesting.data_processing import get_all_backtest_data
from alphalab.markowitz.backtesting.performance_analysis_funcs import (
    analyze,
    before_trading_start,
    initialize,
)
from alphalab.markowitz.portfolio_optimisation.optimisation_classes import (
    OptimalHoldingsStrictFactor,
)
from alphalab.utils import data_visualisation_utils as dvis
from alphalab.utils.alphalens_func_wrappers import show_sample_results
from alphalab.utils.tidy_functions import spearman_cor
from alphalab.utils.zipline_func_wrappers import build_pipeline_engine, get_pricing


def get_data_pipline(args):
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

    pipeline = Pipeline(screen=universe)

    return universe, trading_calendar, bundle_data, engine, data_portal, pipeline

def add_factors_to_pipeline(pipeline, universe):
    pipeline.add(momentum_1yr(252, universe), "Momentum_1YR")
    pipeline.add(
        mean_reversion_5day_smoothed(20, universe),
        "Mean_Reversion_Smoothed",
    )
    pipeline.add(
        overnight_sentiment_smoothed(2, 10, universe),
        "Overnight_Sentiment_Smoothed",
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
        AverageDollarVolume(window_length=20, mask=universe).rank().zscore(),
        "adv_20d",
    )
    pipeline.add(
        AverageDollarVolume(window_length=120, mask=universe).rank().zscore(),
        "adv_120d",
    )

    pipeline.add(
        SimpleMovingAverage(
            inputs=[MarketDispersion(mask=universe)], window_length=20
        ),
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

def main():
    with mlflow.start_run():
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--data_bundle",
            type=str,
            help="Zipline data bundle to use",
            default="quandl",
        )
        parser.add_argument(
            "--end_date", type=str, help="End date for backtest", default="2016-01-05"
        )
        parser.add_argument(
            "--years_back", type=int, help="Number of years to backtest", default=3
        )

        args = parser.parse_args()

        universe_end_date = pd.to_datetime(args.end_date)

        factor_start_date = universe_end_date - pd.DateOffset(
            years=args.years_back, days=1
        )

        mlflow.log_param("universe_end_date", universe_end_date)
        mlflow.log_param("factor_start_date", factor_start_date)
        mlflow.log_param("data_bundle", args.data_bundle)

        universe, trading_calendar, bundle_data, engine, data_portal, pipeline = get_data_pipline()

        pipeline = add_factors_to_pipeline(pipeline, universe)

        all_factors = engine.run_pipeline(
            pipeline, factor_start_date, universe_end_date
        )

        all_factors = create_calendar_features(all_factors, factor_start_date, universe_end_date)

        all_factors["target"] = all_factors.groupby(level=1)["return_5d"].shift(-5)

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


        target_label = "target"

        temp = all_factors.dropna().copy()

        X = temp[ALL_FEATURES]
        y = temp[target_label]

        X_train, X_valid, X_test, y_train, y_valid, y_test = train_valid_test_split(
            X, y, 0.6, 0.2, 0.2
        )

        X_train.to_csv("X_train.csv")
        X_valid.to_csv("X_valid.csv")
        X_test.to_csv("X_test.csv")
        y_train.to_csv("y_train.csv")
        y_valid.to_csv("y_valid.csv")
        y_test.to_csv("y_test.csv")
        mlflow.log_artifact("X_train.csv")
        mlflow.log_artifact("X_valid.csv")
        mlflow.log_artifact("X_test.csv")
        mlflow.log_artifact("y_train.csv")
        mlflow.log_artifact("y_valid.csv")
        mlflow.log_artifact("y_test.csv")

        os.remove("X_train.csv")
        os.remove("X_valid.csv")
        os.remove("X_test.csv")
        os.remove("y_train.csv")
        os.remove("y_valid.csv")
        os.remove("y_test.csv")

        n_days = 10
        n_stocks = 500

        clf_random_state = 0

        clf_parameters = {
            "criterion": "entropy",
            "min_samples_leaf": n_stocks * n_days,
            "oob_score": True,
            "n_jobs": -1,
            "random_state": clf_random_state,
        }
        n_trees_l = [50, 100, 250, 500, 1000]

        all_assets = list(all_factors.reset_index().level_1.unique())

        all_pricing = get_pricing(
            data_portal,
            trading_calendar,
            all_assets,
            factor_start_date,
            universe_end_date,
        )

        train_score = []
        valid_score = []
        oob_score = []

        for n_trees in tqdm(n_trees_l, desc="Training Models", unit="Model"):
            clf = RandomForestClassifier(n_trees, **clf_parameters)

            clf_nov = NoOverlapVoter(clf, "soft")
            clf_nov.fit(X_train, y_train)

            train_score.append(clf_nov.score(X_train, y_train.astype(int).values))
            valid_score.append(clf_nov.score(X_valid, y_valid.astype(int).values))
            oob_score.append(clf_nov.oob_score_)

        dvis.plot(
            [n_trees_l] * 3,
            [train_score, valid_score, oob_score],
            ["train", "validation", "oob"],
            "Random Forrest Accuracy",
            "Number of Trees",
        )

        # number of trees achieving the best oob score
        clf_parameters["n_estimators"] = n_trees_l[np.argmax(oob_score)]

        mlflow.log_param("clf_parameters", clf_parameters)

        clf = RandomForestClassifier(**clf_parameters)
        clf_nov = NoOverlapVoter(clf, "soft")
        clf_nov.fit(pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]))

        print(
            "train: {}, oob: {}, valid: {}".format(
                clf_nov.score(X_train, y_train.values),
                clf_nov.score(X_valid, y_valid.values),
                clf_nov.oob_score_,
            )
        )

        factor_names = [
            "Mean_Reversion_Smoothed",
            "Momentum_1YR",
            "Overnight_Sentiment_Smoothed",
            "adv_120d",
            "volatility_20d",
        ]

        show_sample_results(
            all_factors, X_train, clf_nov, factor_names, pricing=all_pricing
        )
        show_sample_results(
            all_factors, X_valid, clf_nov, factor_names, pricing=all_pricing
        )
        show_sample_results(
            all_factors, X_test, clf_nov, factor_names, pricing=all_pricing
        )

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
                        np.zeros_like(alpha_vectors.loc[end_date]),
                        index=alpha_vectors.loc[end_date].index,
                    )[0]
                else:
                    previous_weights = optimal_weights_dict[last_date][0]
                previous_weights = (
                    pd.DataFrame(alpha_vectors.loc[end_date])
                    .join(previous_weights.rename("prev"))["prev"]
                    .fillna(0)
                )
                optimal_weights_dict[end_date] = OptimalHoldingsStrictFactor(
                    weights_max=0.02,
                    weights_min=-0.02,
                    risk_cap=0.0015,
                    factor_max=0.015,
                    factor_min=-0.015,
                    transaction_cost_max=5,
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

        optimisation_time = time.time() - initial_time

        mlflow.log_metric("time_taken_to_optimise_across_test_set", optimisation_time)
        with open("optimal_weights_dict.pickle", "wb") as handle:
            pickle.dump(optimal_weights_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        mlflow.log_artifact("optimal_weights_dict.pickle")
        os.remove("optimal_weights_dict.pickle")

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

        perf.to_csv("perf.csv")
        mlflow.log_artifact("perf.csv")
        os.remove("perf.csv")

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

        returns, positions, transactions = pf.utils.extract_rets_pos_txn_from_zipline(
            perf
        )

        pf.create_full_tear_sheet(
            returns,
            positions=positions,
            transactions=transactions,
            round_trips=True,
        )


if __name__ == "__main__":
    main()
