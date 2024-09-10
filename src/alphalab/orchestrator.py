import argparse
import os
import pickle
import time

import cvxpy as cvx
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import pyfolio as pf
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from zipline import run_algorithm

from alphalab.combining_alphas.alpha_combination_estimators import NoOverlapVoter
from alphalab.combining_alphas.feature_creation import (
    add_factors_to_pipeline,
    create_calendar_features,
)
from alphalab.combining_alphas.utils import train_valid_test_split
from alphalab.constants import ALL_FEATURES, CLF_PARAMETERS, N_TREES_L
from alphalab.markowitz.backtesting.data_processing import get_all_backtest_data
from alphalab.markowitz.backtesting.performance_analysis_funcs import (
    analyze,
    before_trading_start,
    initialize,
)
from alphalab.markowitz.portfolio_optimisation.optimisation_classes import (
    OptimalHoldingsStrictFactor,
)
from alphalab.utils.plotting import (
    plot_factor_performance,
    plot_multiple_series,
    plot_rolling_autocorrelation,
)
from alphalab.utils.zipline_func_wrappers import get_data_pipline, get_pricing


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

        parser.add_argument(
            "--eda_mode", type=bool, help="Run in EDA mode", default=False
        )

        args = parser.parse_args()

        universe_end_date = pd.to_datetime(args.end_date)

        factor_start_date = universe_end_date - pd.DateOffset(
            years=args.years_back, days=1
        )

        mlflow.log_param("universe_end_date", universe_end_date)
        mlflow.log_param("factor_start_date", factor_start_date)
        mlflow.log_param("data_bundle", args.data_bundle)

        universe, trading_calendar, bundle_data, engine, data_portal, pipeline = get_data_pipline(args)

        pipeline = add_factors_to_pipeline(pipeline, universe, ALL_FEATURES)

        all_factors = engine.run_pipeline(
            pipeline, factor_start_date, universe_end_date
        )

        all_factors = create_calendar_features(all_factors, factor_start_date, universe_end_date)

        all_factors["target"] = all_factors.groupby(level=1)["return_5d"].shift(-5)

        all_factors = all_factors[all_factors["target"].isin([0, 1])]

        if args.eda_mode:
            plot_rolling_autocorrelation(all_factors)

        target_label = "target"

        temp = all_factors.dropna().copy()

        X = temp[ALL_FEATURES]
        y = temp[target_label]

        X_train, X_valid, X_test, y_train, y_valid, y_test = train_valid_test_split(
            X, y, 0.6, 0.2, 0.2
        )

        datasets = {
            "X_train": X_train,
            "X_valid": X_valid,
            "X_test": X_test,
            "y_train": y_train,
            "y_valid": y_valid,
            "y_test": y_test
        }

        for name, data in datasets.items():
            filename = f"{name}.csv"
            data.to_csv(filename)
            mlflow.log_artifact(filename)
            os.remove(filename)

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

        for n_trees in tqdm(N_TREES_L, desc="Training Models", unit="Model"):
            clf = RandomForestClassifier(n_trees, **CLF_PARAMETERS)

            clf_nov = NoOverlapVoter(clf, "soft")
            clf_nov.fit(X_train, y_train)

            train_score.append(clf_nov.score(X_train, y_train.astype(int).values))
            valid_score.append(clf_nov.score(X_valid, y_valid.astype(int).values))
            oob_score.append(clf_nov.oob_score_)

        if args.eda_mode:
            plot_multiple_series(
                [N_TREES_L] * 3,
                [train_score, valid_score, oob_score],
                ["train", "validation", "oob"],
                "Random Forrest Accuracy",
                "Number of Trees",
            )

        # number of trees achieving the best oob score
        CLF_PARAMETERS["n_estimators"] = N_TREES_L[np.argmax(oob_score)]

        mlflow.log_param("clf_parameters", CLF_PARAMETERS)

        clf = RandomForestClassifier(**CLF_PARAMETERS)
        clf_nov = NoOverlapVoter(clf, "soft")
        clf_nov.fit(pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]))

        print(
            "train: {}, oob: {}, valid: {}".format(
                clf_nov.score(X_train, y_train.values),
                clf_nov.score(X_valid, y_valid.values),
                clf_nov.oob_score_,
            )
        )

        plot_factor_performance(all_factors, clf_nov, X_train, all_pricing, X_valid, X_test)
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
