import matplotlib.pyplot as plt

from alphalab.utils.tidy_functions import (
    get_factor_sharpe_ratio,
    plot_factor_returns,
)

import pandas as pd
import numpy as np
import alphalens as al


def build_factor_data(factor_data, pricing):
    return {
        factor_name: al.utils.get_clean_factor_and_forward_returns(
            factor=data, prices=pricing, periods=[1]
        )
        for factor_name, data in factor_data.items()
    }


def get_factor_returns(factor_data):
    ls_factor_returns = pd.DataFrame()

    for factor, factor_data in factor_data.items():
        ls_factor_returns[factor] = al.performance.factor_returns(factor_data).iloc[
            :, 0
        ]

    return ls_factor_returns


def plot_factor_rank_autocorrelation(factor_data):
    ls_FRA = pd.DataFrame()

    unixt_factor_data = {
        factor: factor_data.set_index(
            pd.MultiIndex.from_tuples(
                [(x, y) for x, y in factor_data.index.values], names=["date", "asset"]
            )
        )
        for factor, factor_data in factor_data.items()
    }

    for factor, factor_data in unixt_factor_data.items():
        ls_FRA[factor] = al.performance.factor_rank_autocorrelation(factor_data)

    ls_FRA.plot(title="Factor Rank Autocorrelation", ylim=(0.8, 1.0))
    plt.show()


def show_sample_results(data, samples, classifier, factors, pricing):
    # Calculate the Alpha Score
    prob_array = [-1, 1]
    alpha_score = classifier.predict_proba(samples).dot(np.array(prob_array))

    # Add Alpha Score to rest of the factors
    alpha_score_label = "AI_ALPHA"
    factors_with_alpha = data.loc[samples.index].copy()
    factors_with_alpha[alpha_score_label] = alpha_score

    # Setup data for AlphaLens
    print("Cleaning Data...\n")
    factor_data = build_factor_data(
        factors_with_alpha[factors + [alpha_score_label]], pricing
    )
    print("\n-----------------------\n")

    # Calculate Factor Returns and Sharpe Ratio
    factor_returns = get_factor_returns(factor_data)
    sharpe_ratio = get_factor_sharpe_ratio(factor_returns)

    # Show Results
    print("Sharpe Ratios")
    print(sharpe_ratio.round(2))
    plot_factor_returns(factor_returns)
    plot_factor_rank_autocorrelation(factor_data)
