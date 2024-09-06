from alphalab.markowitz.portfolio_optimisation.data_preprocessing import (
    get_factor_betas,
    get_factor_returns,
    get_factor_cov_matrix,
    get_idiosyncratic_var_matrix,
    get_idiosyncratic_var_vector,
)

from alphalab.utils.zipline_func_wrappers import (
    get_pricing,
)

from alphalab.utils.tidy_functions import fit_pca

import pandas as pd
import numpy as np


def get_all_backtest_data(
    alpha_vectors,
    trading_calendar,
    data_portal,
):
    factor_betas_dict = {}
    risk_factor_cov_matrix_dict = {}
    risk_idiosyncratic_var_vector_dict = {}
    lambdas_dict = {}

    for end_date in pd.DatetimeIndex(
        set(alpha_vectors.reset_index()["level_0"].unique()).intersection(
            trading_calendar.closes.index
        )
    ):
        start_date = end_date - pd.DateOffset(years=5)

        while start_date not in trading_calendar.closes.index:
            start_date -= pd.DateOffset(days=1)

        date_tickers = alpha_vectors.loc[end_date].index
        returns = (
            get_pricing(
                data_portal, trading_calendar, date_tickers, start_date, end_date
            )
            .pct_change()[1:]
            .fillna(0)
        )

        num_factor_exposures = 20
        pca = fit_pca(returns, num_factor_exposures, "full")

        factor_betas_dict[end_date] = get_factor_betas(
            pca, returns.columns.values, np.arange(num_factor_exposures)
        )

        risk_factor_returns = get_factor_returns(
            pca, returns, returns.index, np.arange(num_factor_exposures)
        )

        ann_factor = 252
        risk_factor_cov_matrix_dict[end_date] = get_factor_cov_matrix(
            risk_factor_returns, ann_factor
        )

        risk_idiosyncratic_var_matrix = get_idiosyncratic_var_matrix(
            returns, risk_factor_returns, factor_betas_dict[end_date], ann_factor
        )

        risk_idiosyncratic_var_vector_dict[end_date] = get_idiosyncratic_var_vector(
            returns, risk_idiosyncratic_var_matrix
        )
        start_date_60 = start_date - pd.DateOffset(days=60)
        while start_date_60 not in trading_calendar.closes.index:
            start_date_60 -= pd.DateOffset(days=1)
        d60_close = get_pricing(
            data_portal, trading_calendar, date_tickers, start_date_60, end_date
        ).fillna(0)

        d60_volume = get_pricing(
            data_portal,
            trading_calendar,
            date_tickers,
            start_date_60,
            end_date,
            field="volume",
        )
        average_dollar_volume = np.nanmean(d60_close * d60_volume, axis=0)
        # scaling factor for transaction cost
        lambdas_dict[end_date] = 1 / (average_dollar_volume * 10)

    return (
        factor_betas_dict,
        risk_factor_cov_matrix_dict,
        risk_idiosyncratic_var_vector_dict,
        lambdas_dict,
    )
