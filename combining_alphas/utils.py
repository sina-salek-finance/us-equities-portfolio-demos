import pandas as pd
import numpy as np


#  TODO: replace with a Bayesian optimization
#
def train_valid_test_split(all_x, all_y, train_size, test_size):
    # TODO: replace with a better implementation
    """
    Generate the train, validation, and test dataset.

    Parameters
    ----------
    all_x : DataFrame
        All the input samples
    all_y : Pandas Series
        All the target values
    train_size : float
        The proportion of the data used for the training dataset
    valid_size : float
        The proportion of the data used for the validation dataset
    test_size : float
        The proportion of the data used for the test dataset

    Returns
    -------
    x_train : DataFrame
        The train input samples
    x_valid : DataFrame
        The validation input samples
    x_test : DataFrame
        The test input samples
    y_train : Pandas Series
        The train target values
    y_valid : Pandas Series
        The validation target values
    y_test : Pandas Series
        The test target values
    """
    assert train_size >= 0 and train_size <= 1.0
    assert test_size >= 0 and test_size <= 1.0
    assert train_size + test_size == 1.0

    """ Fix this hack after resolving data issue"""
    all_x_index_level_0 = pd.DatetimeIndex(
        np.sort(all_x.reset_index().level_0.unique())
    )
    len_x_idx = len(all_x_index_level_0)
    train_idx = all_x_index_level_0[: int(len_x_idx * train_size)]
    test_idx = all_x_index_level_0[int(len_x_idx * (train_size)) :]
    x_train = all_x.loc[(train_idx, slice(None)), :]
    x_test = all_x.loc[(test_idx, slice(None)), :]
    y_train = all_y.loc[(train_idx, slice(None))]
    y_test = all_y.loc[(test_idx, slice(None))]

    return x_train, x_test, y_train, y_test
