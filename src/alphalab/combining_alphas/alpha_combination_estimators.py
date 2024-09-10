import abc

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import Bunch


class NoOverlapVoterAbstract(VotingClassifier):
    @abc.abstractmethod
    def _calculate_oob_score(self, classifiers):
        raise NotImplementedError

    @abc.abstractmethod
    def _non_overlapping_estimators(self, x, y, classifiers, n_skip_samples):
        raise NotImplementedError

    def __init__(self, estimator, voting="soft", n_skip_samples=4):
        # List of estimators for all the subsets of data
        estimators = [("clf" + str(i), estimator) for i in range(n_skip_samples + 1)]
        #         estimators = [estimator for i in range(n_skip_samples + 1)]

        self.n_skip_samples = n_skip_samples
        #         pdb.set_trace()
        super().__init__(estimators, voting=voting)

    def fit(self, X, y, sample_weight=None):
        estimator_names, clfs = zip(*self.estimators)
        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_

        clone_clfs = [clone(clf) for clf in clfs]
        self.estimators_ = self._non_overlapping_estimators(
            X, y, clone_clfs, self.n_skip_samples
        )
        self.named_estimators_ = Bunch(**dict(zip(estimator_names, self.estimators_)))
        self.oob_score_ = self._calculate_oob_score(self.estimators_)

        return self


def calculate_oob_score(classifiers):
    """
    Calculate the mean out-of-bag score from the classifiers.

    Parameters
    ----------
    classifiers : list of Scikit-Learn Classifiers
        The classifiers used to calculate the mean out-of-bag score

    Returns
    -------
    oob_score : float
        The mean out-of-bag score
    """

    return np.mean([classifier.oob_score_ for classifier in classifiers])


def non_overlapping_estimators(x, y, classifiers, n_skip_samples):
    """
    Fit the classifiers to non overlapping data.

    Parameters
    ----------
    x : DataFrame
        The input samples
    y : Pandas Series
        The target values
    classifiers : list of Scikit-Learn Classifiers
        The classifiers used to fit on the non overlapping data
    n_skip_samples : int
        The number of samples to skip

    Returns
    -------
    fit_classifiers : list of Scikit-Learn Classifiers
        The classifiers fit to the the non overlapping data
    """

    select_idx = pd.DatetimeIndex(x.reset_index()["level_0"].unique())[
        :: n_skip_samples + 1
    ]

    non_overlapping_x, non_overlapping_y = (x.loc[select_idx], y.loc[select_idx])

    return [
        classifier.fit(non_overlapping_x, non_overlapping_y)
        for classifier in classifiers
    ]


class NoOverlapVoter(NoOverlapVoterAbstract):
    def _calculate_oob_score(self, classifiers):
        return calculate_oob_score(classifiers)

    def _non_overlapping_estimators(self, x, y, classifiers, n_skip_samples):
        return non_overlapping_estimators(x, y, classifiers, n_skip_samples)
