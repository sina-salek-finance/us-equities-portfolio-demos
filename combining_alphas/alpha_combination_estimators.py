import abc

from sklearn.ensemble import VotingClassifier
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split
from combining_alphas.utils import train_valid_test_split
import numpy as np
import xgboost as xgb
import pandas as pd


class NoOverlapVoterAbstract(VotingClassifier):
    @abc.abstractmethod
    def _non_overlapping_estimators(self, x, y, classifiers, n_skip_samples):
        raise NotImplementedError

    def __init__(self, estimator, voting="soft", n_skip_samples=4):
        # List of estimators for all the subsets of data
        estimators = [("clf" + str(i), estimator) for i in range(n_skip_samples + 1)]

        self.n_skip_samples = n_skip_samples
        super().__init__(estimators, voting=voting)

    def fit(self, X, y, sample_weight=None, validate=False):
        estimator_names, clfs = zip(*self.estimators)
        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_

        clone_clfs = [clone(clf) for clf in clfs]
        self.estimators_ = self._non_overlapping_estimators(
            X, y, clone_clfs, self.n_skip_samples, validate
        )
        self.named_estimators_ = Bunch(**dict(zip(estimator_names, self.estimators_)))

        return self


def non_overlapping_estimators(x, y, classifiers, n_skip_samples, validate=False):
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
        The classifiers fit to the non overlapping data
    """

    select_idx = pd.DatetimeIndex(
        np.sort(x.reset_index()["level_0"].unique())[:: n_skip_samples + 1]
    )

    non_overlapping_x, non_overlapping_y = (x.loc[select_idx], y.loc[select_idx])

    # X_train, X_valid, y_train, y_valid = train_test_split(
    #     non_overlapping_x, non_overlapping_y, test_size=0.2, random_state=42
    # )

    X_train, X_valid, y_train, y_valid = train_valid_test_split(
        non_overlapping_x, non_overlapping_y, 0.8, 0.2
    )
    if validate:
        trained_clfs = []
        for classifier in classifiers:
            clf_params = classifier.get_params()
            classifier.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
            clf_params['n_estimators'] = classifier.get_booster().best_iteration + 1
            clf_params['early_stopping_rounds'] = None
            final_clf = xgb.XGBClassifier(**clf_params)
            final_clf.fit(
                pd.concat([X_train, X_valid]) ,
                pd.concat([y_train, y_valid]))
            trained_clfs.append(final_clf)
        return trained_clfs
    else:
        return [
            classifier.fit(
                pd.concat([X_train, X_valid]) ,
                pd.concat([y_train, y_valid]))
            for classifier in classifiers
        ]


class NoOverlapVoter(NoOverlapVoterAbstract):
    def _non_overlapping_estimators(self, x, y, classifiers, n_skip_samples, validate=False):
        return non_overlapping_estimators(x, y, classifiers, n_skip_samples, validate)
