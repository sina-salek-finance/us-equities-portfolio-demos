import logging
from abc import ABC, abstractmethod

import cvxpy as cvx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def predict_portfolio_risk(
    factor_betas, factor_cov_matrix, idiosyncratic_var_matrix, weights
):
    """
    Get the predicted portfolio risk

    Formula for predicted portfolio risk is sqrt(X.T(BFB.T + S)X) where:
      X is the portfolio weights
      B is the factor betas
      F is the factor covariance matrix
      S is the idiosyncratic variance matrix

    Parameters
    ----------
    factor_betas : DataFrame
        Factor betas
    factor_cov_matrix : 2 dimensional Ndarray
        Factor covariance matrix
    idiosyncratic_var_matrix : DataFrame
        Idiosyncratic variance matrix
    weights : DataFrame
        Portfolio weights

    Returns
    -------
    predicted_portfolio_risk : float
        Predicted portfolio risk
    """
    assert len(factor_cov_matrix.shape) == 2
    X = weights.values
    B = factor_betas.values
    F = factor_cov_matrix
    S = idiosyncratic_var_matrix.values

    return np.sqrt(X.T.dot((B.dot(F.dot(B.T)) + S).dot(X)))[0][0]


class AbstractOptimalHoldings(ABC):
    @abstractmethod
    def _get_obj(self, weights, alpha_vector):
        """
        Get the objective function

        Parameters
        ----------
        weights : CVXPY Variable
            Portfolio weights
        alpha_vector : DataFrame
            Alpha vector

        Returns
        -------
        objective : CVXPY Objective
            Objective function
        """

        raise NotImplementedError()

    @abstractmethod
    def _get_constraints(self, weights, factor_betas, risk):
        """
        Get the constraints

        Parameters
        ----------
        weights : CVXPY Variable
            Portfolio weights
        factor_betas : 2 dimensional Ndarray
            Factor betas
        risk: CVXPY Atom
            Predicted variance of the portfolio returns

        Returns
        -------
        constraints : List of CVXPY Constraint
            Constraints
        """

        raise NotImplementedError()

    def _get_risk(
        self,
        weights,
        factor_betas,
        alpha_vector_index,
        factor_cov_matrix,
        idiosyncratic_var_vector,
    ):
        f = factor_betas.loc[alpha_vector_index].values.T * weights
        X = factor_cov_matrix
        S = np.diag(idiosyncratic_var_vector.loc[alpha_vector_index].values.flatten())

        return cvx.quad_form(f, X) + cvx.quad_form(weights, S)

    def find(
        self,
        alpha_vector,
        factor_betas,
        factor_cov_matrix,
        idiosyncratic_var_vector,
        solver,
        previous_weights,
        lambda_,
        max_iters=None,
        verbose=False,
    ):
        if solver is None:
            max_iters = None
            logger.warning("No solver specified. Setting max_iters=None.")
        weights = cvx.Variable(len(alpha_vector))
        risk = self._get_risk(
            weights,
            factor_betas,
            alpha_vector.index,
            factor_cov_matrix,
            idiosyncratic_var_vector,
        )

        constraint_params = [weights, factor_betas, risk]

        if self.transaction_cost_max:
            transaction_costs = cvx.sum(
                cvx.multiply(cvx.power(weights - previous_weights, 2), lambda_)
            )
        else:
            transaction_costs = None
        constraint_params.append(transaction_costs)

        obj = self._get_obj(weights, alpha_vector)
        constraints = self._get_constraints(
            *constraint_params,
        )

        prob = cvx.Problem(obj, constraints)
        if max_iters:
            prob.solve(max_iters=max_iters, verbose=verbose, solver=solver)
        else:
            prob.solve(verbose=verbose, solver=solver)

        optimal_weights = np.asarray(weights.value).flatten()

        return pd.DataFrame(data=optimal_weights, index=alpha_vector.index)


class OptimalHoldings(AbstractOptimalHoldings):
    def _get_obj(self, weights, alpha_vector):
        """
        Get the objective function

        Parameters
        ----------
        weights : CVXPY Variable
            Portfolio weights
        alpha_vector : DataFrame
            Alpha vector

        Returns
        -------
        objective : CVXPY Objective
            Objective function
        """
        assert len(alpha_vector.columns) == 1

        to_minimise = cvx.matmul(-alpha_vector.T, weights)

        return cvx.Minimize(to_minimise)

    def _get_constraints(self, weights, factor_betas, risk, transaction_costs):
        """
        Get the constraints

        Parameters
        ----------
        weights : CVXPY Variable
            Portfolio weights
        factor_betas : 2 dimensional Ndarray
            Factor betas
        risk: CVXPY Atom
            Predicted variance of the portfolio returns

        Returns
        -------
        constraints : List of CVXPY Constraint
            Constraints
        """
        assert len(factor_betas.shape) == 2

        constraints = [
            cvx.matmul(factor_betas.T, weights) <= self.factor_max,
            cvx.matmul(factor_betas.T, weights) >= self.factor_min,
            sum(weights) == 0.0,
            sum(cvx.abs(weights)) <= 1.0,
            weights >= self.weights_min,
            weights <= self.weights_max,
            risk <= self.risk_cap**2,
        ]

        if self.transaction_cost_max is not None:
            constraints.append(transaction_costs <= self.transaction_cost_max)

        return constraints

    def __init__(
        self,
        risk_cap=0.05,
        factor_max=10.0,
        factor_min=-10.0,
        weights_max=0.55,
        weights_min=-0.55,
        transaction_cost_max=None,
    ):
        self.risk_cap = risk_cap
        self.factor_max = factor_max
        self.factor_min = factor_min
        self.weights_max = weights_max
        self.weights_min = weights_min
        self.transaction_cost_max = transaction_cost_max


class OptimalHoldingsStrictFactor(OptimalHoldings):
    def _get_obj(self, weights, alpha_vector):
        """
        Get the objective function

        Parameters
        ----------
        weights : CVXPY Variable
            Portfolio weights
        alpha_vector : DataFrame
            Alpha vector

        Returns
        -------
        objective : CVXPY Objective
            Objective function
        """
        assert len(alpha_vector.columns) == 1

        target_weight = (alpha_vector - alpha_vector.mean()) / alpha_vector.abs().sum()
        to_minimise = cvx.norm2(weights - target_weight.values.reshape(weights.shape))

        return cvx.Minimize(to_minimise)


def determine_optimal_weights_for_all_dates(
    valid_dates,
    alpha_vectors,
    factor_betas_dict,
    risk_factor_cov_matrix_dict,
    risk_idiosyncratic_var_vector_dict,
    lambdas_dict,
):
    optimal_weights_dict = {}
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
                None,
                previous_weights,
                lambdas_dict[end_date],
                None,
            )

            last_date = end_date
        except ValueError as e:
            print(e)
            optimal_weights_dict[end_date] = optimal_weights_dict[last_date]
