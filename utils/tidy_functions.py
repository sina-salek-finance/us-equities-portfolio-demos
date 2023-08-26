from scipy.stats import spearmanr
from sklearn.decomposition import PCA
import numpy as np


def get_factor_sharpe_ratio(factor_returns, annualization_factor=np.sqrt(252)):
    return annualization_factor * factor_returns.mean() / factor_returns.std()


def plot_factor_returns(factor_returns):
    (1 + factor_returns).cumprod().plot(ylim=(0.8, 1.2))
    plt.show()


def spearman_cor(group, col1_name, col2_name):
    """
    Spearman correlation between two columns of a group
    Paramerters:
    -----------
    group:
    col1_name:
    col2_name:
    returns:
    --------
    spearman correlation
    """
    x = group[col1_name]
    y = group[col2_name]
    return spearmanr(x, y)[0]


def fit_pca(returns, num_factor_exposures, svd_solver):
    """
    Fit PCA model with returns.

    Parameters
    ----------
    returns : DataFrame
        Returns for each ticker and date
    num_factor_exposures : int
        Number of factors for PCA
    svd_solver: str
        The solver to use for the PCA model

    Returns
    -------
    pca : PCA
        Model fit to returns
    """
    pca = PCA(n_components=num_factor_exposures, svd_solver=svd_solver)
    pca.fit(returns)
    return pca
