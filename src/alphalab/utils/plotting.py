
import matplotlib.pyplot as plt
import seaborn as sns

from alphalab.utils.alphalens_func_wrappers import show_sample_results
from alphalab.utils.tidy_functions import spearman_cor


def plot_rolling_autocorrelation(all_factors):
    """
    Plots the rolling autocorrelation of shifted target variables over time.

    Parameters:
    all_factors (DataFrame): A pandas DataFrame containing the data with a multi-index.
    """
    for i in range(5):
        all_factors[f"target_{i}"] = all_factors.groupby(level=1)["return_5d"].shift(-5 + i)

    grouped_factors = all_factors.dropna().groupby(level=0)

    sns.set(style="whitegrid")

    plt.figure(figsize=(12, 6))
    for i in range(1, 5):
        label = f"target_{i}"
        ic = grouped_factors.apply(spearman_cor, "target_0", label)
        ic.plot(ylim=(-1, 1), label=label, linewidth=2)

    plt.legend(title="Lag", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("Rolling Autocorrelation of Shifted Targets", fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Spearman Correlation", fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_multiple_series(x_series, y_series, labels, title="", x_label="", y_label=""):
    """
    Plots multiple data series on a single graph with enhanced aesthetics.

    Parameters:
    x_series (list of lists): A list containing x-axis data for each series.
    y_series (list of lists): A list containing y-axis data for each series.
    labels (list of str): A list of labels for each data series.
    title (str): The title of the plot.
    x_label (str): The label for the x-axis.
    y_label (str): The label for the y-axis.
    """
    sns.set(style="whitegrid")

    plt.figure(figsize=(10, 6))

    for x, y, label in zip(x_series, y_series, labels):
        plt.plot(x, y, label=label, linewidth=2)

    plt.title(title, fontsize=16)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.legend(title="Series", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plt.show()

def plot_factor_performance(all_factors,clf_nov, X_train, all_pricing, X_valid, X_test):
    """
    Plots the performance of factors on the training, validation, and test sets.

    """

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
