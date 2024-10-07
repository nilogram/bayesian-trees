from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


def plot_actual_vs_predicted(
    predicted,
    actual,
    xlabel,
    ylabel,
    title,
    size=(10, 7.5),
    **kwargs,
):
    """
    Plots the actual averages vs the predicted ones.

    :param predicted: the predicted averages per bin
    :param actual: the actual averages per bin
    :param xlabel: the label of the x-axis
    :param ylabel: the label of the y-axis
    :param title: the plot title
    :param size: the plot size (in inches)
    :return: the figure object
    """
    f = plt.figure(figsize=size)
    plt.scatter(
        predicted,
        actual,
        **kwargs
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    return f


def plot_performance_chart(
    predictions: pd.DataFrame, 
    actual: pd.DataFrame,
    outcome: str,
    bin_size: int = 100, 
    log_loss: Optional[float] = None
) -> Tuple[Figure, pd.DataFrame]:
    plot_data = pd.concat([predictions[outcome], actual[outcome]], axis=1)
    plot_data.columns = ["predicted", "actual"]
    plot_data.sort_values("predicted", inplace=True, ignore_index=True)
    plot_data["bin"] = (plot_data.index.to_numpy() / bin_size).astype(int)
    plot_data = plot_data.groupby("bin").agg({"predicted": "mean", "actual": "mean"})
    
    title = f"Predictions of the Probability of '{outcome}' Class\nBin Size = {bin_size}"
    if log_loss is not None:
        title += f"\nLog-Loss: {log_loss:.4f}"
    
    f = plot_actual_vs_predicted(
        plot_data["predicted"],
        plot_data["actual"],
        "Predicted",
        "Actual",
        title
    )
    return f, plot_data


def get_log_loss(predictions: pd.DataFrame, actual: pd.DataFrame) -> float:
    return -(actual.values * np.log(predictions.values)).sum(axis=1).mean()
