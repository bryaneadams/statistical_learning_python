import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from numpy.typing import ArrayLike


def confusion_table(
    true_labels: ArrayLike, predicted_labels: ArrayLike
) -> pd.DataFrame:
    """
    Return a data frame version of confusion
    matrix with rows given by predicted label
    and columns the truth.


    Args:
        true_labels (ArrayLike): True labels of your data
        predicted_labels (ArrayLike): Predicted labels of your data

    Returns:
        pd.DataFrame: confusion matrix with rows given by predicted label and columns the truth. The diagonal of the matrix indicates correct predictions.
    """

    labels = sorted(np.unique(list(true_labels) + list(predicted_labels)))
    cm = sklearn.metrics.confusion_matrix(true_labels, predicted_labels)
    df = pd.DataFrame(cm.T, columns=labels)  # swap rows and columns
    df.index = pd.Index(labels, name="Predicted")
    df.columns.name = "Truth"
    return df


def plot_svm(
    X,
    Y,
    svm,
    features=(0, 1),
    xlim=None,
    nx=300,
    ylim=None,
    ny=300,
    ax=None,
    decision_cmap=plt.cm.plasma,
    scatter_cmap=plt.cm.tab10,
    alpha=0.2,
):
    X = np.asarray(X)

    if X.shape[1] < 2:
        raise ValueError("expecting at least 2 columns to display decision boundary")

    X0, X1 = [X[:, i] for i in features]

    if xlim is None:
        xlim = (X0.min() - 0.5 * X0.std(), X0.max() + 0.5 * X0.std())

    if ylim is None:
        ylim = (X1.min() - 0.5 * X1.std(), X1.max() + 0.5 * X1.std())

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # draw the points

    ax.scatter(X0, X1, c=Y, cmap=scatter_cmap)

    # add the contour

    xval, yval = np.meshgrid(
        np.linspace(xlim[0], xlim[1], nx), np.linspace(ylim[0], ylim[1], ny)
    )

    # this will work well when labels are integers

    grid_val = np.array([xval.reshape(-1), yval.reshape(-1)]).T
    X_pred = np.multiply.outer(np.ones(grid_val.shape[0]), X.mean(0))
    X_pred[:, features[0]] = grid_val[:, 0]
    X_pred[:, features[1]] = grid_val[:, 1]

    prediction_val = svm.predict(X_pred)

    ax.contourf(
        xval, yval, prediction_val.reshape(yval.shape), cmap=decision_cmap, alpha=alpha
    )

    # add the support vectors

    ax.scatter(
        X[svm.support_, features[0]],
        X[svm.support_, features[1]],
        marker="+",
        c="k",
        s=200,
    )
