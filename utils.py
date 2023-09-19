import numpy as np
import pandas as pd
import sklearn

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
