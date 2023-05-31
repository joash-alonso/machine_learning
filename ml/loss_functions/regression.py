from typing import Tuple

import numpy as np

from .base import LossFunction


class MeanSquaredError(LossFunction):
    """Calculates the mean squared error between the true and predicted values.

    Args:
        X (np.array): The input data.
        y_true (np.array): The true values.
        y_pred (np.array): The predicted values.

    Returns:
        tuple: A tuple containing the gradients for the weights and bias as well as the cost.
    """

    def __call__(
        self, X: np.array, y_true: np.array, y_pred: np.array
    ) -> Tuple[np.array, np.array, float]:
        n = len(y_true)
        error = y_pred - y_true
        dW = (2 / n) * np.dot(error, X)
        db = (2 / n) * np.sum(error)

        cost = np.mean(np.square(error))
        return dW, db, cost

    def __repr__(self) -> str:
        return "Mean Squared Error"
