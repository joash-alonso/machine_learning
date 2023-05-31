from enum import Enum
from typing import Any, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from ..loss_functions.factory import LossFunctionFactory
from .base import GradientModel, Optimiser


class LinearRegression(GradientModel):
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        learning_rate: float,
        n_iter: int,
        loss: str,
        optimiser: Optimiser,
    ):
        super().__init__(X=X, y=y)
        self.loss = LossFunctionFactory.create_loss_function(loss)
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.optimiser = optimiser
        self.is_fitted = False

    def optimisation_loop(self, X: pd.DataFrame, y: pd.Series) -> None:
        for iteration in range(self.n_iter):
            y_pred = self.predict(X)
            dW, db, cost = self.loss(X, y, y_pred)

            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * db

            self.history[iteration] = cost

    def fit(self) -> None:
        # Add a bias column to the input data
        self.weights = np.zeros(self.X.shape[1])
        self.bias = 0

        if self.optimiser == Optimiser.gradient_descent.value:
            self.optimisation_loop(self.X, self.y)

        elif self.optimiser == Optimiser.mini_batch_gradient_descent.value:
            X_batches, y_batches = self._mini_batch(self.X, self.y, 32)
            for X_batch, y_batch in tqdm(zip(X_batches, y_batches)):
                self.optimisation_loop(X_batch, y_batch)

        elif self.optimiser == Optimiser.stochastic_gradient_descent.value:
            X_batches, y_batches = self._stochastic(self.X, self.y)
            for X_batch, y_batch in tqdm(zip(X_batches, y_batches)):
                self.optimisation_loop(X_batch, y_batch)

        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.weights is not None and self.bias is not None:
            return np.dot(X, self.weights) + self.bias
        else:
            raise Exception("Model has not been fitted yet.")

    def plot_training_curve(self) -> None:
        if self.is_fitted:
            plt.figure(figsize=(10, 6))
            plt.plot(self.history.keys(), self.history.values())
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.title(f"Training Curve - Loss Function: {self.loss.__repr__()}")
            plt.show()
        else:
            raise Exception("Model has not been fitted yet.")
