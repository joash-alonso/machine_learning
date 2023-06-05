from enum import Enum
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from ..loss_functions.factory import LossFunctionFactory
from .base import EarlyStopping, GradientModel, Optimiser


class LinearRegression(GradientModel):
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        learning_rate: float,
        n_iter: int,
        loss: str,
        optimiser: Optimiser,
        early_stopping: EarlyStopping | None = None,
        fit_bias: bool = True,
    ):
        super().__init__(
            X=X,
            y=y,
        )
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.loss = LossFunctionFactory.create_loss_function(loss)
        self.optimiser = optimiser
        self.early_stopping = early_stopping
        self.is_fitted = False
        self.fit_bias = fit_bias

    def optimisation_loop(self, X: pd.DataFrame, y: pd.Series) -> None:
        for iteration in range(self.n_iter):
            y_pred = self.predict(X)
            dW, cost = self.loss(X, y, y_pred)
            print(f"Cost: {cost}")

            self.weights -= self.learning_rate * dW

            self.history[iteration] = cost

            if self.early_stopping:
                if self.early_stopping.check_stop_condition(loss=cost):
                    break

    def fit(self) -> None:
        if self.fit_bias:
            self.X.insert(0, "bias", 1)

        self.weights = np.zeros(self.X.shape[1])

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
        if self.weights is not None:
            return np.dot(X, self.weights)
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

    def visualise_fit(self) -> None:
        if self.is_fitted and self.X.shape[1] == 2:
            plt.figure(figsize=(10, 6))
            plt.scatter(self.X.iloc[:, 1], self.y, label="Actual")
            plt.plot(
                self.X.iloc[:, 1],
                self.predict(self.X),
                label="Predicted",
            )
            plt.xlabel("X")
            plt.ylabel("y")
            plt.title("Linear Regression")
            plt.legend()
            plt.show()
        elif self.is_fitted:
            plt.figure(figsize=(10, 6))
            plt.scatter(np.arange(0, self.X.shape[0], 1), self.y, label="Actual")
            plt.scatter(
                np.arange(0, self.X.shape[0], 1),
                self.predict(self.X),
                label="Predicted",
            )
            plt.xlabel("X")
            plt.ylabel("y")
            plt.title("Linear Regression")
            plt.legend()
            plt.show()
        else:
            raise Exception("Model has not been fitted yet.")
