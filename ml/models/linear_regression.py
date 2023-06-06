from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from ..loss_functions.factory import LossFunctionFactory
from .model import Model
from .optimiser import *


class LinearRegression(Model):
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        learning_rate: float,
        n_iter: int,
        loss: str,
        optimiser_type: str,
        fit_bias: bool = True,
    ):
        # super().__init__(
        #     X=X,
        #     y=y,
        # )
        self.X = X
        self.y = y
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.loss = LossFunctionFactory.create_loss_function(loss)
        self.optimiser_type = optimiser_type
        self.is_fitted = False
        self.fit_bias = fit_bias

    def fit(
        self,
        early_stopping: Optional[EarlyStopping],
        batch_size: Optional[int] = 1,
    ) -> None:
        if self.fit_bias:
            self.X.insert(0, "bias", 1)

        self.weights = np.zeros(self.X.shape[1])
        self.optimiser = OptimiserFactory.create_optimiser(
            self.optimiser_type, self, weights=self.weights
        )

        if isinstance(self.optimiser, GradientDescent):
            self.history, self.weights = self.optimiser(
                self.X,
                self.y,
                self.n_iter,
                self.loss,
                self.learning_rate,
                early_stopping,
            )

        if isinstance(self.optimiser, MiniBatchGradientDescent):
            self.history, self.weights = self.optimiser(
                self.X,
                self.y,
                self.n_iter,
                self.loss,
                self.learning_rate,
                batch_size,
                early_stopping,
            )

        if isinstance(self.optimiser, StochasticGradientDescent):
            self.history, self.weights = self.optimiser(
                self.X,
                self.y,
                self.n_iter,
                self.loss,
                self.learning_rate,
                early_stopping,
            )

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
