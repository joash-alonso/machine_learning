from abc import ABC
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..loss_functions.loss_function import LossFunction
from .model import Model


class EarlyStopping:
    def __init__(self, patience: int, delta: float) -> None:
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0

    def check_stop_condition(self, loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = loss
            return False
        elif (self.best_loss - loss) < self.delta:
            self.best_loss = loss
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = loss
            self.counter = 0
            return False


class Optimiser(ABC):
    def __call__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_iter: int,
        loss_function: LossFunction,
        learning_rate: float = 0.001,
        early_stopping: Optional[EarlyStopping] = None,
    ) -> float:
        raise NotImplementedError("Optimiser not implemented")


class GradientDescent(Optimiser):
    def __init__(self, model: Model, weights: np.array) -> None:
        self.model = model
        self.history = {}
        self.weights = weights

    def __call__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_iter: int,
        loss_function: LossFunction,
        learning_rate: float = 0.001,
        early_stopping: Optional[EarlyStopping] = None,
    ) -> Tuple[Dict[str, float], np.array]:
        for iteration in range(n_iter):
            y_pred = self.model.predict(X)
            dW, cost = loss_function(X, y, y_pred)
            print(f"Cost: {cost}")

            self.weights -= learning_rate * dW

            self.history[iteration] = cost

            if early_stopping:
                if early_stopping.check_stop_condition(loss=cost):
                    break

        return self.history, self.weights

    def __repr__(self) -> str:
        return "Gradient Descent"


class MiniBatchGradientDescent(Optimiser):
    def __init__(self, model: Model) -> None:
        self.model = model
        self.history = {}
        self.weights = None

    def __call__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_iter: int,
        loss_function: LossFunction,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        early_stopping: Optional[EarlyStopping] = None,
    ) -> Tuple[Dict[str, float], np.array]:
        idx = np.random.permutation(X.shape[0])
        X = X.iloc[idx]
        y = y.iloc[idx]

        # Split the data into batches
        n_batches = X.shape[0] // batch_size
        X_batches = np.array_split(X, n_batches)
        y_batches = np.array_split(y, n_batches)

        for X_batch, y_batch in zip(X_batches, y_batches):
            for iteration in range(n_iter):
                y_pred = self.model.predict(X_batch)
                dW, cost = loss_function(X_batch, y_batch, y_pred)
                print(f"Cost: {cost}")

                self.weights -= learning_rate * dW

                self.history[iteration] = cost

                if early_stopping:
                    if early_stopping.check_stop_condition(loss=cost):
                        break

        return self.history, self.weights

    def __repr__(self) -> str:
        return "Mini-Batch Gradient Descent"


class StochasticGradientDescent(Optimiser):
    def __init__(self, model: Model) -> None:
        self.model = model
        self.history = {}
        self.weights = None

    def __call__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_iter: int,
        loss_function: LossFunction,
        learning_rate: float = 0.001,
        early_stopping: Optional[EarlyStopping] = None,
    ) -> Tuple[Dict[str, float], np.array]:
        idx = np.random.permutation(X.shape[0])
        X = X.iloc[idx]
        y = y.iloc[idx]

        # Split the data into batches
        n_batches = X.shape[0] // 1
        X_batches = np.array_split(X, n_batches)
        y_batches = np.array_split(y, n_batches)

        for X_batch, y_batch in zip(X_batches, y_batches):
            for iteration in range(n_iter):
                y_pred = self.model.predict(X_batch)
                dW, cost = loss_function(X_batch, y_batch, y_pred)
                print(f"Cost: {cost}")

                self.weights -= learning_rate * dW

                self.history[iteration] = cost

                if early_stopping:
                    if early_stopping.check_stop_condition(loss=cost):
                        break

        return self.history, self.weights

    def __repr__(self) -> str:
        return "Stochastic Gradient Descent"


class OptimiserFactory:
    @staticmethod
    def create_optimiser(
        optimiser_name: str, model: Model, weights: np.array
    ) -> Optimiser:
        if optimiser_name == "gd":
            return GradientDescent(model, weights)
        elif optimiser_name == "mbgd":
            return MiniBatchGradientDescent(model, weights)
        elif optimiser_name == "sgd":
            return StochasticGradientDescent(model, weights)
        else:
            raise ValueError("Unknown loss function: {}".format(optimiser_name))
