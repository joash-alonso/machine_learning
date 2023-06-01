from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List, Tuple

import numpy as np
import pandas as pd

from ..loss_functions.factory import LossFunctionFactory


class Optimiser(Enum):
    gradient_descent = "gradient_descent"
    mini_batch_gradient_descent = "mini_batch_gradient_descent"
    stochastic_gradient_descent = "stochastic_gradient_descent"


class Model(ABC):
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        self.weights = None
        self.bias = None
        self.history = {}

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        pass


class GradientModel(Model):
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ):
        super().__init__(X, y)

    def _mini_batch(
        self, X: pd.DataFrame, y: pd.Series, batch_size: int
    ) -> Tuple[np.array, np.array]:
        # Randomly shuffle the data
        idx = np.random.permutation(X.shape[0])
        X = X.iloc[idx]
        y = y.iloc[idx]

        # Split the data into batches
        n_batches = X.shape[0] // batch_size
        X_batches = np.array_split(X, n_batches)
        y_batches = np.array_split(y, n_batches)

        return X_batches, y_batches

    def _stochastic(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.array, np.array]:
        return self._mini_batch(X, y, 1)

    @abstractmethod
    def optimisation_loop(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass


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
