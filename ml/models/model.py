from abc import ABC, abstractmethod

import pandas as pd


class Model(ABC):
    def __init__(self):
        self.weights = None
        self.history = {}

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        pass
