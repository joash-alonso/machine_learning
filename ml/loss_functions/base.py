from abc import ABC, abstractmethod

import numpy as np


class LossFunction(ABC):
    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        raise NotImplementedError("Loss function not implemented")
