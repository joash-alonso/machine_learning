from typing import List

import numpy as np
from base.optimisers import Optimiser
from .loss_functions.regression import MeanSquaredError


class GradientDescent(Optimiser):
    def __init__(
        self, features: List[str], learning_rate: float = 0.001, n_iter: int = 1000, loss_function: str = "mse"
    ):
        super().__init__(learning_rate)
        self.features = features
        self.parameters = {}
        self.n_iter = n_iter
        self.gradients = {}
        self.loss_function = loss_function

    def __initialise_paramaeters(self):
        for key in self.features:
            self.parameters[key] = 0

    def update_parameters(self):
        for step in range(self.n_iter):
            gradients = 
            for key in self.parameters.keys():
                self.parameters[key] -= self.learning_rate * self.gradients[key]
        return self.parameters, self.gradients

    def update_history(self, step, gradients):
        self.history[step] = gradients
