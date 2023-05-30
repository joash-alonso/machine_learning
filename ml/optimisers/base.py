from abc import ABC, abstractmethod


class Optimiser(ABC):
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.history = {}

    @abstractmethod
    def update_parameters(self, parameters, gradients):
        raise NotImplementedError("update_parameters method not implemented")

    def update_history(self, step, gradients):
        self.history[step] = gradients
