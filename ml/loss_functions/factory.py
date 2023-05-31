from .classification import *
from .regression import *


class LossFunctionFactory:
    @staticmethod
    def create_loss_function(loss_function_name: str):
        if loss_function_name == "mse":
            return MeanSquaredError()
        else:
            raise ValueError("Unknown loss function: {}".format(loss_function_name))
