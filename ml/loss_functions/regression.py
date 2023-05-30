from loss_functions import LossFunction


class MeanSquaredError(LossFunction):
    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        return ((y_true - y_pred) ** 2).mean()
