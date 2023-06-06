from .loss_function import LossFunction

# class SoftmaxCrossEntropy(LossFunction):
#     """Calculates the softmax cross entropy between the true and predicted values.

#     Args:
#         y_true (np.array): The true values.
#         y_pred (np.array): The predicted values.

#     Returns:
#         float: The softmax cross entropy.
#     """

#     def __call__(self, y_true: np.array, y_pred: np.array) -> float:
#         return -np.mean(y_true * np.log(y_pred))


# class BinaryCrossEntropy(LossFunction):
#     """Calculates the binary cross entropy between the true and predicted values.

#     Args:
#         y_true (np.array): The true values.
#         y_pred (np.array): The predicted values.

#     Returns:
#         float: The binary cross entropy.
#     """

#     def __call__(self, y_true: np.array, y_pred: np.array) -> float:
#         return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# class CategoricalCrossEntropy(LossFunction):
#     """Calculates the categorical cross entropy between the true and predicted values.

#     Args:
#         y_true (np.array): The true values.
#         y_pred (np.array): The predicted values.

#     Returns:
#         float: The categorical cross entropy.
#     """

#     def __call__(self, y_true: np.array, y_pred: np.array) -> float:
#         return -np.mean(y_true * np.log(y_pred))
