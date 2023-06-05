import pytest

from ml.loss_functions.factory import LossFunctionFactory


def test_get_loss_function_mse():
    mse = LossFunctionFactory.create_loss_function("mse")
    assert mse.__class__.__name__ == "MeanSquaredError"


def test_get_loss_function_mae():
    mae = LossFunctionFactory.create_loss_function("mae")
    assert mae.__class__.__name__ == "MeanAbsoluteError"


def test_get_loss_function_invalid():
    with pytest.raises(ValueError):
        LossFunctionFactory.create_loss_function("invalid_loss_function")
