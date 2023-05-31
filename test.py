# Import the data
import numpy as np
import pandas as pd

from ml.models.linear_regression import LinearRegression

# Import the data
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

X = pd.DataFrame(data)
y = pd.Series(target)

# Create the model
model = LinearRegression(
    X,
    y,
    learning_rate=0.00000001,
    n_iter=1000,
    loss="mse",
    optimiser="stochastic_gradient_descent",
)

model.fit()

# Test the model
predictions = model.predict(X)

print("Predictions:", predictions)

model.plot_training_curve()
