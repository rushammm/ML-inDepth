import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# load the dataset
housing = fetch_california_housing()

X = housing.data[:, :5]
y = housing.target

# scale the features
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# add bias
X = np.column_stack([np.ones(len(X)), X])

def train_lasso(X, y, alpha, lr=0.0001, epochs=3000):
    w = np.zeros(X.shape[1])

    for _ in range(epochs):
        # predictions with current w
        predictions = X @ w

        # how wrong are we
        errors = predictions - y

        # gradient: which direction reduces error
        grad_error = (2 / len(X)) * X.T @ errors

        # gradient: which direction reduces slope size
        grad_penalty = alpha * np.sign(w)
        grad_penalty[0] = 0  # don't penalise intercept

        # take one small step
        w = w - lr * (grad_error + grad_penalty)

    return w

w_lasso = train_lasso(X, y, alpha=0.01)
pred_lasso = X @ w_lasso
mse_lasso = np.mean((pred_lasso - y) ** 2)

print(f"Weights: {w_lasso}")
print(f"MSE: {mse_lasso:.4f}")

# Feature Weights visualization
feature_names = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population"]
plt.barh(feature_names, w_lasso[1:], color="coral")
plt.axvline(x=0, color="black", linestyle="-", lw=0.5)
plt.xlabel("Weight")
plt.title("Lasso: Feature Weights (α=0.01)")
plt.tight_layout()
plt.show()
