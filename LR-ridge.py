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

# identity matrix (don't penalize bias)
I = np.eye(X.shape[1])
I[0, 0] = 0

# train (λ = 1)
lam = 1.0
w = np.linalg.inv(X.T @ X + lam * I) @ X.T @ y


# predict & evaluate
pred = X @ w
mse = np.mean((pred - y) ** 2)
print(f"Weights: {w}")
print(f"MSE: {mse:.4f}")

# Predicted vs Actual
plt.scatter(y, pred, alpha=0.3, s=10)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Predicted vs Actual")
plt.show()
