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

# identity matrix (don't penalize the intercept)
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

# Feature Weights visualization
feature_names = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population"]
plt.barh(feature_names, w[1:], color="steelblue")
plt.axvline(x=0, color="black", linestyle="-", lw=0.5)
plt.xlabel("Weight")
plt.title("Ridge: Feature Weights (λ=1)")
plt.tight_layout()
plt.show()
