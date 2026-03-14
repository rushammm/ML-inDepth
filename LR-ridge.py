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

# --- Visualizations ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# 1. Predicted vs Actual
axes[0].scatter(y, pred, alpha=0.3, s=10)
axes[0].plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
axes[0].set_xlabel("Actual")
axes[0].set_ylabel("Predicted")
axes[0].set_title("Predicted vs Actual")

# 2. Residuals distribution
residuals = y - pred
axes[1].hist(residuals, bins=50, edgecolor="black", alpha=0.7)
axes[1].axvline(x=0, color="r", linestyle="--")
axes[1].set_xlabel("Residual")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Residuals Distribution")

# 3. Feature weights (excluding bias)
feature_names = housing.feature_names[:5]
axes[2].barh(feature_names, w[1:], color="steelblue")
axes[2].axvline(x=0, color="black", linestyle="-", lw=0.5)
axes[2].set_xlabel("Weight")
axes[2].set_title("Feature Weights (Ridge)")

plt.tight_layout()
plt.show()
