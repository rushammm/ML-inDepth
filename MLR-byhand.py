import numpy as np
import matplotlib
matplotlib.use('Agg')  # saving to files as direct plots were crashing
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
# dataset
housing = fetch_california_housing()

# using the first 3 features for Multi Linear Regression
X = housing.data[:, :3]
y = housing.target


# adding column of 1's because we want to treat b (intercept) like a slope 
ones = np.ones((len(X), 1))
X = np.column_stack([ones, X])

# train 
w = np.linalg.inv(X.T @ X) @ X.T @ y

# predict
predictions = X @ w

# MSE
mse = np.mean((predictions - y) ** 2)

# r² score = shows how much variance the model explains
ss_res = np.sum((y - predictions) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - (ss_res / ss_tot)

# results
print(f"Weights (bias + {X.shape[1]-1} features): {w}")
print(f"MSE: {mse:.4f}")
print(f"R²:  {r2:.4f}")
print(f"\nSample predictions vs actual (first 5):")
for i in range(5):
    print(f"  Pred: {predictions[i]:.2f}, Actual: {y[i]:.2f}")



print("Features used:", housing.feature_names[:3])

# 3D scatter plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# plot actual data points 
scatter = ax.scatter(X[:, 1], X[:, 2], X[:, 3], c=y, cmap='viridis', alpha=0.5)

ax.set_xlabel('MedInc')
ax.set_ylabel('HouseAge')
ax.set_zlabel('AveRooms')
ax.set_title('California Housing - 3D Scatter (color = price)')

plt.colorbar(scatter, label='House Price')
plt.savefig('3d_scatter.png', dpi=150)
plt.close()
print("Saved: 3d_scatter.png")

# predicted vs actual plot
plt.figure(figsize=(8, 6))
plt.scatter(y, predictions, alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Perfect prediction')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title(f'Predicted vs Actual (R² = {r2:.4f})')
plt.legend()
plt.savefig('predicted_vs_actual.png', dpi=150)
plt.close()
print("Saved: predicted_vs_actual.png")
 