import numpy as np


class PCA:

    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components = None      # the top-k eigenvectors (our new axes)
        self.mean = None            # need this to center new data too
        self.explained_variance_ratio = None  # how much variance each PC captures

    def fit(self, X):
        # Step 1: center the data (subtract mean of each feature)
        self.mean = X.mean(axis=0)
        X_centered = X - self.mean

        # Step 2: covariance matrix
        # rowvar=False because each row is a sample, each column is a feature
        cov = np.cov(X_centered, rowvar=False)

        # Step 3: eigendecomposition
        # eigh is for symmetric matrices (cov is symmetric) — faster + stable
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Step 4: sort by eigenvalue descending, pick top k
        # eigh returns ascending, so we reverse
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        # keep top n_components eigenvectors as our new axes
        self.components = eigenvectors[:, :self.n_components]

        # variance each kept component captures, as a fraction of total
        self.explained_variance_ratio = eigenvalues[:self.n_components] / eigenvalues.sum()

    def transform(self, X):
        # Step 5: project centered data onto the new axes
        X_centered = X - self.mean
        return X_centered @ self.components

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# iris: 4 features → reduce to 2 so we can plot it
data = load_iris()
X, y = data.data, data.target

# ALWAYS scale before PCA — PCA is variance-based, so unit differences distort results
X_scaled = StandardScaler().fit_transform(X)

model = PCA(n_components=2)
X_reduced = model.fit_transform(X_scaled)

print(f"Original shape:  {X.shape}")
print(f"Reduced shape:   {X_reduced.shape}")
print(f"PC1 variance:    {model.explained_variance_ratio[0]*100:.2f}%")
print(f"PC2 variance:    {model.explained_variance_ratio[1]*100:.2f}%")
print(f"Total captured:  {model.explained_variance_ratio.sum()*100:.2f}%")

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis', s=40, alpha=0.8)
plt.xlabel('PC1 (most variance)')
plt.ylabel('PC2 (2nd most variance)')
plt.title('PCA from scratch — Iris 4D → 2D')
plt.show()
