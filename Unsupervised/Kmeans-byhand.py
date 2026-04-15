import numpy as np

class KMeans:

    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None

    def fit(self, X):
        # initialize K random centroids
        # pick K random rows from X as starting centroids
        random_centroids = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iters):
            # assign each point to nearest centroid
            # for each point, calc distance to each centroid
            labels = self._assign_clusters(X)

            # Step 3: update centroids (mean of each cluster)
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])

            # Step 4: check convergence (if centroids didn't move, break)
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

    def _assign_clusters(self, X):
        # distance from every point to every centroid
        distances = np.sqrt(((X[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))
        # pick the centroid with the smallest distance for each point
        return np.argmin(distances, axis=1)

    def predict(self, X):
        return self._assign_clusters(X)


from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# generate fake data with 3 clusters
X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42)

# train
model = KMeans(k=3, max_iters=100)
model.fit(X)

# predict
labels = model.predict(X)

# plot
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=40, alpha=0.7)
plt.scatter(model.centroids[:, 0], model.centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('K-Means Clustering (from scratch)')
plt.legend()
plt.show()
