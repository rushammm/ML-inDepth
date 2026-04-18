import numpy as np


class DBSCAN:

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps                  # neighborhood radius
        self.min_samples = min_samples  # min neighbors to be a "core" point
        self.labels = None

    def fit(self, X):
        n = len(X)
        self.labels = np.full(n, -1)    # -1 = noise by default
        cluster_id = 0

        for i in range(n):
            # skip if already assigned to a cluster
            if self.labels[i] != -1:
                continue

            neighbors = self._neighbors(X, i)

            # not enough neighbors → leave as noise (-1)
            if len(neighbors) < self.min_samples:
                continue

            # core point → start a new cluster and expand
            self._expand(X, i, neighbors, cluster_id)
            cluster_id += 1

    def _neighbors(self, X, i):
        # indices of all points within eps of point i
        dists = np.sqrt(((X - X[i]) ** 2).sum(axis=1))
        return list(np.where(dists <= self.eps)[0])

    def _expand(self, X, i, neighbors, cluster_id):
        # claim the starting point
        self.labels[i] = cluster_id

        # walk through neighbors, and neighbors-of-neighbors if they're also core
        queue = list(neighbors)
        while queue:
            j = queue.pop(0)

            if self.labels[j] == -1:            # was noise → flip to cluster
                self.labels[j] = cluster_id

                # if j is also core, add ITS neighbors to the queue
                j_neighbors = self._neighbors(X, j)
                if len(j_neighbors) >= self.min_samples:
                    queue.extend(j_neighbors)


from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# moons dataset — two crescent shapes (K-means fails hard on this)
X, y = make_moons(n_samples=200, noise=0.05, random_state=42)

# train
model = DBSCAN(eps=0.2, min_samples=5)
model.fit(X)

# plot
plt.scatter(X[:, 0], X[:, 1], c=model.labels, cmap='viridis', s=40, alpha=0.7)
plt.title('DBSCAN (from scratch) — noise points shown in dark')
plt.show()
