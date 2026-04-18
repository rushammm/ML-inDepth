import numpy as np


class Hierarchical:

    def __init__(self, k=3):
        self.k = k          # how many clusters we want at the end
        self.labels = None

    def fit(self, X):
        n = len(X)

        # start: every point is its own cluster
        # clusters is a list of lists of point indices
        clusters = [[i] for i in range(n)]

        # keep merging until we have k clusters left
        while len(clusters) > self.k:
            # find the two closest clusters
            a, b = self._closest_pair(X, clusters)

            # merge them (put b's points into a, remove b)
            clusters[a] = clusters[a] + clusters[b]
            clusters.pop(b)

        # assign a label to each point based on final cluster
        self.labels = np.zeros(n, dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            for point_idx in cluster:
                self.labels[point_idx] = cluster_id

    def _closest_pair(self, X, clusters):
        # dist between clusters = min dist between any two points
        best_dist = np.inf
        best_pair = (0, 1)

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = self._cluster_distance(X, clusters[i], clusters[j])
                if d < best_dist:
                    best_dist = d
                    best_pair = (i, j)

        return best_pair

    def _cluster_distance(self, X, c1, c2):
        # min pairwise distance between points in c1 and points in c2
        points1 = X[c1]
        points2 = X[c2]
        dists = np.sqrt(((points1[:, np.newaxis] - points2) ** 2).sum(axis=2))
        return dists.min()


from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# fake data with 3 clusters (small n because this is O(n^3))
X, y = make_blobs(n_samples=50, centers=3, cluster_std=0.8, random_state=42)

# train
model = Hierarchical(k=3)
model.fit(X)

# plot
plt.scatter(X[:, 0], X[:, 1], c=model.labels, cmap='viridis', s=40, alpha=0.7)
plt.title('Hierarchical Clustering (from scratch, single linkage)')
plt.show()
