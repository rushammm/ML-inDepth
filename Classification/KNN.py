import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        return [self._predict_single(x) for x in np.array(X)]

    def _predict_single(self, x):
        # compute distances from x to all training points
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

        # get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]

        # get their labels
        k_labels = self.y_train[k_indices]

        # return the most common label
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]


# --- simple demo ---
if __name__ == "__main__":
    # training data: 2 features, 2 classes (0 and 1)
    X_train = [[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]]
    y_train = [0, 0, 0, 1, 1, 1]

    # test points
    X_test = [[2, 2], [7, 6], [5, 4]]

    knn = KNN(k=3)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)

    for point, pred in zip(X_test, predictions):
        print(f"Point {point} -> Class {pred}")
