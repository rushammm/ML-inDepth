import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k

# knn is a lazy algo , it stores data only. doesnt have actual training 
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

# wrapper func for predict single (so we dont have to loop through each point manually)
    def predict(self, X):
        return [self._predict_single(x) for x in np.array(X)]

    def _predict_single(self, x):
    # compute euclidean distances from x to all training points
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

        # indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]

        # their labels
        k_labels = self.y_train[k_indices]

        # return the most common label
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]


if __name__ == "__main__":
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    data = load_digits()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    knn = KNN(k=5)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
