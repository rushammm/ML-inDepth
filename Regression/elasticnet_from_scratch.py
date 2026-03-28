import numpy as np

class ElasticNet:
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter

    def _soft_threshold(self, rho, lam):
        if rho > lam:
            return rho - lam
        elif rho < -lam:
            return rho + lam
        return 0.0

    def fit(self, X, y):
        n, p = X.shape
        self.coef_ = np.zeros(p)
        self.intercept_ = np.mean(y)
        y = y - self.intercept_

        for _ in range(self.max_iter):
            for j in range(p):
                
                r = y - X @ self.coef_ + X[:, j] * self.coef_[j]
                # r = actual_values - prediction_using_all_features + feature_j_contribution

                # update coefficient
                rho = X[:, j] @ r
                lam_l1 = n * self.alpha * self.l1_ratio
                lam_l2 = n * self.alpha * (1 - self.l1_ratio)

                self.coef_[j] = self._soft_threshold(rho, lam_l1) / (
                    X[:, j] @ X[:, j] + lam_l2
                )
        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        y_pred = self.predict(X)
        return 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

# testing

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_california_housing

    # load the dataset
    housing = fetch_california_housing()

    X = housing.data[:, :5]
    y = housing.target
    # scale the features
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    model = ElasticNet(alpha=0.01, l1_ratio=0.5)
    model.fit(X, y)

    pred = model.predict(X)
    mse = np.mean((pred - y) ** 2)

    print(f"Weights: {model.coef_}")
    print(f"MSE: {mse:.4f}")

    # feature weights visualization
    feature_names = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population"]
    plt.barh(feature_names, model.coef_, color="coral")
    plt.axvline(x=0, color="black", linestyle="-", lw=0.5)
    plt.xlabel("Weight")
    plt.title("ElasticNet: Feature Weights (α=0.01)")
    plt.tight_layout()
    plt.show()

