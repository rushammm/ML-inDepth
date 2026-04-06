import numpy as np
from collections import Counter

# reusing decision tree code

class DecisionTree:

    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.root = None

    def gini(self, y):
        counts = Counter(y)
        impurity = 1
        for label in counts:
            prob = counts[label] / len(y)
            impurity -= prob ** 2
        return impurity

    def best_split(self, X, y):
        best_gini = float('inf')
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                y_left = y[left_mask]
                y_right = y[~left_mask]
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                n = len(y)
                gini_split = (len(y_left) / n) * self.gini(y_left) + \
                             (len(y_right) / n) * self.gini(y_right)
                if gini_split < best_gini:
                    best_gini = gini_split
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < 2:
            return Counter(y).most_common(1)[0][0]
        feature, threshold = self.best_split(X, y)
        left_mask = X[:, feature] <= threshold
        left = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self.build_tree(X[~left_mask], y[~left_mask], depth + 1)
        return {'feature': feature, 'threshold': threshold, 'left': left, 'right': right}

    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    def predict_one(self, x, node):
        if not isinstance(node, dict):
            return node
        if x[node['feature']] <= node['threshold']:
            return self.predict_one(x, node['left'])
        return self.predict_one(x, node['right'])

    def predict(self, X):
        return np.array([self.predict_one(x, self.root) for x in X])


class RandomForest:

    def __init__(self, n_trees=100, max_depth=10):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def bootstrap_sample(self, X, y):
        indices = np.random.choice(len(X), len(X), replace=True) # picks rows randomly from the data
        return X[indices], y[indices]

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            X_sample, y_sample = self.bootstrap_sample(X, y) # new sample created randomly from rows 
            tree = DecisionTree(max_depth=self.max_depth) # decision tree trained on the new sample
            tree.fit(X_sample, y_sample)
            self.trees.append(tree) # tree stored

    def predict(self, X):
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        # majority vote for each sample
        return np.array([Counter(col).most_common(1)[0][0] for col in all_preds.T])


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForest(n_trees=100, max_depth=10)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
