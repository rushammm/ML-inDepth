import numpy as np
from collections import Counter

class DecisionTree:
    
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
    
    # gini Impurity — measures how mixed a node is
    def gini(self, y):
        counts = Counter(y)
        impurity = 1
        for label in counts:
            prob = counts[label] / len(y)
            impurity -= prob ** 2
        return impurity
    
    # find the best split for a node
    def best_split(self, X, y):
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        # try every feature - brute force 
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            
            # try every unique value as a threshold
            for threshold in thresholds:
                left_mask  = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                y_left  = y[left_mask]
                y_right = y[right_mask]
                
                # skip if one side is empty
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                
                # weighted gini of this split
                n = len(y)
                gini_split = (len(y_left) / n)  * self.gini(y_left) + \
                             (len(y_right) / n) * self.gini(y_right)
                
                # best split
                if gini_split < best_gini:
                    best_gini      = gini_split
                    best_feature   = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    # build the tree recursively
    def build_tree(self, X, y, depth=0):
        n_samples = len(y)
        n_classes = len(np.unique(y))
        
        # stopping conditions (if tree reaches max-depth, only one class left, few samples left)
        if depth >= self.max_depth or n_classes == 1 or n_samples < self.min_samples_split:
            # return leaf node — most common class
            return Counter(y).most_common(1)[0][0]
        
        # find best split
        feature, threshold = self.best_split(X, y)
        
        # split data
        left_mask  = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        # recursively build left and right subtrees
        left  = self.build_tree(X[left_mask],  y[left_mask],  depth + 1)
        right = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        
        # return node as dictionary
        return {
            'feature'   : feature,
            'threshold' : threshold,
            'left'      : left,
            'right'     : right
        }
    
    # fit() — just builds the tree
    def fit(self, X, y):
        self.root = self.build_tree(X, y)
    
    # predict a single sample
    def predict_one(self, x, node):
        # If leaf node — return class
        if not isinstance(node, dict):
            return node
        
        # go left or right based on threshold
        if x[node['feature']] <= node['threshold']:
            return self.predict_one(x, node['left'])
        else:
            return self.predict_one(x, node['right'])
    
    # predict all samples
    def predict(self, X):
        return np.array([self.predict_one(x, self.root) for x in X])


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# load data
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train
model = DecisionTree(max_depth=5, min_samples_split=2)
model.fit(X_train, y_train)

# predict
predictions = model.predict(X_test)

# accuracy
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")