import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingClassifier:
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.max_depth     = max_depth
        self.trees         = []
        self.initial_pred  = None
    
    # sigmoid to convert to probabilities
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def fit(self, X, y):
        #  initial prediction - since we need to have a starting point 
        mean = np.mean(y)
        # log-odds cuz gbm work with log values not probabilities 
        self.initial_pred = np.log(mean / (1 - mean))

        # store initial pred in F
        F = np.full(len(y), self.initial_pred)
        
        for _ in range(self.n_estimators):
            
            # convert F into probability using sigmoid 
            probabilities = self.sigmoid(F)
            # calc residuals (actual - pred)
            residuals     = y - probabilities
            
            # train a tree on residuals = sees how wrong are we
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            
            # update F using the tree's predictions and learning rate
            F = F + self.learning_rate * tree.predict(X)
            
            # save tree
            self.trees.append(tree)
    
    def predict_proba(self, X):
        # start with initial prediction
        F = np.full(X.shape[0], self.initial_pred)
        
        # add each tree's contribution
        for tree in self.trees:
            F = F + self.learning_rate * tree.predict(X)
        
        return self.sigmoid(F)
    
    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load data
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")