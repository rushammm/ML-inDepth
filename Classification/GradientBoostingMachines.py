import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingClassifier:
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.max_depth     = max_depth
        self.trees         = []
        self.initial_pred  = None
    
    # Sigmoid to convert raw scores to probabilities
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def fit(self, X, y):
        # Step 1 — initial prediction (log odds of the mean)
        mean = np.mean(y)
        self.initial_pred = np.log(mean / (1 - mean))
        
        # Start with initial prediction for all samples
        F = np.full(len(y), self.initial_pred)
        
        for _ in range(self.n_estimators):
            
            # Step 2 — compute residuals (negative gradient)
            probabilities = self.sigmoid(F)
            residuals     = y - probabilities
            
            # Step 3 — fit a tree to the residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            
            # Step 4 — update predictions
            F = F + self.learning_rate * tree.predict(X)
            
            # Save tree
            self.trees.append(tree)
    
    def predict_proba(self, X):
        # Start with initial prediction
        F = np.full(X.shape[0], self.initial_pred)
        
        # Add each tree's contribution
        for tree in self.trees:
            F = F + self.learning_rate * tree.predict(X)
        
        return self.sigmoid(F)
    
    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)


# ---- Usage ----
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