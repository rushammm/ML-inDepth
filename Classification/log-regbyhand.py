import numpy as np

class LogisticRegression:
    
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    # define sigmoid function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    # train the model
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # initialize weights and bias to zero
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # gradient descent
        for _ in range(self.epochs):
            
            # forward pass ( linear eq (with bias n weights) + sigmoid call)
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)
            
            # compute gradients (how much weights n bias contributed to error)
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # update weights
            self.weights = self.weights - self.lr * dw
            self.bias    = self.bias    - self.lr * db
    
    #  predict probabilities
    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    #  predict class (0 or 1) -- threshold applied 
    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)


# breast cancer dataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(learning_rate=0.01, epochs=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy:.4f}")
