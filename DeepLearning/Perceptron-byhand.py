import numpy as np

class Perceptron:
    
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr      = learning_rate
        self.epochs  = epochs
        self.weights = None
        self.bias    = None
    
    # step function 
    def step_function(self, z):
        return np.where(z >= 0, 1, 0)
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # initialize weights and bias to zero
        self.weights = np.zeros(n_features)
        self.bias    = 0
        
        # training loop
        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                
                # Forward pass 
                z         = np.dot(x_i, self.weights) + self.bias
                predicted = self.step_function(z)
                
                # Perceptron update rule
                update         = self.lr * (y[idx] - predicted)
                self.weights  += update * x_i
                self.bias     += update
    
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.step_function(z)



from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# create linearly separable dataset
X, y = make_classification(n_samples=500, n_features=2, 
                            n_redundant=0, n_informative=2,
                            random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# scale features
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# train
model = Perceptron(learning_rate=0.01, epochs=1000)
model.fit(X_train, y_train)

# predict
predictions = model.predict(X_test)

# accuracy
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")