import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, C=1.0, epochs=1000):
        self.lr = learning_rate
        self.C = C
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    # train the model
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # convert labels to -1 and 1 (SVM uses -1/1 not 0/1)
        y_ = np.where(y <= 0, -1, 1)
        
        # initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # gradient descent
        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                
                # check if point is correctly classified with margin
                condition = y_[idx] * (np.dot(x_i, self.weights) + self.bias) >= 1
                
                if condition:
                    # point is outside margin — only regularization update
                    self.weights = self.weights - self.lr * (2 * self.weights)
                    
                else:
                    # point is inside or wrong side of margin — full update
                    self.weights = self.weights - self.lr * (2 * self.weights - self.C * np.dot(x_i, y_[idx]))
                    self.bias    = self.bias    + self.lr * self.C * y_[idx]
    
    # predict
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return np.sign(z)  # returns -1 or 1

# breast cancer dataset from sklearn 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale features - IMP for svm
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# train
model = SVM(learning_rate=0.001, C=1.0, epochs=1000)
model.fit(X_train, y_train)

# predict
predictions = model.predict(X_test)

# convert predictions back to 0/1 for accuracy
predictions = np.where(predictions == -1, 0, 1)

# accuracy
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")




## In SVM, labels are -1 and +1 (not 0 and 1)
# The model gives a raw score using wX + b
# - Positive score = model thinks class +1
# - Negative score = model thinks class -1

# To check if prediction is correct, we multiply actual × predicted (score):
# - Same sign (both + or both -) → correct prediction → positive result
# - Different sign → wrong prediction → negative result

# We check >= 1 not just > 0 because:
# SVM wants points to be BEYOND the margin (at +1 or -1), not just past the boundary (0)
# - result >= 1 → correct AND outside margin → small weight update
# - result < 1  → wrong OR too close to boundary → big weight update