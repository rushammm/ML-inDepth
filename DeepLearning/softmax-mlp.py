import numpy as np

# activations 
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def softmax(z):
    # subtract max for numerical stability (prevents overflow in exp)
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


#  one-hot encoding 
def one_hot(y, n_classes):
    # turns labels like [0, 2, 1] into [[1,0,0],[0,0,1],[0,1,0]]
    m = y.shape[0]
    onehot = np.zeros((m, n_classes))
    onehot[np.arange(m), y] = 1
    return onehot


class MLP:
    def __init__(self, n_input, n_hidden, n_output, lr=0.1):
        self.W1 = np.random.randn(n_input,  n_hidden) * 0.5
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = np.random.randn(n_hidden, n_output) * 0.5
        self.b2 = np.zeros((1, n_output))
        self.lr = lr

    # forward propagation
    def forward(self, X):
        # layer 1: input -> hidden  (sigmoid — same as before)
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = sigmoid(self.Z1)

        # layer 2: hidden -> output (Softmax — NEW)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2

    # backpropagation
    def backward(self, X, y_onehot):
        m = X.shape[0]

        # OUTPUT LAYER
        # magic: when softmax pairs with cross-entropy,
        # the gradient simplifies to (predicted - true), same shape as sigmoid+BCE
        dZ2 = self.A2 - y_onehot
        dW2 = (self.A1.T @ dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # HIDDEN LAYER
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * sigmoid_derivative(self.A1)
        dW1 = (X.T @ dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # UPDATE WEIGHTS
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    # training loop
    def train(self, X, y_onehot, epochs=2000):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y_onehot)

            if epoch % 200 == 0:
                # categorical cross-entropy loss
                loss = -np.mean(np.sum(y_onehot * np.log(self.A2 + 1e-8), axis=1))
                # accuracy
                preds = np.argmax(self.A2, axis=1)
                truth = np.argmax(y_onehot, axis=1)
                acc = np.mean(preds == truth)
                print(f"Epoch {epoch:5d} | Loss: {loss:.4f} | Acc: {acc:.2%}")


if __name__ == "__main__":
    np.random.seed(42)

    from sklearn.datasets import make_blobs

    # 3 classes, 2 features, 300 samples
    X, y = make_blobs(n_samples=300, centers=3, n_features=2,
                      cluster_std=1.2, random_state=42)
    y_onehot = one_hot(y, n_classes=3)

    model = MLP(n_input=2, n_hidden=8, n_output=3, lr=0.1)
    model.train(X, y_onehot, epochs=2000)

    print("\nSample predictions:")
    preds = model.forward(X[:5])
    for xi, true_label, prob_dist in zip(X[:5], y[:5], preds):
        predicted_class = np.argmax(prob_dist)
        print(f"  {xi.round(2)} -> probs: {prob_dist.round(3)}  "
              f"predicted: {predicted_class}  truth: {true_label}")
