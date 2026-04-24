import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(a):
    # a is already sigmoid(z), so derivative is a * (1 - a)
    return a * (1 - a)

class MLP:
    def __init__(self, n_input, n_hidden, n_output, lr=0.1):
        # random small weights, zero biases 

        self.W1 = np.random.randn(n_input,  n_hidden) * 0.5
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = np.random.randn(n_hidden, n_output) * 0.5
        self.b2 = np.zeros((1, n_output))
        self.lr = lr

    # forward propagation 
    def forward(self, X):
        # layer 1: input -> hidden
        self.Z1 = X @ self.W1 + self.b1          # weighted sum
        self.A1 = sigmoid(self.Z1)                # activation

        # layer 2: hidden -> output
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = sigmoid(self.Z2)                # final prediction
        return self.A2

    # backpropagation 
    def backward(self, X, y):
        m = X.shape[0]  # number of samples

        # OUTPUT LAYER: how wrong was the prediction? 
        dZ2 = self.A2 - y                          # error at output
        dW2 = (self.A1.T @ dZ2) / m                # gradient for W2
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # HIDDEN LAYER: propagate blame backward 
        dA1 = dZ2 @ self.W2.T                      # blame on hidden activations
        dZ2_hidden = dA1 * sigmoid_derivative(self.A1)   # apply activation derivative
        dW1 = (X.T @ dZ2_hidden) / m
        db1 = np.sum(dZ2_hidden, axis=0, keepdims=True) / m

        #  UPDATE WEIGHTS (gradient descent) 
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    #  training loop 
    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y)

            if epoch % 1000 == 0:
                # binary cross-entropy loss
                loss = -np.mean(y * np.log(self.A2 + 1e-8) +
                                (1 - y) * np.log(1 - self.A2 + 1e-8))
                print(f"Epoch {epoch:5d} | Loss: {loss:.4f}")


if __name__ == "__main__":
    np.random.seed(42)

    # the XOR problem — what a single perceptron CAN'T solve
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

    y = np.array([[0],
                  [1],
                  [1],
                  [0]])

    model = MLP(n_input=2, n_hidden=4, n_output=1, lr=0.5)
    model.train(X, y, epochs=10000)

    print("\nFinal predictions:")
    preds = model.forward(X)
    for xi, yi, pi in zip(X, y, preds):
        print(f"  {xi} -> predicted {pi[0]:.3f}  (truth: {yi[0]})")
