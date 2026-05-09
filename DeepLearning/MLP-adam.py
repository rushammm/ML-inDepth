import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(a):
    # a is already sigmoid(z), so derivative is a * (1 - a)
    return a * (1 - a)

class MLP:
    def __init__(self, n_input, n_hidden, n_output, lr=0.01,
                 beta1=0.9, beta2=0.999, eps=1e-8):
        # random small weights, zero biases
        self.W1 = np.random.randn(n_input,  n_hidden) * 0.5
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = np.random.randn(n_hidden, n_output) * 0.5
        self.b2 = np.zeros((1, n_output))
        self.lr = lr

        # adam hyperparameters
        self.beta1 = beta1   # decay rate for direction memory (m)
        self.beta2 = beta2   # decay rate for magnitude memory (v)
        self.eps   = eps     # tiny number to prevent /0
        self.t     = 0       # timestep counter (for bias correction)

        # FIRST moment: running avg of gradient (direction, like momentum)
        self.mW1 = np.zeros_like(self.W1)
        self.mb1 = np.zeros_like(self.b1)
        self.mW2 = np.zeros_like(self.W2)
        self.mb2 = np.zeros_like(self.b2)

        # SECOND moment: running avg of gradient squared (magnitude, for per-weight LR)
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)

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

        # ADAM UPDATE
        self.t += 1
        b1, b2, eps, lr = self.beta1, self.beta2, self.eps, self.lr

        # 1) update first moment (direction)
        self.mW2 = b1 * self.mW2 + (1 - b1) * dW2
        self.mb2 = b1 * self.mb2 + (1 - b1) * db2
        self.mW1 = b1 * self.mW1 + (1 - b1) * dW1
        self.mb1 = b1 * self.mb1 + (1 - b1) * db1

        # 2) update second moment (magnitude — gradient squared)
        self.vW2 = b2 * self.vW2 + (1 - b2) * (dW2 ** 2)
        self.vb2 = b2 * self.vb2 + (1 - b2) * (db2 ** 2)
        self.vW1 = b2 * self.vW1 + (1 - b2) * (dW1 ** 2)
        self.vb1 = b2 * self.vb1 + (1 - b2) * (db1 ** 2)

        # 3) bias correction — early steps under-estimate moments since they start at 0
        bc1 = 1 - b1 ** self.t
        bc2 = 1 - b2 ** self.t

        # 4) step: lr * (corrected m) / (sqrt(corrected v) + eps)
        self.W2 -= lr * (self.mW2 / bc1) / (np.sqrt(self.vW2 / bc2) + eps)
        self.b2 -= lr * (self.mb2 / bc1) / (np.sqrt(self.vb2 / bc2) + eps)
        self.W1 -= lr * (self.mW1 / bc1) / (np.sqrt(self.vW1 / bc2) + eps)
        self.b1 -= lr * (self.mb1 / bc1) / (np.sqrt(self.vb1 / bc2) + eps)
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

    model = MLP(n_input=2, n_hidden=4, n_output=1, lr=0.05)
    model.train(X, y, epochs=10000)

    print("\nFinal predictions:")
    preds = model.forward(X)
    for xi, yi, pi in zip(X, y, preds):
        print(f"  {xi} -> predicted {pi[0]:.3f}  (truth: {yi[0]})")
