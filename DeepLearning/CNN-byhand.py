import numpy as np


def conv2d(image, filter_):
    """Single-channel 2D conv. (H, W) * (kH, kW) -> (H-kH+1, W-kW+1)."""
    H, W = image.shape
    kH, kW = filter_.shape
    out_H = H - kH + 1
    out_W = W - kW + 1

    output = np.zeros((out_H, out_W))
    for i in range(out_H):
        for j in range(out_W):
            patch = image[i:i+kH, j:j+kW]
            output[i, j] = np.sum(patch * filter_)
    return output


def conv2d_multi(image, filters):
    """Apply multiple filters to one image, stack feature maps along channel axis."""
    feature_maps = [conv2d(image, f) for f in filters]
    return np.stack(feature_maps, axis=0)


def relu(X):
    """Element-wise max(0, x)."""
    return np.maximum(0, X)


def maxpool_forward(X, pool_size=2, stride=2):
    """Max pool each channel independently. (C, H, W) -> (C, out_H, out_W)."""
    C, H, W_in = X.shape
    out_H = (H - pool_size) // stride + 1
    out_W = (W_in - pool_size) // stride + 1
    output = np.zeros((C, out_H, out_W))

    for c in range(C):
        for i in range(out_H):
            for j in range(out_W):
                top  = i * stride
                left = j * stride
                window = X[c, top:top+pool_size, left:left+pool_size]
                output[c, i, j] = np.max(window)
    return output


def flatten(X):
    """(C, H, W) -> (C*H*W,)."""
    return X.reshape(-1)


def dense_forward(X, W, b):
    """Fully-connected layer: X @ W + b."""
    return X @ W + b


def softmax(z):
    """Softmax with max-subtraction for numerical stability."""
    z_shifted = z - np.max(z)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z)


def conv_forward(X, W, b, padding=0, stride=1):
    """
    Multi-channel conv layer forward pass.
      X: (in_channels, H, W)
      W: (out_channels, in_channels, kH, kW)
      b: (out_channels,)
    Returns: (out_channels, out_H, out_W)
    """
    in_channels, H, W_in = X.shape
    out_channels, _, kH, kW = W.shape

    if padding > 0:
        X_padded = np.pad(X, ((0, 0), (padding, padding), (padding, padding)))
    else:
        X_padded = X

    out_H = (H + 2 * padding - kH) // stride + 1
    out_W = (W_in + 2 * padding - kW) // stride + 1

    output = np.zeros((out_channels, out_H, out_W))

    for c in range(out_channels):
        for i in range(out_H):
            for j in range(out_W):
                top  = i * stride
                left = j * stride
                patch = X_padded[:, top:top+kH, left:left+kW]
                output[c, i, j] = np.sum(patch * W[c]) + b[c]

    return output


class CNN:
    """
    Architecture:
      conv1 (1 -> 16, 3x3, pad=1)  -> ReLU -> maxpool 2x2  ->  (16, 14, 14)
      conv2 (16 -> 32, 3x3, pad=1) -> ReLU -> maxpool 2x2  ->  (32, 7, 7)
      flatten  ->  dense (1568 -> 10)  ->  softmax
    """

    def __init__(self):
        self.W1 = np.random.randn(16, 1, 3, 3) * 0.1
        self.b1 = np.zeros(16)

        self.W2 = np.random.randn(32, 16, 3, 3) * 0.1
        self.b2 = np.zeros(32)

        self.W3 = np.random.randn(1568, 10) * 0.1
        self.b3 = np.zeros(10)

    def forward(self, X):
        # X shape: (1, 28, 28)
        z1 = conv_forward(X, self.W1, self.b1, padding=1, stride=1)   # (16, 28, 28)
        a1 = relu(z1)
        p1 = maxpool_forward(a1, pool_size=2, stride=2)                # (16, 14, 14)

        z2 = conv_forward(p1, self.W2, self.b2, padding=1, stride=1)  # (32, 14, 14)
        a2 = relu(z2)
        p2 = maxpool_forward(a2, pool_size=2, stride=2)                # (32, 7, 7)

        f  = flatten(p2)                                               # (1568,)
        z3 = dense_forward(f, self.W3, self.b3)                        # (10,)
        probs = softmax(z3)
        return probs


if __name__ == "__main__":
    # single filter on a tiny image
    image = np.array([
        [1, 5, 5],
        [1, 5, 5],
        [1, 5, 5],
    ], dtype=np.float32)

    vertical_edge = np.array([
        [-1, 1],
        [-1, 1],
    ], dtype=np.float32)

    feature_map = conv2d(image, vertical_edge)
    print("Input image:")
    print(image)
    print("\nVertical-edge filter:")
    print(vertical_edge)
    print("\nFeature map (high values = edge found):")
    print(feature_map)

    # multiple filters
    horizontal_edge = np.array([
        [-1, -1],
        [ 1,  1],
    ], dtype=np.float32)

    blur = np.array([
        [0.25, 0.25],
        [0.25, 0.25],
    ], dtype=np.float32)

    filters = np.stack([vertical_edge, horizontal_edge, blur], axis=0)
    feature_maps = conv2d_multi(image, filters)
    names = ["vertical edge", "horizontal edge", "blur"]

    print(f"\n\nWith {filters.shape[0]} filters -> output shape {feature_maps.shape}")
    for i, fm in enumerate(feature_maps):
        print(f"\nFeature map {i} ({names[i]}):")
        print(fm)

    # conv_forward with channels + padding + stride
    print("\n\n=== conv_forward demo ===")
    np.random.seed(0)
    X = np.random.randn(1, 6, 6)
    W = np.random.randn(4, 1, 3, 3) * 0.1
    b = np.zeros(4)

    out = conv_forward(X, W, b, padding=1, stride=1)
    print(f"input shape:  {X.shape}")
    print(f"filter shape: {W.shape}")
    print(f"output shape: {out.shape}  (padding=1 keeps spatial size)")

    out2 = conv_forward(X, W, b, padding=1, stride=2)
    print(f"with stride=2: {out2.shape}  (spatial halved)")

    # ReLU + pool + flatten
    print("\n\n=== ReLU + maxpool + flatten ===")
    after_relu = relu(out)
    print(f"after conv:    {out.shape}")
    print(f"after ReLU:    {after_relu.shape}  (min: {out.min():.3f} -> {after_relu.min():.3f})")

    after_pool = maxpool_forward(after_relu, pool_size=2, stride=2)
    print(f"after maxpool: {after_pool.shape}")

    after_flat = flatten(after_pool)
    print(f"after flatten: {after_flat.shape}")

    # dense + softmax
    print("\n\n=== dense + softmax ===")
    W_dense = np.random.randn(36, 10) * 0.1
    b_dense = np.zeros(10)
    logits = dense_forward(after_flat, W_dense, b_dense)
    probs  = softmax(logits)
    print(f"logits shape: {logits.shape}")
    print(f"probs shape:  {probs.shape}  (sum: {probs.sum():.4f})")
    print(f"predicted class: {np.argmax(probs)}")

    # full CNN forward pass
    print("\n\n=== full CNN forward pass ===")
    np.random.seed(7)
    fake_image = np.random.randn(1, 28, 28).astype(np.float32)
    model = CNN()
    output = model.forward(fake_image)
    print(f"input:  (1, 28, 28)")
    print(f"output: {output.shape}  (sum: {output.sum():.4f})")
    print(f"predicted class: {np.argmax(output)}")
