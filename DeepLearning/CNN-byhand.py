import numpy as np


def conv2d(image, filter_):
    """
    Single-channel 2D convolution. Slides `filter_` over `image`,
    element-wise multiplies + sums at each position to produce a feature map.

    image  shape: (H, W)
    filter shape: (kH, kW)
    output shape: (H - kH + 1, W - kW + 1)
    """
    H, W = image.shape
    kH, kW = filter_.shape
    out_H = H - kH + 1
    out_W = W - kW + 1

    output = np.zeros((out_H, out_W))
    for i in range(out_H):
        for j in range(out_W):
            patch = image[i:i+kH, j:j+kW]      # the window the filter sits on
            output[i, j] = np.sum(patch * filter_)
    return output


def conv2d_multi(image, filters):
    """
    Run multiple filters over the same image. Each filter produces
    its own feature map; we stack them along a new "channels" axis.

    image    shape: (H, W)
    filters  shape: (num_filters, kH, kW)
    output   shape: (num_filters, H - kH + 1, W - kW + 1)
    """
    feature_maps = [conv2d(image, f) for f in filters]
    return np.stack(feature_maps, axis=0)


if __name__ == "__main__":
    # ---- bite 2 demo: one filter, one feature map ----
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

    # ---- bite 3 demo: many filters, stack of feature maps ----
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

    print(f"\n\nWith {filters.shape[0]} filters -> output shape {feature_maps.shape}")
    for i, fm in enumerate(feature_maps):
        names = ["vertical edge", "horizontal edge", "blur"]
        print(f"\nFeature map {i} ({names[i]}):")
        print(fm)
