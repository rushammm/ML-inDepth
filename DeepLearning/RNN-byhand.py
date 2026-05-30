
import numpy as np


def rnn_forward(X, h0, W_xh, W_hh, b_h):
#    Run the RNN forward and cache everything backward will need.
    T = X.shape[0]
    hidden_dim = h0.shape[0]
    H = np.zeros((T, hidden_dim))

    h = h0
    for t in range(T):
        # the one RNN equation — same line, reused every timestep
        h = np.tanh(W_xh @ X[t] + W_hh @ h + b_h)
        H[t] = h

    cache = (X, h0, H, W_xh, W_hh, b_h)
    return H, cache


def rnn_backward(dH, cache):
   
    # BPTT: gradient of some loss w.r.t. all RNN parameters.

    # Args:
    #     dH    : (T, hidden_dim)  — dL/dh_t for every timestep
    #             (if your loss only uses h_T, fill rows 0..T-2 with zeros)
    #     cache : tuple returned by rnn_forward

    # Returns:
    #     dW_xh, dW_hh, db_h, dh0, dX — gradients with matching shapes
   
    X, h0, H, W_xh, W_hh, b_h = cache
    T, hidden_dim = H.shape

    dW_xh = np.zeros_like(W_xh)
    dW_hh = np.zeros_like(W_hh)
    db_h  = np.zeros_like(b_h)
    dX    = np.zeros_like(X)
    dh_next = np.zeros(hidden_dim)   # gradient flowing in from the "future" (step t+1)

    # walk backward through time
    for t in reversed(range(T)):
        # total gradient on h_t = direct loss grad + grad coming back from h_{t+1}
        dh_t = dH[t] + dh_next

        # pull gradient through the tanh: dL/dz_t = dh_t * (1 - h_t^2)
        dz_t = dh_t * (1 - H[t] ** 2)

        # previous hidden state  (h_{t-1} = h0 when t == 0)
        h_prev = H[t-1] if t > 0 else h0

        # accumulate weight grads — the "+=" is the weight-sharing-across-time step
        dW_xh += np.outer(dz_t, X[t])
        dW_hh += np.outer(dz_t, h_prev)
        db_h  += dz_t

        # send gradient backward: into x_t, and into h_{t-1} (becomes next iter's dh_next)
        dX[t]   = W_xh.T @ dz_t
        dh_next = W_hh.T @ dz_t

    dh0 = dh_next   # whatever gradient is left after walking all the way back
    return dW_xh, dW_hh, db_h, dh0, dX


#  sanity test 1: forward matches the by-hand walkthrough 
if __name__ == "__main__":
    W_xh = np.array([[0.5]])
    W_hh = np.array([[0.5]])
    b_h  = np.array([0.0])
    h0   = np.array([0.0])
    X    = np.array([[1.0], [1.0]])

    H, _ = rnn_forward(X, h0, W_xh, W_hh, b_h)
    print(f"h_1 = {H[0]}   (expected ~0.4621)")
    print(f"h_2 = {H[1]}   (expected ~0.6241)")

    # sanity test 2: BPTT vs finite differences (gradient check) 
    print("\nGradient check (BPTT vs numerical):")
    rng = np.random.default_rng(0)
    T, input_dim, hidden_dim = 4, 3, 5

    X    = rng.standard_normal((T, input_dim))
    h0   = rng.standard_normal(hidden_dim) * 0.1
    W_xh = rng.standard_normal((hidden_dim, input_dim))  * 0.3
    W_hh = rng.standard_normal((hidden_dim, hidden_dim)) * 0.3
    b_h  = rng.standard_normal(hidden_dim) * 0.1

    # toy loss: sum of all hidden states  → dL/dh_t = 1 for every t
    def loss_fn(W_xh, W_hh, b_h, h0):
        H, _ = rnn_forward(X, h0, W_xh, W_hh, b_h)
        return H.sum()

    # analytical gradients via BPTT
    H, cache = rnn_forward(X, h0, W_xh, W_hh, b_h)
    dH = np.ones_like(H)
    dW_xh, dW_hh, db_h_grad, dh0, dX = rnn_backward(dH, cache)

    # numerical gradient for W_hh (same recipe works for any param)
    eps = 1e-5
    dW_hh_num = np.zeros_like(W_hh)
    for i in range(W_hh.shape[0]):
        for j in range(W_hh.shape[1]):
            Wp = W_hh.copy(); Wp[i, j] += eps
            Wm = W_hh.copy(); Wm[i, j] -= eps
            dW_hh_num[i, j] = (loss_fn(W_xh, Wp, b_h, h0) - loss_fn(W_xh, Wm, b_h, h0)) / (2 * eps)

    err_Whh = np.max(np.abs(dW_hh - dW_hh_num))
    print(f"  max abs diff on dW_hh : {err_Whh:.2e}    (good if < 1e-6)")

    # quick numerical check on b_h too
    db_h_num = np.zeros_like(b_h)
    for i in range(b_h.shape[0]):
        bp = b_h.copy(); bp[i] += eps
        bm = b_h.copy(); bm[i] -= eps
        db_h_num[i] = (loss_fn(W_xh, W_hh, bp, h0) - loss_fn(W_xh, W_hh, bm, h0)) / (2 * eps)
    err_bh = np.max(np.abs(db_h_grad - db_h_num))
    print(f"  max abs diff on db_h  : {err_bh:.2e}    (good if < 1e-6)")
