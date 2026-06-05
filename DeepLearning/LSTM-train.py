
# Train the from-scratch LSTM on a "remember the first number" task,
# so we can watch the forget gate learn to hold long-term memory.
#
# Task: a length-T sequence. x[0] = +1 or -1 (the signal). x[1..T-1] = noise.
#       At the LAST step, predict whether x[0] was positive. To win, the LSTM
#       MUST carry x[0] across all T steps -> a direct test of the cell highway.

import numpy as np
import importlib.util
import os

# --- load our hand-built LSTM (filename is hyphenated, so load it manually) ---
_path = os.path.join(os.path.dirname(__file__), "LSTM-byhand.py")
_spec = importlib.util.spec_from_file_location("lstm_byhand", _path)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
lstm_forward, lstm_backward, sigmoid = _m.lstm_forward, _m.lstm_backward, _m.sigmoid


def make_sequence(T, rng):
    # x[0] is the signal (+1/-1); the rest is small noise. y = 1 if signal>0.
    x = rng.normal(0, 0.3, size=(T, 1))
    signal = 1.0 if rng.random() < 0.5 else -1.0
    x[0, 0] = signal
    y = 1.0 if signal > 0 else 0.0
    return x, y


def init_params(input_dim, hidden_dim, rng):
    p = {}
    for gate in ["f", "i", "g", "o"]:
        p["W_x" + gate] = rng.normal(0, 1, (hidden_dim, input_dim)) / np.sqrt(input_dim)
        p["W_h" + gate] = rng.normal(0, 1, (hidden_dim, hidden_dim)) / np.sqrt(hidden_dim)
        p["b_" + gate] = np.zeros(hidden_dim)          # start forget bias at 0 -> watch it climb
    # output head: h_T -> single logit
    p["W_out"] = rng.normal(0, 1, (1, hidden_dim)) / np.sqrt(hidden_dim)
    p["b_out"] = np.zeros(1)
    return p


# --- Adam state (you've built this before) ---
def init_adam(params):
    return ({k: np.zeros_like(v) for k, v in params.items()},
            {k: np.zeros_like(v) for k, v in params.items()})


def adam_step(params, grads, m, v, t, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
    for k in params:
        m[k] = b1 * m[k] + (1 - b1) * grads[k]
        v[k] = b2 * v[k] + (1 - b2) * grads[k] ** 2
        mhat = m[k] / (1 - b1 ** t)
        vhat = v[k] / (1 - b2 ** t)
        params[k] -= lr * mhat / (np.sqrt(vhat) + eps)


def train():
    rng = np.random.default_rng(0)
    T, input_dim, hidden_dim = 15, 1, 16
    batch = 32
    steps = 400

    params = init_params(input_dim, hidden_dim, rng)
    m, v = init_adam(params)
    h0 = np.zeros(hidden_dim)
    c0 = np.zeros(hidden_dim)

    print(f"{'step':>5} {'loss':>8} {'acc':>6} {'mean forget gate':>18}")
    for step in range(1, steps + 1):
        # accumulate gradients over a mini-batch
        gsum = {k: np.zeros_like(val) for k, val in params.items()}
        loss_sum, correct, forget_sum = 0.0, 0, 0.0

        for _ in range(batch):
            x, y = make_sequence(T, rng)

            # ---- forward through the LSTM, then the output head ----
            H, C, cache = lstm_forward(x, h0, c0, params)
            hT = H[-1]
            z = params["W_out"] @ hT + params["b_out"]      # (1,)
            p = sigmoid(z)[0]
            p = np.clip(p, 1e-7, 1 - 1e-7)
            loss_sum += -(y * np.log(p) + (1 - y) * np.log(1 - p))
            correct += int((p > 0.5) == (y > 0.5))
            forget_sum += cache[5].mean()                   # F is cache[5]

            # ---- backward: output head first ----
            dz = np.array([p - y])                          # dL/dz
            gsum["W_out"] += np.outer(dz, hT)
            gsum["b_out"] += dz
            dhT = params["W_out"].T @ dz                    # (hidden,)

            # only the last step gets gradient from the loss
            dH = np.zeros_like(H)
            dH[-1] = dhT
            grads, dh0, dc0, dX = lstm_backward(dH, cache)
            for k in grads:
                gsum[k] += grads[k]

        # average grads over the batch and take one Adam step
        for k in gsum:
            gsum[k] /= batch
        adam_step(params, gsum, m, v, step)

        if step == 1 or step % 40 == 0:
            print(f"{step:>5} {loss_sum/batch:>8.4f} {correct/batch:>6.2f} "
                  f"{forget_sum/batch:>18.3f}")

    return params


if __name__ == "__main__":
    train()
