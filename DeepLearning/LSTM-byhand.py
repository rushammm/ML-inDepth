
import numpy as np


def sigmoid(z):
    # the gate squish: maps anything to (0, 1) -> "how much" of something
    return 1.0 / (1.0 + np.exp(-z))


def lstm_forward(X, h0, c0, params):
    # Run the LSTM forward and cache everything backward will need.
    #
    # Args:
    #   X      : (T, input_dim)   -- one input vector per timestep
    #   h0     : (hidden_dim,)    -- initial whisper memory (output state)
    #   c0     : (hidden_dim,)    -- initial notebook  (cell state)
    #   params : dict of weights/biases, one set per gate (f, i, g, o)
    #            W_x* : (hidden_dim, input_dim)   acts on x_t
    #            W_h* : (hidden_dim, hidden_dim)  acts on h_{t-1}
    #            b_*  : (hidden_dim,)
    #
    # Returns:
    #   H     : (T, hidden_dim)  -- output state at every timestep
    #   C     : (T, hidden_dim)  -- cell state (notebook) at every timestep
    #   cache : everything backward (BPTT) will need

    T = X.shape[0]
    hidden_dim = h0.shape[0]

    # storage for the two memory lines through time
    H = np.zeros((T, hidden_dim))
    C = np.zeros((T, hidden_dim))

    # storage for the gates -- backward needs the exact values we used
    F = np.zeros((T, hidden_dim))   # forget gate
    I = np.zeros((T, hidden_dim))   # input gate
    G = np.zeros((T, hidden_dim))   # candidate (new content)
    O = np.zeros((T, hidden_dim))   # output gate

    h = h0
    c = c0
    for t in range(T):
        x = X[t]

        # the four little layers -- each is squish(W_x @ x + W_h @ h + b),
        # the SAME shape as one line of your RNN, just four of them.
        f = sigmoid(params["W_xf"] @ x + params["W_hf"] @ h + params["b_f"])  # erase how much old
        i = sigmoid(params["W_xi"] @ x + params["W_hi"] @ h + params["b_i"])  # write how much new
        g = np.tanh(params["W_xg"] @ x + params["W_hg"] @ h + params["b_g"])  # what the new stuff is
        o = sigmoid(params["W_xo"] @ x + params["W_ho"] @ h + params["b_o"])  # read out how much

        # update the notebook: keep a fraction of old, add a fraction of new
        c = f * c + i * g
        # this step's output: a filtered peek at the notebook
        h = o * np.tanh(c)

        # stash everything
        H[t] = h
        C[t] = c
        F[t] = f
        I[t] = i
        G[t] = g
        O[t] = o

    cache = (X, h0, c0, H, C, F, I, G, O, params)
    return H, C, cache


def lstm_backward(dH, cache):
    # BPTT: gradient of the loss w.r.t. every LSTM parameter.
    #
    # Args:
    #   dH    : (T, hidden_dim)  -- dL/dh_t for every timestep
    #           (if the loss only uses h_T, fill rows 0..T-2 with zeros)
    #   cache : tuple returned by lstm_forward
    #
    # Returns:
    #   grads : dict of gradients, one per entry in params (same keys/shapes)
    #   dh0, dc0, dX

    X, h0, c0, H, C, F, I, G, O, params = cache
    T, hidden_dim = H.shape

    # one gradient accumulator per weight, shaped like the weight itself
    grads = {k: np.zeros_like(v) for k, v in params.items()}
    dX = np.zeros_like(X)

    # gradients flowing in "from the future" (step t+1) -- start at zero
    dh_next = np.zeros(hidden_dim)   # into h_t via the recurrence
    dc_next = np.zeros(hidden_dim)   # into c_t via the notebook highway

    for t in reversed(range(T)):
        x = X[t]
        # the previous step's memories (or the initial ones at t == 0)
        h_prev = H[t - 1] if t > 0 else h0
        c_prev = C[t - 1] if t > 0 else c0
        f, i, g, o, c = F[t], I[t], G[t], O[t], C[t]

        # total gradient landing on h_t: from the loss here + from the future
        dh = dH[t] + dh_next

        # ---- back through  h = o * tanh(c) ----
        do = dh * np.tanh(c)
        dc = dh * o * (1 - np.tanh(c) ** 2)
        dc += dc_next                    # plus whatever flowed back from c_{t+1}

        # ---- back through  c = f * c_prev + i * g ----
        df = dc * c_prev
        di = dc * g
        dg = dc * i
        dc_prev = dc * f                 # <<< THE HIGHWAY: just multiply by f.
                                         #     no W_hh, no tanh-squish. this is the fix.

        # ---- back through each gate's squish (sigmoid' = s(1-s), tanh' = 1-t^2) ----
        daf = df * f * (1 - f)
        dai = di * i * (1 - i)
        dag = dg * (1 - g ** 2)
        dao = do * o * (1 - o)

        # ---- accumulate weight gradients (outer products), same as your RNN ----
        for gate, da in (("f", daf), ("i", dai), ("g", dag), ("o", dao)):
            grads["W_x" + gate] += np.outer(da, x)
            grads["W_h" + gate] += np.outer(da, h_prev)
            grads["b_" + gate]  += da

        # ---- gradient to this step's input, and to h_prev / c_prev ----
        dX[t] = (params["W_xf"].T @ daf + params["W_xi"].T @ dai
                 + params["W_xg"].T @ dag + params["W_xo"].T @ dao)
        dh_next = (params["W_hf"].T @ daf + params["W_hi"].T @ dai
                   + params["W_hg"].T @ dag + params["W_ho"].T @ dao)
        dc_next = dc_prev                # carry the notebook gradient one step back

    return grads, dh_next, dc_next, dX
