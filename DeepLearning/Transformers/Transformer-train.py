
# Train the from-scratch TransformerBlock on an "associative recall" task,
# so we can watch self-attention learn CONTENT-BASED LOOKUP.
#
# Task: a sequence = [query, content, content, content, content].
#   - Each content token carries a (color, number).  All colors are distinct.
#   - The query token carries ONE color and a blank number.
#   - Goal: output the number of the content token whose color MATCHES the query.
#
# Why this proves attention: to win, the query's W_q must build a vector that
# dot-products HIGH against the matching color's W_k key, and W_v must ferry that
# token's number back. That is exactly match -> normalize -> gather. An MLP can't
# do it: the answer depends on WHICH other token matches, i.e. pure routing.

import torch
import torch.nn as nn
import importlib.util
import os

# --- load our hand-built block (filename is hyphenated, so load it manually) ---
_path = os.path.join(os.path.dirname(__file__), "Transformer-block.py")
_spec = importlib.util.spec_from_file_location("transformer_block", _path)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
TransformerBlock = _m.TransformerBlock

N_COLORS = 4        # how many distinct colors exist
N_NUMBERS = 4       # the numbers (also our class labels 0..3)
N_CONTENT = 4       # content tokens per sequence (one per color)
FEAT = N_COLORS + N_NUMBERS + 1     # raw token width: color | number | is_query


def make_batch(batch, rng):
    # Build `batch` sequences. Returns x:(batch, seq, FEAT) and y:(batch,) labels.
    seq_len = N_CONTENT + 1                          # +1 for the query token
    x = torch.zeros(batch, seq_len, FEAT)
    y = torch.zeros(batch, dtype=torch.long)

    for b in range(batch):
        colors = torch.randperm(N_COLORS, generator=rng)[:N_CONTENT]   # distinct colors
        numbers = torch.randint(0, N_NUMBERS, (N_CONTENT,), generator=rng)

        # positions 1..N_CONTENT are the content tokens
        for i in range(N_CONTENT):
            x[b, i + 1, colors[i]] = 1.0                       # color one-hot
            x[b, i + 1, N_COLORS + numbers[i]] = 1.0           # number one-hot

        # position 0 is the query: pick one of the present colors, blank number
        pick = torch.randint(0, N_CONTENT, (1,), generator=rng).item()
        x[b, 0, colors[pick]] = 1.0                            # query's color
        x[b, 0, FEAT - 1] = 1.0                                # is_query flag
        y[b] = numbers[pick]                                   # answer = matching number

    return x, y


class RecallModel(nn.Module):
    # embed raw features -> one TransformerBlock -> read the QUERY position -> classify
    def __init__(self, d_model=32, ff_hidden=64):
        super().__init__()
        self.embed = nn.Linear(FEAT, d_model)
        self.block = TransformerBlock(d_model, ff_hidden)
        self.head = nn.Linear(d_model, N_NUMBERS)

    def forward(self, x):
        h = self.embed(x)                 # (batch, seq, d_model)
        h = self.block(h)                 # attention does the routing
        q = h[:, 0, :]                    # the query position now holds the gathered answer
        return self.head(q)               # (batch, N_NUMBERS) logits


def train():
    rng = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    model = RecallModel()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    print(f"{'step':>5} {'loss':>8} {'acc':>6}")
    for step in range(1, 601):
        x, y = make_batch(64, rng)
        logits = model(x)
        loss = loss_fn(logits, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step == 1 or step % 50 == 0:
            acc = (logits.argmax(1) == y).float().mean().item()
            print(f"{step:>5} {loss.item():>8.4f} {acc:>6.2f}")

    # ---- proof: show WHERE the query token decided to look ----
    model.eval()
    x, y = make_batch(1, rng)
    with torch.no_grad():
        model(x)
    w = model.block.attn.last_weights[0]      # (seq, seq) attention weights
    query_row = w[0]                          # how the query attends to every position
    print("\nquery attends to positions:", [f"{p:.2f}" for p in query_row.tolist()])
    print("(position 0 is the query itself; 1..4 are content. The big number"
          " should sit on the matching-color token.)")


if __name__ == "__main__":
    train()
