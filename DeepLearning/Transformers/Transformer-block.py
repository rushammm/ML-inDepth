import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    # Single-head self-attention: the exact 3 steps you coded in numpy,
    # but now Q/K/V come from LEARNED weight matrices (W_q, W_k, W_v).
    def __init__(self, d_model):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model)   # turns a word vector into its query
        self.W_k = nn.Linear(d_model, d_model)   # ... its key
        self.W_v = nn.Linear(d_model, d_model)   # ... its val
        self.d_model = d_model

    def forward(self, x):                         # x: (batch, seq_len, d_model)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # STEP 1 — match: every query dotted with every key (scaled)
        scores = (Q @ K.transpose(-2, -1)) / (self.d_model ** 0.5)   # (batch, seq, seq)
        # STEP 2 — normalize: softmax into attention weights
        weights = F.softmax(scores, dim=-1)
        self.last_weights = weights                                  # stash for inspection
        # STEP 3 — gather: weighted blend of the values
        return weights @ V                                          # (batch, seq, d_model)


class FeedForward(nn.Module):
    # Per-word MLP — your MLP-byhand, two layers with a nonlinearity.
    def __init__(self, d_model, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    # One full block = attention sub-layer + feed-forward sub-layer,
    # each wrapped with layer-norm and a residual connection.
    def __init__(self, d_model, ff_hidden):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = SelfAttention(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = FeedForward(d_model, ff_hidden)

    def forward(self, x):
        # norm -> attention -> ADD back the input (residual highway)
        x = x + self.attn(self.norm1(x))
        # norm -> feed-forward -> ADD back the input (residual highway)
        x = x + self.ff(self.norm2(x))
        return x


if __name__ == "__main__":
    torch.manual_seed(0)
    batch, seq_len, d_model = 1, 5, 16     # 1 sentence, 5 words, 16-dim vectors
    x = torch.randn(batch, seq_len, d_model)

    block = TransformerBlock(d_model, ff_hidden=64)
    out = block(x)

    print("input shape :", tuple(x.shape))
    print("output shape:", tuple(out.shape), " <- same shape, refined contents")
    print("parameters  :", sum(p.numel() for p in block.parameters()))
