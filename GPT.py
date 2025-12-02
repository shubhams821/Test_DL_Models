import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Combined QKV projection
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        B, T, C = x.shape

        qkv = self.qkv(x)  # (B,T,3C)
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape into heads
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # attention scores
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)   # (B, heads, T, T)

        # apply causal mask
        if mask is not None:
            att = att.masked_fill(mask == 0, float('-inf'))

        att = F.softmax(att, dim=-1)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, ff_hidden_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, ff_hidden_dim)

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, dim=256, depth=4, heads=4, max_len=256):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_len, dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, ff_hidden_dim=4*dim)
            for _ in range(depth)
        ])

        self.ln_final = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # Causal mask: T x T lower triangular
        self.register_buffer(
            "mask", torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, idx):
        B, T = idx.shape

        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))

        x = tok + pos

        mask = self.mask[:, :, :T, :T]  # crop

        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_final(x)
        logits = self.head(x)
        return logits


# ----------------------------
# Try a forward pass
# ----------------------------
vocab_size = 1000
model = GPT(vocab_size)

x = torch.randint(0, vocab_size, (2, 32))  # (B=2, T=32)
logits = model(x)

print("Output:", logits.shape)  # (2, 32, vocab_size)
