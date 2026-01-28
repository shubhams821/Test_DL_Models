from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    dim: int = 512
    n_heads: int = 4
    n_layers: int = 4
    vocab_size: int = 1000
    ffn_multiplier: int = 4
    use_cacheing: bool = False
    max_batch_size: int = 32
    max_seq_size: int = 100


class MultiHeadAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.dim % args.n_heads == 0

        self.dim = args.dim
        self.n_heads = args.n_heads
        self.head_dim = self.dim // self.n_heads
        self.cache_k = None if not args.use_cacheing else torch.zeros(args.max_batch_size, args.max_seq_size, args.n_heads, self.head_dim)
        self.cache_v = None if not args.use_cacheing else torch.zeros(args.max_batch_size, args.max_seq_size, args.n_heads, self.head_dim)
        
        # linear projections
        self.W_Q = nn.Linear(self.dim, self.dim, bias=False)
        self.W_K = nn.Linear(self.dim, self.dim, bias=False)
        self.W_V = nn.Linear(self.dim, self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim)

    
    def forward(self, x, mask: torch.Tensor, start_pos: int):
        B, T, C = x.shape
        
        q, k, v = self.W_Q(x), self.W_K(x), self.W_V(x)
        
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim)

        if self.cache_k is not None:
            self.cache_k[:B, start_pos:start_pos + T, :, :] = k
            self.cache_v[:B, start_pos:start_pos + T, :, :] = v  # Fixed: was cache_V
            keys = self.cache_k[:B, :start_pos + T, :, :]
            values = self.cache_v[:B, :start_pos + T, :, :]
        else:
            keys = k
            values = v

        q = q.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        att = q @ keys.transpose(-2, -1) / (self.head_dim ** 0.5)
        if mask is not None:
            att = att.masked_fill(mask == 0, float("-inf"))
        
        att = F.softmax(att, dim=-1)
        out = att @ values  # Fixed: was v instead of values
        out = out.transpose(1, 2).contiguous().view(B, T, self.dim)  # Fixed: reshape back
        return self.out_proj(out)
    

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.hidden_dim = args.ffn_multiplier * self.dim

        self.net = nn.Sequential(
            nn.Linear(self.dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.dim)  # Fixed: was self.dim instead of self.hidden_dim
        )
    
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.ln1 = nn.LayerNorm(args.dim)
        self.attn = MultiHeadAttention(args)
        self.ln2 = nn.LayerNorm(args.dim)
        self.ff = FeedForward(args)
    
    def forward(self, x, mask, start_pos):  # Fixed: added start_pos parameter
        x = x + self.attn(self.ln1(x), mask, start_pos)  # Fixed: pass start_pos
        x = x + self.ff(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.tok_emb = nn.Embedding(args.vocab_size, args.dim)
        self.pos_emb = nn.Embedding(args.max_seq_size, args.dim)

        self.blocks = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])  # Fixed: create list of blocks
        self.ln_final = nn.LayerNorm(args.dim)
        self.out_head = nn.Linear(args.dim, args.vocab_size)
    
    def forward(self, toks, start_pos: int):
        B, T = toks.shape

        tok_emb = self.tok_emb(toks)
        pos = torch.arange(start_pos, start_pos + T, device=toks.device)  # Fixed: create position indices
        pos_emb = self.pos_emb(pos)

        x = tok_emb + pos_emb
        
        # Create causal mask
        mask = torch.tril(torch.ones(T, T, device=toks.device)).view(1, 1, T, T)
        
        for block in self.blocks:
            x = block(x, mask, start_pos)  # Fixed: pass start_pos
        x = self.ln_final(x)
        logits = self.out_head(x)
        return logits


# Fixed: instantiate with parentheses
model = Transformer(ModelArgs())
idx = torch.randint(0, ModelArgs().vocab_size, (2, 32))  # Fixed: proper shape (B, T)

start_pos = 0
logits = model(idx, start_pos)  # Fixed: pass start_pos

last_token = torch.argmax(logits[:, -1], dim=-1).unsqueeze(1)  # Fixed: add batch dimension
start_pos = idx.shape[1]
for i in range(10):
    logits = model(last_token, start_pos)  # Fixed: pass start_pos
    last_token = torch.argmax(logits[:, -1], dim=-1).unsqueeze(1)  # Fixed: maintain shape
    start_pos += 1
