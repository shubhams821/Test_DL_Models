import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        
        #combined QKV
        self.W_QKV = nn.Linear(dim , 3*dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, mask = None, kv_cache= None):
        B, T, C = x.shape
        qkv = self.W_QKV(x)

        q, k, v = qkv.chunk(3, dim =-1)
        q = q.view(B, T, self.heads, self.head_dim).transpose(1,2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1,2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1,2)

        if kv_cache is not None:
            k_, v_ = kv_cache
            k = torch.concat([k_, k], dim = -2)
            v = torch.concat([v_, v], dim = -2)
            att = q @ k.transpose(-2,-1)/(self.head_dim**0.5)
        else:
            att = q @ k.transpose(-2,-1)/(self.head_dim**0.5)
            if mask is not None:
                att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim = -1)
        out = att @ v
        out = out.transpose(1,2).contiguous().view(B, T, C)
        return self.out_proj(out), (k,v)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, hidden_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, hidden_dim)

    def forward(self, x, mask, kv_cache):
        x_, kv_cache = self.attn(self.ln1(x), mask, kv_cache)
        x = x_ + x
        x = x + self.ff(self.ln2(x))
        return x, kv_cache

class GPT(nn.Module):
    def __init__(self, vocab_size, dim= 256, heads = 4, depth = 4, max_len = 256):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_len, dim)

        self.blocks = nn.ModuleList(
            TransformerBlock(dim, heads, hidden_dim=3*dim)
            for i in range(depth)
        )
        self.ln_final = nn.LayerNorm(dim)
        self.heads = nn.Linear(dim, vocab_size)
        self.register_buffer(
            "mask", torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(0)
        )
    def forward(self, idx, kv_cache_dict=None):
        B, T = idx.shape

        if kv_cache_dict is not None:
            tok_emb = self.token_emb(idx[:,-1])
            tok_emb = tok_emb.unsqueeze(1)
            pos_emb = self.pos_emb(torch.tensor(T-1))
        
        else:
            kv_cache_dict = {i:None for i in range(len(self.blocks))}
            tok_emb = self.token_emb(idx)
            pos_emb = self.pos_emb(torch.arange(T, device = idx.device))

        x = tok_emb + pos_emb
        mask = self.mask[:,:,:T, :T]
        for i, block in enumerate(self.blocks):
            x, kv_cache = block(x, mask, kv_cache_dict[i])
            kv_cache_dict[i] = kv_cache
        
        x = self.ln_final(x)
        logits = self.heads(x)
        return logits, kv_cache_dict


if __name__ == "__main__":
    vocab_size = 1000
    model = GPT(vocab_size)
    idx = torch.randint(0, vocab_size, (2,32))
    idx2 = idx
    kv_cache_dict = None
    for i in range(10):
        logits, kv_cache_dict = model(idx, kv_cache_dict)
        tokens = logits.argmax(dim = -1)[:,-1]
        logits2, _ = model(idx2, None)
        tokens2 = logits2.argmax(dim = -1)[:,-1]
        idx = torch.concat([idx, tokens.unsqueeze(1)], dim = -1)
        idx2 = torch.concat([idx2, tokens2.unsqueeze(1)], dim =-1)
