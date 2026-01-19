# ==== moe.py ====
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass
import math


@dataclass
class MoeArgs:
    num_experts: int
    num_experts_per_tok: int


class MoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, moe_args: MoeArgs):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.num_experts = moe_args.num_experts
        self.num_experts_per_tok = moe_args.num_experts_per_tok

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x = x.view(-1, x.shape[-1])
        
        # Compute gate scores
        scores = self.gate(x)
        expert_weights, expert_indices = torch.topk(
            scores, self.num_experts_per_tok, dim=-1
        )
        expert_weights = F.softmax(expert_weights, dim=-1)
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # Route to experts
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(expert_indices == i)
            if batch_idx.shape[0] == 0:
                continue
            expert_out = expert(x[batch_idx])
            output[batch_idx] += expert_weights[batch_idx, nth_expert, None] * expert_out
        
        return output.view(*orig_shape)


# ==== cache.py ====
class BufferCache:
    """KV cache for inference"""
    def __init__(
        self,
        n_layers: int,
        max_batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
    ):
        self.max_seq_len = max_seq_len
        cache_shape = (max_batch_size, max_seq_len, n_kv_heads, head_dim)
        self.k_cache: List[torch.Tensor] = [
            torch.zeros(cache_shape, dtype=dtype) for _ in range(n_layers)
        ]
        self.v_cache: List[torch.Tensor] = [
            torch.zeros(cache_shape, dtype=dtype) for _ in range(n_layers)
        ]
        
    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        seqpos: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bs, seqlen = k.shape[0], k.shape[1]
        self.k_cache[layer_idx][:bs, seqpos:seqpos + seqlen] = k
        self.v_cache[layer_idx][:bs, seqpos:seqpos + seqlen] = v
        
        return (
            self.k_cache[layer_idx][:bs, :seqpos + seqlen],
            self.v_cache[layer_idx][:bs, :seqpos + seqlen],
        )
    
    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None):
        """Move cache to device"""
        for i in range(len(self.k_cache)):
            self.k_cache[i] = self.k_cache[i].to(device=device, dtype=dtype)
            self.v_cache[i] = self.v_cache[i].to(device=device, dtype=dtype)
        return self


# ==== rope.py ====
def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:, None, :]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ==== args.py ====
@dataclass
class LoRAArgs:
    rank: int = 8
    scaling: float = 1.0
    

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    head_dim: int = 128
    hidden_dim: int = 14336
    n_heads: int = 32
    n_kv_heads: int = 8
    norm_eps: float = 1e-5
    vocab_size: int = 32000
    rope_theta: float = 10000.0
    max_batch_size: int = 1
    max_seq_len: int = 32768
    
    # Parallelism
    num_pipeline_ranks: int = 1
    
    # MoE
    moe: Optional[MoeArgs] = None
    
    # LoRA
    lora: Optional[LoRAArgs] = None


# ==== lora.py ====
class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        scaling: float = 1.0,
        bias: bool = False,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = scaling
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.linear(x)
        if self.scaling > 0:
            result += (x @ self.lora_A @ self.lora_B) * self.scaling
        return result


# ==== transformer_layers.py ====
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, seqlen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, seqlen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, seqlen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        lora: Optional[LoRAArgs] = None,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = head_dim

        if lora is not None:
            self.wq = LoRALinear(dim, n_heads * head_dim, rank=lora.rank, scaling=lora.scaling, bias=False)
            self.wk = LoRALinear(dim, n_kv_heads * head_dim, rank=lora.rank, scaling=lora.scaling, bias=False)
            self.wv = LoRALinear(dim, n_kv_heads * head_dim, rank=lora.rank, scaling=lora.scaling, bias=False)
            self.wo = LoRALinear(n_heads * head_dim, dim, rank=lora.rank, scaling=lora.scaling, bias=False)
        else:
            self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
            self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
            self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
            self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[BufferCache] = None,
        layer_idx: int = 0,
        seqpos: int = 0,
    ) -> torch.Tensor:
        bs, seqlen, _ = x.shape

        xq = self.wq(x).view(bs, seqlen, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bs, seqlen, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bs, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        if cache is not None:
            xk, xv = cache.update(layer_idx, xk, xv, seqpos)

        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)

        output = output.transpose(1, 2).contiguous().view(bs, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        lora: Optional[LoRAArgs] = None,
    ):
        super().__init__()
        if lora is not None:
            self.w1 = LoRALinear(dim, hidden_dim, rank=lora.rank, scaling=lora.scaling, bias=False)
            self.w2 = LoRALinear(hidden_dim, dim, rank=lora.rank, scaling=lora.scaling, bias=False)
            self.w3 = LoRALinear(dim, hidden_dim, rank=lora.rank, scaling=lora.scaling, bias=False)
        else:
            self.w1 = nn.Linear(dim, hidden_dim, bias=False)
            self.w2 = nn.Linear(hidden_dim, dim, bias=False)
            self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        norm_eps: float,
        lora: Optional[LoRAArgs] = None,
        moe: Optional[MoeArgs] = None,
    ):
        super().__init__()
        self.attention = Attention(
            dim=dim,
            head_dim=head_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            lora=lora,
        )
        
        # MoE or standard FFN
        if moe is not None:
            experts = [
                FeedForward(dim=dim, hidden_dim=hidden_dim, lora=lora)
                for _ in range(moe.num_experts)
            ]
            gate = nn.Linear(dim, moe.num_experts, bias=False)
            self.feed_forward = MoeLayer(experts, gate, moe)
        else:
            self.feed_forward = FeedForward(dim=dim, hidden_dim=hidden_dim, lora=lora)
            
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[BufferCache] = None,
        layer_idx: int = 0,
        seqpos: int = 0,
    ) -> torch.Tensor:
        h = x + self.attention(
            self.attention_norm(x),
            freqs_cis,
            mask,
            cache,
            layer_idx,
            seqpos,
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


# ==== transformer.py ====
import os


def is_torchrun() -> bool:
    required_vars = ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE"]
    return all(var in os.environ for var in required_vars)


class Transformer(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        pipeline_rank: int = 0,
        num_pipeline_ranks: int = 1,
    ):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.pipeline_rank = pipeline_rank
        self.num_pipeline_ranks = num_pipeline_ranks
        
        # Only first rank has embeddings
        if pipeline_rank == 0:
            self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        else:
            self.tok_embeddings = None
            
        # Only last rank has output layers
        if pipeline_rank == num_pipeline_ranks - 1:
            self.norm = RMSNorm(args.dim, eps=args.norm_eps)
            self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        else:
            self.norm = None
            self.output = None
            
        # Initialize all layers but slice off those not of this rank
        layers = [
            TransformerBlock(
                dim=args.dim,
                hidden_dim=args.hidden_dim,
                n_heads=args.n_heads,
                n_kv_heads=args.n_kv_heads,
                head_dim=args.head_dim,
                norm_eps=args.norm_eps,
                lora=args.lora,
                moe=args.moe,
            )
            for _ in range(args.n_layers)
        ]
        
        # Distribute layers across pipeline ranks
        num_layers_per_rank = math.ceil(self.n_layers / self.num_pipeline_ranks)
        offset = self.pipeline_rank * num_layers_per_rank
        end = min(self.n_layers, offset + num_layers_per_rank)
        self.layers = nn.ModuleDict({str(i): layers[i] for i in range(offset, end)})
        self.n_local_layers = len(self.layers)

        # Precompute RoPE freqs
        self.freqs_cis = precompute_freqs_cis(
            args.head_dim,
            args.max_seq_len * 2,
            args.rope_theta,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        seqlens: Optional[List[int]] = None,
        cache: Optional[BufferCache] = None,
    ) -> torch.Tensor:
        # First rank: embed tokens
        if self.pipeline_rank == 0:
            h = self.tok_embeddings(input_ids)
        else:
            # In real PP, this would receive activations from previous rank
            h = input_ids
            
        bs, seqlen = h.shape[0], h.shape[1]
        seqpos = 0 if cache is None else cache.max_seq_len
        freqs_cis = self.freqs_cis[seqpos:seqpos + seqlen].to(h.device)

        # Causal mask
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=h.device)
            mask = torch.triu(mask, diagonal=1)
            mask = mask.unsqueeze(0).unsqueeze(0)

        # Forward through local layers
        for layer_id, layer in self.layers.items():
            layer_idx = int(layer_id)
            h = layer(h, freqs_cis, mask, cache, layer_idx, seqpos)

        # Last rank: output projection
        if self.pipeline_rank == self.num_pipeline_ranks - 1:
            h = self.norm(h)
            output = self.output(h)
            return output
        else:
            # In real PP, this would send activations to next rank
            return h

    @staticmethod
    def from_folder(
        folder,
        max_batch_size: int = 1,
        num_pipeline_ranks: int = 1,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        """Load model from folder with params.json"""
        import json
        from pathlib import Path
        
        folder = Path(folder)
        with open(folder / "params.json", "r") as f:
            params = json.load(f)
        
        model_args = ModelArgs(
            dim=params["dim"],
            n_layers=params["n_layers"],
            head_dim=params.get("head_dim", params["dim"] // params["n_heads"]),
            hidden_dim=params.get("hidden_dim", 4 * params["dim"]),
            n_heads=params["n_heads"],
            n_kv_heads=params.get("n_kv_heads", params["n_heads"]),
            vocab_size=params["vocab_size"],
            norm_eps=params.get("norm_eps", 1e-5),
            rope_theta=params.get("rope_theta", 10000.0),
            max_batch_size=max_batch_size,
            max_seq_len=params.get("max_seq_len", 32768),
            num_pipeline_ranks=num_pipeline_ranks,
        )
        
        if "moe" in params:
            model_args.moe = MoeArgs(
                num_experts=params["moe"]["num_experts"],
                num_experts_per_tok=params["moe"]["num_experts_per_tok"],
            )
            
        if "lora" in params:
            model_args.lora = LoRAArgs(
                rank=params["lora"].get("rank", 8),
                scaling=params["lora"].get("scaling", 1.0),
            )
        
        # Get pipeline rank from environment or default to 0
        pipeline_rank = int(os.environ.get("RANK", 0)) % num_pipeline_ranks
        
        model = Transformer(
            model_args,
            pipeline_rank=pipeline_rank,
            num_pipeline_ranks=num_pipeline_ranks,
        ).to(device=device, dtype=dtype)
        
        return model


# ==== Example Usage ====
if __name__ == "__main__":
    print("=" * 60)
    print("Example 1: Standard Transformer")
    print("=" * 60)
    args = ModelArgs(
        dim=512,
        n_layers=8,
        n_heads=8,
        n_kv_heads=4,
        head_dim=64,
        hidden_dim=2048,
        vocab_size=32000,
    )
    model = Transformer(args)
    tokens = torch.randint(0, args.vocab_size, (2, 128))
    output = model(tokens)
    print(f"Output shape: {output.shape}\n")
    
    print("=" * 60)
    print("Example 2: MoE Transformer")
    print("=" * 60)
    moe_args = ModelArgs(
        dim=512,
        n_layers=8,
        n_heads=8,
        n_kv_heads=4,
        head_dim=64,
        hidden_dim=2048,
        vocab_size=32000,
        moe=MoeArgs(num_experts=8, num_experts_per_tok=2),
    )
    moe_model = Transformer(moe_args)
    output = moe_model(tokens)
    print(f"Output shape: {output.shape}\n")
    
    print("=" * 60)
    print("Example 3: Pipeline Parallel (2 ranks)")
    print("=" * 60)
    pp_args = ModelArgs(
        dim=512,
        n_layers=8,
        n_heads=8,
        n_kv_heads=4,
        head_dim=64,
        hidden_dim=2048,
        vocab_size=32000,
        num_pipeline_ranks=2,
    )
    
    # Rank 0: has embeddings, layers 0-3
    rank0_model = Transformer(pp_args, pipeline_rank=0, num_pipeline_ranks=2)
    print(f"Rank 0 has {rank0_model.n_local_layers} layers: {list(rank0_model.layers.keys())}")
    print(f"Rank 0 has embeddings: {rank0_model.tok_embeddings is not None}")
    print(f"Rank 0 has output: {rank0_model.output is not None}")
    
    # Rank 1: has layers 4-7, norm, output
    rank1_model = Transformer(pp_args, pipeline_rank=1, num_pipeline_ranks=2)
    print(f"Rank 1 has {rank1_model.n_local_layers} layers: {list(rank1_model.layers.keys())}")
    print(f"Rank 1 has embeddings: {rank1_model.tok_embeddings is not None}")
    print(f"Rank 1 has output: {rank1_model.output is not None}\n")
    
    print("=" * 60)
    print("Example 4: LoRA Fine-tuning")
    print("=" * 60)
    lora_args = ModelArgs(
        dim=512,
        n_layers=4,
        n_heads=8,
        n_kv_heads=4,
        head_dim=64,
        hidden_dim=2048,
        vocab_size=32000,
        lora=LoRAArgs(rank=8, scaling=2.0),
    )
    lora_model = Transformer(lora_args)
    print(f"LoRA model created with rank {lora_args.lora.rank}\n")
    
    if is_torchrun():
        print("Running with torchrun - distributed training enabled")
    else:
        print("Running in single-process mode")
