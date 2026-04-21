import torch

def get_rotary_embeddings(seq_len, dim, device="cpu"):
    """
    Generates sin and cos matrices for RoPE.

    Args:
        seq_len: sequence length
        dim: head dimension (must be even)
    """
    assert dim % 2 == 0, "dim must be even"

    # Compute frequencies
    theta = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))

    # Positions
    positions = torch.arange(seq_len, device=device).float()

    # Outer product → [seq_len, dim/2]
    freqs = torch.einsum("i,j->ij", positions, theta)

    # Expand for even/odd pairing
    sin = torch.sin(freqs)
    cos = torch.cos(freqs)

    return sin, cos


def apply_rotary_pos_emb(q, k, sin, cos):
    """
    Applies RoPE to query and key.

    q, k: [batch, heads, seq_len, dim]
    sin, cos: [seq_len, dim/2]
    """
    # Split even and odd parts
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]

    # Expand sin/cos for broadcasting
    sin = sin.unsqueeze(0).unsqueeze(0)  # [1,1,seq,dim/2]
    cos = cos.unsqueeze(0).unsqueeze(0)

    # Apply rotation
    q_rotated = torch.cat([q1 * cos - q2 * sin,
                           q1 * sin + q2 * cos], dim=-1)

    k_rotated = torch.cat([k1 * cos - k2 * sin,
                           k1 * sin + k2 * cos], dim=-1)

    return q_rotated, k_rotated



# Example shapes
batch = 2
heads = 4
seq_len = 16
dim = 64

q = torch.randn(batch, heads, seq_len, dim)
k = torch.randn(batch, heads, seq_len, dim)

sin, cos = get_rotary_embeddings(seq_len, dim)

q, k = apply_rotary_pos_emb(q, k, sin, cos)
