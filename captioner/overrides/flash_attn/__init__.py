# Minimal flash_attn shim for environments without flash-attn wheels.
# Provides the symbols Florence imports, implemented via torch SDPA fallback.

import torch

def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
    # q,k,v expected shape: (batch, seqlen, nheads, head_dim) or similar
    # Torch SDPA expects (batch, nheads, seqlen, head_dim)
    def _to_bnhd(x):
        if x.dim() == 4 and x.shape[1] != x.shape[2]:
            # assume (b, s, h, d) -> (b, h, s, d)
            return x.permute(0, 2, 1, 3).contiguous()
        return x

    q2 = _to_bnhd(q)
    k2 = _to_bnhd(k)
    v2 = _to_bnhd(v)

    # softmax_scale is handled by SDPA internally; ignore if provided
    out = torch.nn.functional.scaled_dot_product_attention(
        q2, k2, v2,
        attn_mask=None,
        dropout_p=dropout_p if torch.is_grad_enabled() else 0.0,
        is_causal=causal,
    )

    # back to (b, s, h, d) if needed
    if q.dim() == 4 and q.shape[1] != q.shape[2]:
        out = out.permute(0, 2, 1, 3).contiguous()
    return out
