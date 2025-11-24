# Minimal attention/norm/FFN blocks used by the translator backbone
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rot = torch.zeros_like(x)
    x_rot[..., ::2] = -x2
    x_rot[..., 1::2] = x1
    return x * cos + x_rot * sin


class GroupedQueryAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_groups: int, dropout: float = 0.0, qk_norm: bool = False):
        super().__init__()
        assert num_heads % max(1, num_kv_groups) == 0
        self.dim = dim
        self.num_heads = int(num_heads)
        self.num_kv_groups = max(1, int(num_kv_groups))
        self.group_size = self.num_heads // self.num_kv_groups
        assert dim % num_heads == 0
        self.head_dim = dim // num_heads
        self.dropout = dropout

        self.Wq = nn.Linear(dim, self.num_heads * self.head_dim, bias=False)
        self.Wk = nn.Linear(dim, self.num_kv_groups * self.head_dim, bias=False)
        self.Wv = nn.Linear(dim, self.num_kv_groups * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim) if qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else None

        self._rope_cache: dict[tuple[int, torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}

    def _rope_cos_sin(self, T: int, device: torch.device, dtype: torch.dtype):
        key = (T, device, dtype)
        cached = self._rope_cache.get(key)
        if cached is not None:
            return cached
        dim_half = self.head_dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim_half, device=device, dtype=torch.float32) / dim_half))
        t = torch.arange(T, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
        sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
        cos = cos.to(dtype).unsqueeze(0).unsqueeze(0)
        sin = sin.to(dtype).unsqueeze(0).unsqueeze(0)
        self._rope_cache[key] = (cos, sin)
        return cos, sin

    def forward(self, x: torch.Tensor, past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, use_cache: bool = False, position_offset: int | torch.Tensor = 0):
        B, T_new, _ = x.shape
        q = self.Wq(x).view(B, T_new, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = self.Wk(x).view(B, T_new, self.num_kv_groups, self.head_dim).transpose(1, 2).contiguous()
        v = self.Wv(x).view(B, T_new, self.num_kv_groups, self.head_dim).transpose(1, 2).contiguous()

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        if isinstance(position_offset, int):
            cos, sin = self._rope_cos_sin(position_offset + T_new, x.device, q.dtype)
            if position_offset > 0:
                cos = cos[:, :, position_offset: position_offset + T_new, :]
                sin = sin[:, :, position_offset: position_offset + T_new, :]
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)
        else:
            off = position_offset.to(device=x.device, dtype=torch.long)
            max_off = int(off.max().item())
            cos_all, sin_all = self._rope_cos_sin(max_off + T_new, x.device, q.dtype)
            ar = torch.arange(T_new, device=x.device, dtype=torch.long)
            idx = (off.unsqueeze(1) + ar.unsqueeze(0))
            cos_b = cos_all.squeeze(0).squeeze(0)[idx].unsqueeze(1)
            sin_b = sin_all.squeeze(0).squeeze(0)[idx].unsqueeze(1)
            q = _apply_rope(q, cos_b, sin_b)
            k = _apply_rope(k, cos_b, sin_b)

        if past_kv is not None:
            k_p, v_p = past_kv
            k = torch.cat([k_p, k], dim=2)
            v = torch.cat([v_p, v], dim=2)

        is_causal = past_kv is None
        # Prefer Flash, then MemEff, then Math; allow FP32 via Math
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            if x.device.type == "cuda" and q.dtype not in (torch.float16, torch.bfloat16):
                amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    out = F.scaled_dot_product_attention(
                        q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal
                    )
            else:
                out = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal
                )
        out = out.transpose(1, 2).contiguous().view(B, T_new, self.dim)
        out = self.out_proj(out)
        if use_cache:
            return out, (k, v)
        return out


class SwiGLU(nn.Module):
    """SwiGLU FFN with parameter names matching checkpoints (w1, w2, w3):
    - w1: Linear(dim -> hidden)
    - w2: Linear(hidden -> dim)
    - w3: Linear(dim -> hidden)
    Forward: w2(silu(w1(x)) * w3(x))
    """
    def __init__(self, dim: int, hidden_mult: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * hidden_mult)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float = 0.0, num_kv_groups: Optional[int] = None, qk_norm: bool = False, attn_type: str = "mha"):
        super().__init__()
        if attn_type == "gqa":
            self.attn = GroupedQueryAttention(dim, num_heads=num_heads, num_kv_groups=(num_kv_groups or num_heads), dropout=dropout)
        else:
            self.attn = MultiHeadAttention(dim=dim, num_heads=num_heads, dropout=dropout)
        self.ffn = SwiGLU(dim, hidden_mult=mlp_ratio, dropout=dropout)
        self.ln1 = RMSNorm(dim)
        self.ln2 = RMSNorm(dim)

    def forward(self, x: torch.Tensor, past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, use_cache: bool = False, position_offset: int = 0):
        a = self.attn(self.ln1(x), past_kv=past_kv, use_cache=use_cache, position_offset=position_offset)
        if use_cache:
            a, kv = a
        x = x + a
        x = x + self.ffn(self.ln2(x))
        if use_cache:
            return x, kv
        return x


class MultiHeadAttention(nn.Module):
    """Standard MHA with fused qkv and RoPE, SDPA backend selection.
    Matches checkpoint naming: qkv (dim->3*dim) and out_proj (dim->dim).
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0, use_rope: bool = True):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.use_rope = use_rope

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self._rope_cache: dict[tuple[int, torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}

    def _rope_cos_sin(self, T: int, device: torch.device, dtype: torch.dtype):
        key = (T, device, dtype)
        cached = self._rope_cache.get(key)
        if cached is not None:
            return cached
        dim_half = self.head_dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim_half, device=device, dtype=torch.float32) / dim_half))
        t = torch.arange(T, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
        sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
        cos = cos.to(dtype).unsqueeze(0).unsqueeze(0)
        sin = sin.to(dtype).unsqueeze(0).unsqueeze(0)
        self._rope_cache[key] = (cos, sin)
        return cos, sin

    def forward(self, x: torch.Tensor, past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, use_cache: bool = False, position_offset: int = 0):
        B, T_new, _ = x.shape
        qkv = self.qkv(x).view(B, T_new, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k_new, v_new = qkv[0].contiguous(), qkv[1].contiguous(), qkv[2].contiguous()

        if self.use_rope:
            cos, sin = self._rope_cos_sin(position_offset + T_new, x.device, q.dtype)
            if position_offset > 0:
                cos = cos[:, :, position_offset: position_offset + T_new, :]
                sin = sin[:, :, position_offset: position_offset + T_new, :]
            q = _apply_rope(q, cos, sin)
            k_new = _apply_rope(k_new, cos, sin)

        if past_kv is not None:
            k, v = past_kv
            k = torch.cat([k, k_new], dim=2)
            v = torch.cat([v, v_new], dim=2)
        else:
            k, v = k_new, v_new

        is_causal = past_kv is None
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            if x.device.type == "cuda" and q.dtype not in (torch.float16, torch.bfloat16):
                amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    out = F.scaled_dot_product_attention(
                        q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal
                    )
            else:
                out = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal
                )
        out = out.transpose(1, 2).contiguous().view(B, T_new, self.dim)
        if out.dtype != x.dtype:
            out = out.to(x.dtype)
        out = self.out_proj(out)
        if use_cache:
            return out, (k, v)
        return out
