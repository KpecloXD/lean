# src/lean/model.py
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.g


def rope_freqs(seq_len, head_dim, theta=10000.0, device="cpu"):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device) / head_dim))
    pos = torch.arange(seq_len, device=device).float()
    f = torch.outer(pos, inv)
    return torch.cos(f), torch.sin(f)


def apply_rope(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    rot = torch.cat((-x2, x1), dim=-1)
    c = torch.cat((cos, cos), dim=-1).unsqueeze(0).unsqueeze(0)
    s = torch.cat((sin, sin), dim=-1).unsqueeze(0).unsqueeze(0)
    return x * c + rot * s


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.q = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.kv_heads * cfg.head_dim, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.kv_heads * cfg.head_dim, bias=False)
        self.o = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, cache=None, padding_mask=None):
        B, S, _ = x.shape
        c = self.cfg

        q = self.q(x).view(B, S, c.n_heads, c.head_dim).transpose(1, 2)
        k = self.k(x).view(B, S, c.kv_heads, c.head_dim).transpose(1, 2)
        v = self.v(x).view(B, S, c.kv_heads, c.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)

        new_cache = (k, v)

        k = k.unsqueeze(2).expand(B, c.kv_heads, c.n_rep, k.shape[2], c.head_dim).contiguous().reshape(B, c.n_heads, -1, c.head_dim)
        v = v.unsqueeze(2).expand(B, c.kv_heads, c.n_rep, v.shape[2], c.head_dim).contiguous().reshape(B, c.n_heads, -1, c.head_dim)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(c.head_dim)
        T = scores.shape[-1]

        past = T - S
        if cache is None or S > 1:
            mask_value = torch.finfo(scores.dtype).min
            causal_mask = torch.triu(
                torch.full((S, T), mask_value, device=x.device, dtype=scores.dtype),
                diagonal=past + 1,
            )
            scores = scores + causal_mask

        if padding_mask is not None:
            if padding_mask.shape[1] == S:
                pm = padding_mask.view(B, 1, 1, S)
            else:
                pm = padding_mask.view(B, 1, 1, T)
            scores = scores.masked_fill(pm, torch.finfo(scores.dtype).min)

        attn = self.drop(F.softmax(scores, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, S, c.d_model)
        return self.o(out), new_cache


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.n2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin, cache=None, padding_mask=None):
        h, cache = self.attn(self.n1(x), cos, sin, cache, padding_mask)
        x = x + h
        x = x + self.ffn(self.n2(x))
        return x, cache


class Lean(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.emb.weight

        cos, sin = rope_freqs(cfg.max_seq, cfg.head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self._init_weights()

    def _init_weights(self):
        std = 0.02
        for p in self.parameters():
            if p.dim() == 2:
                nn.init.normal_(p, mean=0.0, std=std)
            elif p.dim() == 1:
                nn.init.ones_(p)

        residual_std = std / math.sqrt(2 * self.cfg.n_layers)
        for blk in self.blocks:
            nn.init.normal_(blk.attn.o.weight, mean=0.0, std=residual_std)
            nn.init.normal_(blk.ffn.down.weight, mean=0.0, std=residual_std)

    def forward(self, ids, cache=None, start_pos=0, padding_mask=None):
        S = ids.shape[1]
        x = self.emb(ids)
        cos = self.cos[start_pos:start_pos + S]
        sin = self.sin[start_pos:start_pos + S]

        # cache=None  -> считаем всю последовательность с нуля
        # cache=[]    -> первый шаг генерации (прошлого ещё нет)
        # cache=[...] -> продолжаем генерацию, в каждом элементе (k, v) слоя
        new_cache = []
        for i, blk in enumerate(self.blocks):
            prev = cache[i] if cache else None
            x, layer_cache = blk(x, cos, sin, prev, padding_mask)
            new_cache.append(layer_cache)

        return self.lm_head(self.norm_f(x)), new_cache
