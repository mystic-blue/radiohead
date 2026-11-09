"""RoPE-oscillator attention (Khadraoui, Arenas, Arola-Fernandez 2026) with the
interface of this repository's LoheAttention, so that it can be trained under the
exact TinyStories protocol of Pasqualetti & Guo (training/train_tinystories.py).

Keys and queries are unit content vectors in R^{d_osc} carried by the free
evolution of the substrate, s_n = exp(-t_n Omega0) e_n (the RoPE rotation);
the coupling is the overlap of free states; the probe for token i has detuning
eps_i Omega0 and settles at the exact fixed point z* = (lambda I - eps_i Omega0)^{-1} h.
Read-out: resonant (Cauchy) against the encoded keys. No anchors, no positional
parameters. The physics is a verbatim copy of rope_osc.py in the companion repo.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-8
_EPS_CUSP = 1e-6


def _rope(v, omega, pos):
    B, H, T, d = v.shape
    ang = -pos.view(T, 1) * omega.view(1, -1)
    c, s = ang.cos().view(1, 1, T, -1), ang.sin().view(1, 1, T, -1)
    x = v.view(B, H, T, d // 2, 2)
    x1, x2 = x.unbind(-1)
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), -1).view(B, H, T, d)


class RopeOscAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_head, d_osc, p=4.0, res_gamma=0.4,
                 causal=True, beta_init=4.0, rope_base=10000.0, eps_frozen=False):
        super().__init__()
        assert d_osc % 2 == 0
        self.H, self.d_head, self.d, self.m = n_heads, d_head, d_osc, d_osc // 2
        self.p, self.res_gamma, self.causal = float(p), float(res_gamma), causal
        self.lam_min = 1e-3
        self.W_q = nn.Linear(d_model, n_heads * d_osc, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_osc, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_o = nn.Linear(n_heads * d_head, d_model, bias=True)
        k = torch.arange(self.m, dtype=torch.float32)
        self.register_buffer("omega", rope_base ** (-2.0 * k / d_osc), persistent=False)
        self.beta = nn.Parameter(torch.full((n_heads,), float(beta_init)))
        self.w_bias = nn.Parameter(torch.zeros(n_heads))
        self.eps_proj = nn.Linear(d_model, n_heads)
        nn.init.zeros_(self.eps_proj.weight); nn.init.zeros_(self.eps_proj.bias)
        self.eps_scale = nn.Parameter(torch.tensor(1.0))
        if eps_frozen:
            with torch.no_grad():
                self.eps_scale.zero_()
            for prm in (self.eps_scale, self.eps_proj.weight, self.eps_proj.bias):
                prm.requires_grad_(False)

    def _settle(self, h, eps):
        B, H, T, d = h.shape
        hk = h.view(B, H, T, self.m, 2)
        a = (hk * hk).sum(-1)
        eta = eps.unsqueeze(-1) * self.omega.view(1, 1, 1, -1)
        eta2 = eta * eta
        R = torch.sqrt(a.sum(-1) + _EPS_CUSP)

        def N(lam):
            return (a / (lam.unsqueeze(-1) ** 2 + eta2)).sum(-1)

        N0 = (a / eta2.clamp_min(1e-12)).sum(-1)
        locked = N0 >= 1.0
        with torch.no_grad():
            lo = torch.full_like(R, self.lam_min)
            hi = R.clamp_min(self.lam_min + 1e-6)
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                big = N(mid) > 1.0
                lo = torch.where(big, mid, lo)
                hi = torch.where(big, hi, mid)
            lam0 = 0.5 * (lo + hi)
        Nl = N(lam0)
        dN = -2.0 * lam0 * (a / (lam0.unsqueeze(-1) ** 2 + eta2) ** 2).sum(-1)
        lam = (lam0 - (Nl - 1.0) / dN.clamp(max=-1e-8)).clamp_min(self.lam_min)
        Jh = torch.stack((-hk[..., 1], hk[..., 0]), -1)
        den = (lam.unsqueeze(-1) ** 2 + eta2).unsqueeze(-1)
        z_lock = (lam.unsqueeze(-1).unsqueeze(-1) * hk + eta.unsqueeze(-1) * Jh) / den
        u = (1.0 / N0.clamp_min(1e-12)).sqrt().clamp_min(1.0)
        mag = 1.0 / (u + torch.sqrt((u * u - 1.0).clamp_min(0.0) + _EPS_CUSP))
        dirv = Jh / eta.unsqueeze(-1).abs().clamp_min(1e-6) * torch.sign(eps).view(B, H, T, 1, 1)
        dirv = dirv / dirv.view(B, H, T, -1).norm(dim=-1).clamp_min(1e-6).view(B, H, T, 1, 1)
        z_drift = mag.view(B, H, T, 1, 1) * dirv
        z = torch.where(locked.view(B, H, T, 1, 1), z_lock, z_drift).view(B, H, T, d)
        return z, lam, locked, R

    def forward(self, x, padding_mask: Optional[torch.Tensor] = None):
        B, T, _ = x.shape
        H, d = self.H, self.d
        q = F.normalize(self.W_q(x).view(B, T, H, d).transpose(1, 2).float(), dim=-1)
        k = F.normalize(self.W_k(x).view(B, T, H, d).transpose(1, 2).float(), dim=-1)
        pos = torch.arange(T, device=x.device, dtype=torch.float32)
        sq, sk = _rope(q, self.omega, pos), _rope(k, self.omega, pos)
        allowed = torch.ones(T, T, device=x.device)
        if self.causal:
            allowed = torch.tril(allowed)
        allowed = allowed.view(1, 1, T, T)
        if padding_mask is not None:
            allowed = allowed * (~padding_mask).float().view(B, 1, 1, T)
        overlap = torch.matmul(sq, sk.transpose(-2, -1))
        scores = self.beta.view(1, -1, 1, 1) * overlap + self.w_bias.view(1, -1, 1, 1)
        w = F.softplus(scores) * allowed
        wn = w / w.sum(dim=-1, keepdim=True).clamp_min(_EPS)
        h = torch.einsum("bhtj,bhjd->bhtd", wn, sk)
        eps = self.eps_scale * self.eps_proj(x).float().transpose(1, 2)
        z, lam, locked, R = self._settle(h, eps)
        zm = torch.einsum("bhtd,bhjd->bhtj", z, sk)
        zn2 = (z * z).sum(-1, keepdim=True)
        shifted = 1.0 / (self.res_gamma ** 2 + (zn2 + 1.0 - 2.0 * zm).clamp(min=0.0))
        if self.p != 1.0:
            shifted = shifted.pow(self.p)
        shifted = shifted * allowed
        attn = shifted / (shifted.sum(dim=-1, keepdim=True) + _EPS)
        v = self.W_v(x).view(B, T, H, self.d_head).transpose(1, 2)
        out = torch.einsum("bhij,bhjd->bhid", attn.to(v.dtype), v)
        out = out.transpose(1, 2).contiguous().view(B, T, H * self.d_head)
        self._last_debug = {"locked_frac": locked.float().mean().detach(),
                            "rho": (eps.abs() * self.omega[0] / R.clamp_min(1e-6)).mean().detach()}
        return self.W_o(out), attn


class _RopeOscLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_head, d_ff, dropout, d_osc, p, res_gamma, eps_frozen):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RopeOscAttention(d_model, n_heads, d_head, d_osc, p=p, res_gamma=res_gamma,
                                     causal=True, eps_frozen=eps_frozen)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(d_ff, d_model), nn.Dropout(dropout))

    def forward(self, x, padding_mask=None):
        out, _ = self.attn(self.norm1(x), padding_mask=padding_mask)
        x = x + out
        return x + self.ffn(self.norm2(x))


class RopeOscLanguageTransformer(nn.Module):
    """Same backbone as LoheLanguageTransformer (pre-norm, GELU FFN, dropout,
    untied head) with the RoPE-oscillator attention. use_pe=False removes the
    sinusoidal encoding: position then enters only through the free evolution."""

    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2, d_ff=512,
                 max_seq_len=128, dropout=0.1, d_osc=32, p=4.0, res_gamma=0.4,
                 use_pe=False, eps_frozen=False):
        super().__init__()
        from .transformer import SinusoidalPositionalEncoding
        d_head = d_model // n_heads
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = (SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)
                        if use_pe else nn.Dropout(dropout))
        self.layers = nn.ModuleList([
            _RopeOscLayer(d_model, n_heads, d_head, d_ff, dropout, d_osc, p, res_gamma, eps_frozen)
            for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, padding_mask=None):
        h = self.pos_enc(self.embedding(x))
        for layer in self.layers:
            h = layer(h, padding_mask=padding_mask)
        return self.head(self.norm(h))
