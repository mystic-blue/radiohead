"""Attention layers compared in the letter (19 Oct 2026).  Every layer takes the pre-normalized residual stream
x (B, T, d_model) and returns (output, weights (B, H, T, T) or None, diagnostics).  Weights are returned only when
`need_weights` is set (analysis); otherwise the fastest exact path is used.

    SM        softmax + RoPE                                     Vaswani et al. 2017; Su et al. 2024
    GATED     SM with an elementwise sigmoid gate on the SDPA    Qiu et al., NeurIPS 2025 (G1, the variant they
              output, gate computed from the same x              recommend; as in their modeling_qwen3.py: the gate
                                                                 is extra outputs of the query projection, no bias)
    SIGMOID   sigmoid(q.k / sqrt d + b), b = -log(n_train),      Ramapuram et al., ICLR 2025 (their language setup:
              ALiBi, QK-norm                                     ALiBi, fixed b = -log of the training length, QK norm)
    PG        Pasqualetti & Guo, TMLR 2026, as published         (copied from bench/models.py, checked there against
              (anchors per position, softplus coupling)           the official code to machine precision)
    KM        the radio head: implicit (bisection + one Newton step) or explicit (closed width + `steps` Newton
              steps); controls `norm` (weights forced to sum |w|^2 = 1: no silence), `noage` (eps = 0: no position) and
              `noant` (G = 1, no query: no content)

The mask (causal; on synthetic tasks also the diagonal) belongs to the protocol and is the same for every layer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Cfg:
    model: str = "SM"             # SM | GATED | SIGMOID | PG | KM
    mask_self: bool = False       # synthetic tasks: the diagonal is masked (protocol)
    max_len: int = 128            # training length (PG: number of anchors; SIGMOID: b = -log max_len)
    pos_len: int = 0              # PG: anchors up to the longest test length (unseen ones stay at their init)
    pg_d_osc: int = 32            # PG: oscillator width (largest published)
    km_explicit: int = 0          # KM: 0 = implicit; n > 0 = explicit width + n closed Newton steps
    km_delta: float = 1e-3        # KM: fixed line width D = delta K
    km_rho: float = 1e-8          # KM: initial isotropic floor of the antenna (learned per head)
    km_norm: bool = False         # KM control: weights rescaled to sum |w|^2 = 1 (no silent branch)
    km_noage: bool = False        # KM control: eps = 0 (no ageing, so no position)
    km_noant: bool = False        # KM control: G = 1 (no antenna, so no content: every token is heard alike)
    km_same: bool = False         # KM: every head starts at the same ageing rate T^-1/2 (no geometric ladder)
    km_k0: float = 1.5            # KM: initial K / K_c for a context of random contents (1.5: locked; < 1: silent)

    def to_dict(self):
        return asdict(self)


def allowed(T, device, mask_self, key_valid=None):
    """(1|B, 1, T, T) bool: causal; mask_self also removes the diagonal (except row 0, which would be empty)."""
    ok = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
    if mask_self:
        eye = torch.eye(T, dtype=torch.bool, device=device)
        eye[0, 0] = False
        ok = ok & ~eye
    ok = ok.view(1, 1, T, T)
    if key_valid is not None:
        ok = ok & key_valid.bool().view(-1, 1, 1, T)
    return ok


def entropy(a, ok):
    """Mean normalized entropy of the rows of a (B,H,T,T), in units of log(number of visible keys)."""
    p = a / a.sum(-1, keepdim=True).clamp_min(1e-12)
    h = -(p * p.clamp_min(1e-12).log()).sum(-1)
    n = ok.float().sum(-1).clamp_min(2.0).expand_as(h)
    return (h / n.log()).mean()


def rope(v, omega, pos):
    """Rotate plane k of v (B, H, T, d) by omega_k * pos_t."""
    B, H, T, d = v.shape
    ang = pos.view(T, 1) * omega.view(1, -1)
    c, s = ang.cos().view(1, 1, T, -1), ang.sin().view(1, 1, T, -1)
    x1, x2 = v.reshape(B, H, T, d // 2, 2).unbind(-1)
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), -1).view(B, H, T, d)


class _Heads(nn.Module):
    def __init__(self, d_model, H):
        super().__init__()
        assert d_model % H == 0
        self.H, self.dh = H, d_model // H
        self.need_weights = False

    def split(self, y):
        B, T, _ = y.shape
        return y.view(B, T, self.H, -1).transpose(1, 2)

    def merge(self, y):
        B, H, T, d = y.shape
        return y.transpose(1, 2).reshape(B, T, H * d)


# ------------------------------------------------------------------ softmax, gated, sigmoid

class SM(_Heads):
    gated = False
    alibi = False                                                  # ALIBI: linear biases instead of RoPE (Press et al. 2022)

    def __init__(self, cfg: Cfg, d_model, H):
        super().__init__(d_model, H)
        self.cfg = cfg
        self.W_q = nn.Linear(d_model, d_model * (2 if self.gated else 1), bias=False)   # GATED: gate = extra outputs
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=True)
        self.register_buffer("omega", 10000.0 ** (-torch.arange(0, self.dh, 2).float() / self.dh), persistent=False)
        if self.alibi == "learned":                                # ALIBIL: one learned slope per head, kept positive
            init = alibi_slopes(H) if H > 1 else torch.tensor([0.5])   # one head: the steepest published slope, 2^-1
            self.log_slopes = nn.Parameter(init.log())
        elif self.alibi:
            self.register_buffer("slopes", alibi_slopes(H), persistent=False)

    def forward(self, x, key_valid=None):
        B, T, _ = x.shape
        qg = self.W_q(x)
        if self.gated:
            qg = qg.view(B, T, self.H, 2 * self.dh)
            q, gate = qg[..., :self.dh].transpose(1, 2), qg[..., self.dh:].transpose(1, 2)
        else:
            q = self.split(qg)
        k, v = self.split(self.W_k(x)), self.split(self.W_v(x))
        pos = torch.arange(T, device=x.device, dtype=torch.float32)
        ok = allowed(T, x.device, self.cfg.mask_self, key_valid)
        if self.alibi:
            dist = (pos.view(T, 1) - pos.view(1, T)).clamp_min(0.0)
            sl = self.log_slopes.exp() if self.alibi == "learned" else self.slopes
            bias = torch.where(ok, -sl.view(1, -1, 1, 1) * dist, torch.tensor(float("-inf"), device=x.device))
        else:
            q, k = rope(q, self.omega, pos), rope(k, self.omega, pos)
            bias = None
        a, dbg = None, {}
        if self.need_weights:
            logits = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
            logits = logits + bias if bias is not None else logits.masked_fill(~ok, float("-inf"))
            a = torch.nan_to_num(torch.softmax(logits, -1), nan=0.0)
            y = a @ v
            dbg["entropy"] = entropy(a.detach(), ok)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=(bias if bias is not None else ok))
            y = torch.nan_to_num(y, nan=0.0)                       # a row with no visible key (padding) reads nothing
        if self.gated:
            g = torch.sigmoid(gate)
            y = y * g
            dbg["gate_mean"] = g.detach().mean()
        return self.W_o(self.merge(y)), a, dbg


class GATED(SM):
    gated = True


class ALIBI(SM):
    alibi = True


class ALIBIL(SM):
    alibi = "learned"


def alibi_slopes(H):
    """Press et al. 2022: geometric slopes 2^(-8 h / H), h = 1..H (H a power of two; otherwise their interpolation)."""
    def pow2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * start ** i for i in range(n)]
    if math.log2(H).is_integer():
        s = pow2(H)
    else:
        c = 2 ** math.floor(math.log2(H))
        s = pow2(c) + pow2(2 * c)[0::2][: H - c]
    return torch.tensor(s, dtype=torch.float32)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class SIGMOID(_Heads):
    def __init__(self, cfg: Cfg, d_model, H):
        super().__init__(d_model, H)
        self.cfg = cfg
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=True)
        self.q_norm, self.k_norm = RMSNorm(self.dh), RMSNorm(self.dh)
        self.register_buffer("slopes", alibi_slopes(H), persistent=False)
        self.b = -math.log(cfg.max_len)

    def forward(self, x, key_valid=None):
        B, T, _ = x.shape
        q, k, v = self.q_norm(self.split(self.W_q(x))), self.k_norm(self.split(self.W_k(x))), self.split(self.W_v(x))
        pos = torch.arange(T, device=x.device, dtype=torch.float32)
        dist = (pos.view(T, 1) - pos.view(1, T)).clamp_min(0.0)
        logits = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh) - self.slopes.view(1, -1, 1, 1) * dist + self.b
        ok = allowed(T, x.device, self.cfg.mask_self, key_valid)
        a = torch.sigmoid(logits) * ok
        y = a @ v
        dbg = {"wsum": a.detach().sum(-1)}
        return self.W_o(self.merge(y)), (a if self.need_weights else None), dbg


# ------------------------------------------------------------------ PG (as published)

def sinusoidal(n, d):
    pos = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


class PG(_Heads):
    """LoheAttention: W_ij = softplus(q_i.k_j / sqrt d_h) (causal), x_i* = normalize(sum_j W_ij m_j) with m_j the
    learned anchor of position j on S^{d_osc-1}, a_ij = (1 + <x_i*, m_j>) normalized, output sum_j a_ij v_j, W_o.
    Identical to bench/models.py PG with qkv=full, pg_p=1 (tests.py)."""

    def __init__(self, cfg: Cfg, d_model, H):
        super().__init__(d_model, H)
        self.cfg = cfg
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=True)
        self.anchors = nn.Parameter(F.normalize(torch.randn(H, max(cfg.max_len, cfg.pos_len), cfg.pg_d_osc), dim=-1))

    def forward(self, x, key_valid=None):
        B, T, _ = x.shape
        if T > self.anchors.shape[1]:
            raise RuntimeError(f"PG: {T} positions and only {self.anchors.shape[1]} anchors (it cannot extrapolate)")
        q, k, v = self.split(self.W_q(x)), self.split(self.W_k(x)), self.split(self.W_v(x))
        ok = allowed(T, x.device, self.cfg.mask_self, key_valid)
        W = F.softplus(q @ k.transpose(-1, -2) / math.sqrt(self.dh)) * ok
        anc = self.anchors[:, :T]
        x_star = F.normalize(torch.einsum("bhij,hjd->bhid", W, anc), dim=-1, eps=1e-8)
        a = (1.0 + torch.einsum("bhid,hjd->bhij", x_star, anc)).clamp(min=0.0)
        a = a * allowed(T, x.device, self.cfg.mask_self)          # as in the original: the readout keeps padding keys
        a = a / a.sum(-1, keepdim=True).clamp(min=1e-8)
        y = a @ v
        return self.W_o(self.merge(y)), (a if self.need_weights else None), {"entropy": entropy(a.detach(), ok)}


# ------------------------------------------------------------------ KM, the radio head

def solve_mu(K2m, D2, Dn, iters=20):
    """Implicit width: D = mu (1 - sum_j K2m_ij / (mu^2 + D2_ij)), mu = lambda + D.  For D > 0 the root is unique and
    g(mu) = mu (1 - S(mu)) increases strictly; bisection without gradients, then one differentiable Newton step
    (implicit-function theorem).  Same as bench/models.py solve_mu_noise, except that the bisection evaluates only
    g (not its derivative) and stops at 20 iterations instead of 24: on trained models mu reaches float32 precision
    after 14, and on random inputs from deep silence to strong locking (T = 512) the error after 20 is < 1e-5
    (solver_check.py)."""
    Dv = Dn.view(1, -1, 1)

    def g(mu):
        den = mu.unsqueeze(-1) ** 2 + D2
        S = (K2m / den).sum(-1)
        dS = (-2.0 * mu.unsqueeze(-1) * K2m / den.square()).sum(-1)
        return mu * (1.0 - S), 1.0 - S - mu * dS

    with torch.no_grad():
        Dd = Dv.detach().expand(K2m.shape[0], -1, K2m.shape[2])
        lo = torch.zeros_like(Dd)
        hi = Dd + K2m.sum(-1).sqrt() + 1e-6
        for _ in range(iters):
            m = 0.5 * (lo + hi)
            below = m * (1.0 - (K2m / (m.unsqueeze(-1) ** 2 + D2)).sum(-1)) < Dd
            lo, hi = torch.where(below, m, lo), torch.where(below, hi, m)
        mu0 = 0.5 * (lo + hi)
    gv, dg = g(mu0)
    return mu0 - (gv - Dv) / dg.clamp_min(1e-12)


def explicit_mu(K2m, D2, Dn, steps=2):
    """Explicit width, no loop over iterations: an effective station with the same sum K2m/D2 and sum K2m/D2^2,
    locked with line width D (the cubic (mu - D)(mu^2 + Db^2) = KG2 mu, largest real root by Cardano), then `steps`
    closed Newton steps on ln S(mu) = ln(1 - D/mu) in ln mu.  Same as bench/models.py explicit_mu."""
    Dv = Dn.view(1, -1, 1)
    sc = torch.maximum(K2m.sum(-1), (1e-3 * Dv).square()).clamp_min(1e-24).sqrt()
    k2 = K2m / sc.unsqueeze(-1) ** 2
    d2 = (D2 / sc.unsqueeze(-1) ** 2).clamp_min(1e-6)
    dd = Dv / sc
    Ts = (k2 / d2).sum(-1)
    U = (k2 / d2.square()).sum(-1).clamp_min(1e-30)
    Db2, KG2 = Ts / U, Ts.square() / U
    c2, c1, c0 = -dd, Db2 - KG2, -dd * Db2
    p = c1 - c2.square() / 3
    q = 2 * c2 ** 3 / 27 - c2 * c1 / 3 + c0
    disc = (q / 2).square() + (p / 3) ** 3
    one = disc > 0
    sq = torch.where(one, disc, torch.ones_like(disc)).sqrt()
    cbrt = lambda z: torch.sign(z) * (z.abs() + 1e-12) ** (1.0 / 3)
    t1 = cbrt(-q / 2 + sq) + cbrt(-q / 2 - sq)
    pm = torch.where(one, torch.ones_like(p), (-p / 3).clamp_min(1e-12))
    arg = (-q / 2 / pm ** 1.5).clamp(-1 + 1e-6, 1 - 1e-6)
    t3 = 2 * pm.sqrt() * torch.cos(torch.arccos(arg) / 3)
    x = torch.where(one, t1, t3) - c2 / 3
    x = torch.maximum(x, dd * (1 + 1e-4))
    for _ in range(steps):
        den = x.unsqueeze(-1) ** 2 + d2
        a = k2 / den
        S = a.sum(-1).clamp_min(1e-30)
        c = (a * x.unsqueeze(-1) ** 2 / den).sum(-1) / S
        gap = (x - dd).clamp_min(1e-6 * dd)
        phi = torch.log(S) - torch.log(gap / x)
        x = torch.maximum(x * torch.exp(phi / (2 * c + dd / gap)), dd * (1 + 1e-6))
    return torch.maximum(x, dd * (1 + 1e-6)) * sc


class KM(_Heads):
    """The radio head.  Per head, d_h = 2m real coordinates read as C^m (J = multiplication by i, plane by plane).
    Token j: unit content v_j = W_v x_j / |.| (keys are the values), ageing at eps_h, detuning
    Delta_ij = alpha_i - eps_h (i - j), dial alpha_i = W_alpha x_i.  Antenna a_i = W_q x_i / |.|, gain
    G_ij^2 = rho + (1 - rho) |<a_i|v_j>|^(2n).  Width mu_i from sum_j K^2 G_ij^2 / (mu_i^2 + Delta_ij^2) = 1 - D/mu_i,
    D = delta K.  Output g sum_j w_ij v_j with w_ij = K G_ij / (mu_i + i Delta_ij), then W_o.
    Init ('head'): the H heads of every layer age geometrically from eps = 1 to 1/T, dial at age 1.5, K/K_c = 1.5 for a
    typical context, n = 1, gain 1."""

    def __init__(self, cfg: Cfg, d_model, H):
        super().__init__(d_model, H)
        self.cfg, self.m = cfg, self.dh // 2
        T = cfg.max_len
        self.W_q = None if cfg.km_noant else nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=True)
        self.W_alpha = nn.Linear(d_model, H, bias=True)
        eps = torch.full((H,), T ** -0.5) if (H == 1 or cfg.km_same) else torch.exp(math.log(1.0 / T) * torch.arange(H).float() / (H - 1))
        S = sum((L - 1.5) ** -2 for L in range(1, T))
        g = 1.0 if cfg.km_noant else math.sqrt(cfg.km_rho + (1 - cfg.km_rho) * 2.0 / self.dh)
        if cfg.km_noage:                     # every token at the same detuning alpha: K_c = alpha / (g sqrt(n)), n ~ T/2
            eps = torch.zeros(H)
            alpha0 = torch.ones(H)
            K = cfg.km_k0 * alpha0 / (g * math.sqrt(T / 2))
        else:
            alpha0 = 1.5 * eps
            K = cfg.km_k0 * eps / (g * math.sqrt(S))
        self.log_eps = nn.Parameter(eps.clamp_min(1e-30).log(), requires_grad=not cfg.km_noage)
        if not cfg.km_noant:
            self.raw_rho = nn.Parameter(torch.full((H,), math.log(cfg.km_rho / (1.0 - cfg.km_rho))))
            self.log_n = nn.Parameter(torch.zeros(H))
        self.log_K = nn.Parameter(K.log())
        self.log_gain = nn.Parameter(torch.zeros(H))
        self._alpha0 = alpha0
        self.reset_mechanism()

    def reset_mechanism(self):
        nn.init.zeros_(self.W_alpha.weight)
        with torch.no_grad():
            self.W_alpha.bias.copy_(self._alpha0)

    def eps(self):
        return torch.zeros_like(self.log_eps) if self.cfg.km_noage else self.log_eps.exp()

    def forward(self, x, key_valid=None):
        B, T, _ = x.shape
        H, cfg = self.H, self.cfg
        unit = lambda y: F.normalize(self.split(y).float(), dim=-1)
        v = unit(self.W_v(x))
        if cfg.km_noant:
            G2 = torch.ones(1, 1, T, T, device=x.device)
        else:
            a = unit(self.W_q(x))
            Jv = torch.stack((-v[..., 1::2], v[..., 0::2]), -1).flatten(-2)
            ov2 = (a @ v.transpose(-1, -2)).square() + (a @ Jv.transpose(-1, -2)).square()      # |<a_i|v_j>|^2
            rho = torch.sigmoid(self.raw_rho).view(1, H, 1, 1)
            G2 = rho + (1.0 - rho) * ov2.clamp(1e-12, 1.0).pow(self.log_n.exp().view(1, H, 1, 1))
        alpha = self.W_alpha(x).float().transpose(1, 2)                                         # (B,H,T)
        pos = torch.arange(T, device=x.device, dtype=torch.float32)
        age = (pos.view(T, 1) - pos.view(1, T)).clamp_min(0.0)
        D = alpha.unsqueeze(-1) - self.eps().view(1, H, 1, 1) * age
        ok = allowed(T, x.device, cfg.mask_self, key_valid).to(D.dtype)
        K = self.log_K.exp()
        Kv = K.view(1, H, 1, 1)
        G2m = G2 * ok
        K2m = Kv ** 2 * G2m
        D2 = D.square()
        Dn = cfg.km_delta * K
        mu = explicit_mu(K2m, D2, Dn, cfg.km_explicit) if cfg.km_explicit else solve_mu(K2m, D2, Dn)
        le = mu.unsqueeze(-1)
        den = le ** 2 + D2
        G = (G2m + 1e-12).sqrt() * ok
        yr, yi = Kv * G * le / den, -Kv * G * D / den
        w2 = K2m / den                                                                          # |w_ij|^2
        wsum = w2.sum(-1)
        if cfg.km_norm:                                   # control: no silent branch, sum_j |w_ij|^2 = 1 always
            s = wsum.clamp_min(1e-12).rsqrt().unsqueeze(-1)
            yr, yi, w2 = yr * s, yi * s, w2 * s.square()
        vr, vi = v[..., 0::2], v[..., 1::2]                                                     # sum_j w_ij v_j, i -> J
        zr, zi = yr @ vr - yi @ vi, yr @ vi + yi @ vr
        z = torch.stack((zr, zi), -1).flatten(-2) * self.log_gain.exp().view(1, H, 1, 1)
        dbg = {"KoverKc": (K2m / D2.clamp_min(1e-8)).sum(-1).sqrt().detach(), "wsum": wsum.detach(),
               "mu": (mu / K.view(1, H, 1)).detach(), "G2sum": G2m.sum(-1).detach()}
        return self.W_o(self.merge(z).to(x.dtype)), (w2.detach() if self.need_weights else None), dbg


LAYERS = {"SM": SM, "ALIBI": ALIBI, "ALIBIL": ALIBIL, "GATED": GATED, "SIGMOID": SIGMOID, "PG": PG, "KM": KM}
