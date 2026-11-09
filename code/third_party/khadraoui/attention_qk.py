"""The three attention mechanisms we compare: the softmax baseline, the
zero-frequency model and the frequency model."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-8
_EPS_CUSP = 1e-6
_EPS_DIR = 1e-18


_GRAD_PROBE = False


def set_grad_probe(on: bool = True) -> None:
    """Record what gradient reaches each stage, into the dict last_debug
    returns."""
    global _GRAD_PROBE
    _GRAD_PROBE = bool(on)


def grad_probe_on() -> bool:
    return _GRAD_PROBE


def _probe(store, name: str, t: torch.Tensor) -> torch.Tensor:
    """Stash the size of dL/dt per token under name, one number per query,
    when the probe is on."""
    if store is None or not torch.is_grad_enabled() or not t.requires_grad:
        return t

    def hook(gr, _n=name, _s=store):
        gr = gr.detach().float()
        _s[_n] = gr.abs() if gr.dim() <= 3 else gr.norm(dim=-1)

    t.register_hook(hook)
    return t


def _is_compiling() -> bool:
    """True while torch.compile is tracing, when diagnostics cannot be built."""
    try:
        return torch.compiler.is_compiling()
    except Exception:
        return False


def build_bias_mask(B: int, T: int, device, dtype, causal: bool,
                    key_pad: Optional[torch.Tensor], mask_self: bool = False) -> torch.Tensor:
    """Additive mask for softmax attention: zero where a query may look, a
    large negative number where it may not."""
    neg = torch.finfo(dtype).min
    bias = torch.zeros(T, T, device=device, dtype=dtype)
    if causal:
        causal_block = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), 1)
        bias = bias.masked_fill(causal_block, neg)
    if mask_self:
        eye = torch.eye(T, device=device, dtype=torch.bool)
        if causal:
            eye[0, 0] = False
        bias = bias.masked_fill(eye, neg)
    bias = bias.unsqueeze(0).unsqueeze(0).expand(B, 1, T, T)
    if key_pad is not None:
        pad = (~key_pad.bool()).view(B, 1, 1, T)
        bias = bias.masked_fill(pad, neg)
    return bias


def build_allowed(B: int, T: int, device, dtype, causal: bool,
                  key_pad: Optional[torch.Tensor], mask_self: bool = False) -> torch.Tensor:
    """Boolean mask for the oscillator readouts, which multiply by zero
    rather than adding a negative bias."""
    allowed = torch.ones(T, T, device=device, dtype=torch.bool)
    if causal:
        allowed = torch.tril(allowed)
    if mask_self:
        eye = torch.eye(T, device=device, dtype=torch.bool)
        if causal:
            eye[0, 0] = False
        allowed = allowed & ~eye
    allowed = allowed.unsqueeze(0).unsqueeze(0).expand(B, 1, T, T)
    if key_pad is not None:
        allowed = allowed & key_pad.bool().view(B, 1, 1, T)
    return allowed.to(dtype)


READOUTS = ("affine", "resonant", "exponential")

COUPLINGS = ("softplus", "rational", "none")


def coupling_rational(s: torch.Tensor) -> torch.Tensor:
    """Piecewise approximation of exp(s)."""
    dt = s.dtype
    s = s.float()
    neg = 1.0 / (1.0 - s + 0.5 * s * s)
    return torch.where(s >= 0.0, s + 1.0, neg).to(dt)


def coupling_weights(scores, kind: str) -> torch.Tensor:
    """Turn raw coupling scores into the weights that build the field.
    softplus and rational are positive, none keeps the sign."""
    if kind == "softplus":
        return F.softplus(scores)
    if kind == "rational":
        return coupling_rational(scores)
    if kind == "none":
        return scores
    raise ValueError(f"unknown coupling {kind!r}, expected one of {COUPLINGS}")


def readout_weights(zm: torch.Tensor, z: torch.Tensor, allowed: torch.Tensor,
                    kind: str, p: float, gam2=None, beta=None) -> torch.Tensor:
    """Turn the quantity <z, m_c> into attention weights, by the affine,
    resonant or exponential read-out."""
    if kind == "resonant":
        zn2 = (z * z).sum(-1, keepdim=True).to(zm.dtype)
        shifted = 1.0 / (gam2 + (zn2 + 1.0 - 2.0 * zm).clamp(min=0.0))
    elif kind == "exponential":
        s = torch.where(allowed > 0, beta * zm, torch.full_like(zm, -float("inf")))
        mx = s.detach().amax(dim=-1, keepdim=True)
        shifted = torch.exp(s - torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx)))
    else:
        shifted = (1.0 + zm).clamp(min=0.0)
    if p != 1.0 and kind != "exponential":
        shifted = shifted.pow(p)
    shifted = shifted * allowed
    denom = shifted.sum(dim=-1, keepdim=True, dtype=torch.float32)
    return shifted / (denom + 1e-8).to(zm.dtype)


def coupling_stats(scores: torch.Tensor, w: torch.Tensor,
                   allowed: torch.Tensor) -> dict:
    """Per token summaries of a coupling row, averaged over the allowed
    positions only."""
    ok = allowed > 0
    na = allowed.sum(dim=-1).clamp_min(1.0)
    wm = (w * allowed).sum(dim=-1) / na
    ws = ((((w - wm.unsqueeze(-1)) ** 2) * allowed).sum(dim=-1) / na).clamp_min(0.0).sqrt()
    sm = (scores * allowed).sum(dim=-1) / na
    ss = ((((scores - sm.unsqueeze(-1)) ** 2) * allowed).sum(dim=-1) / na).clamp_min(0.0).sqrt()
    return {
        "w_mean": wm.detach(), "w_std": ws.detach(),
        "w_rel": (ws / (wm + _EPS)).detach(),
        "w_max": w.masked_fill(~ok, 0.0).amax(dim=-1).detach(),
        "w_zero": (((w == 0) & ok).sum(dim=-1) / na).detach(),
        "s_mean": sm.detach(), "s_std": ss.detach(),
        "s_max": scores.masked_fill(~ok, torch.finfo(scores.dtype).min).amax(dim=-1).detach(),
    }


def coupling_at(kind: str, s: float) -> float:
    """The coupling response at a single score, off the graph, for sizing."""
    if kind == "softplus":
        return math.log1p(math.exp(-abs(s))) + max(s, 0.0)
    if kind == "rational":
        return s + 1.0 if s >= 0.0 else 1.0 / (1.0 - s + 0.5 * s * s)
    raise ValueError(f"coupling {kind!r} has no size to measure, expected one "
                     f"of {COUPLINGS[:2]}")


def initial_N(cfg) -> float:
    """The coupling total a row carries before training, used to initialize 
    the coupling strength K."""
    T = float(getattr(cfg, "max_seq_len", 1))
    drop_self = 1.0 if bool(getattr(cfg, "mask_self", False)) else 0.0
    reads_all = (str(getattr(cfg, "task", "lm")) == "lm"
                 or str(getattr(cfg, "pooling", "mean")) != "index")
    wide = bool(getattr(cfg, "causal", False)) and reads_all
    n = (0.5 * (T + 1.0) if wide else T) - drop_self
    return max(n, 1.0) * coupling_at(str(getattr(cfg, "coupling", "softplus")),
                                     float(getattr(cfg, "coupling_bias_init", 0.0)))


def initial_coherence(anchors: torch.Tensor, cfg) -> float:
    """The initial coherence R_init."""
    m = F.normalize(anchors.detach().float(), dim=-1)
    T = m.shape[1]
    drop_self = 1.0 if bool(getattr(cfg, "mask_self", False)) else 0.0
    reads_all = (str(getattr(cfg, "task", "lm")) == "lm"
                 or str(getattr(cfg, "pooling", "mean")) != "index")
    if bool(getattr(cfg, "causal", False)) and reads_all:
        s = torch.cumsum(m, dim=1) - drop_self * m
        n = torch.arange(1, T + 1, device=m.device, dtype=m.dtype) - drop_self
    else:
        s = (m.sum(dim=1) - drop_self * m[:, -1]).unsqueeze(1)
        n = torch.full((1,), T - drop_self, device=m.device, dtype=m.dtype)
    keep = n > 0
    return float((s[:, keep].norm(dim=-1) / n[keep]).mean().clamp(1e-6, 1.0))


def inv_softplus(y: float) -> float:
    """The raw value whose softplus is y, for starting a positive parameter."""
    y = float(y)
    assert y > 0.0, f"a positive parameter needs a positive start, got {y}"
    return y if y > 20.0 else math.log(math.expm1(y))


def settled_direction(h: torch.Tensor, h_floor: float, fallback: torch.Tensor,
                      scale: Optional[torch.Tensor] = None,
                      R_floor: float = 0.0) -> torch.Tensor:
    """The settled state an oscillator reaches, with a positive floor."""
    n2 = (h * h).sum(dim=-1, keepdim=True)
    if h_floor > 0.0:
        return h / torch.sqrt(n2 + h_floor ** 2)
    big = n2 > _EPS_DIR * _EPS_DIR
    z = torch.where(big, h / torch.sqrt(torch.where(big, n2, torch.ones_like(n2))),
                    fallback.to(h.dtype))
    if R_floor > 0.0 and scale is not None:
        den = torch.maximum(n2.sqrt(), R_floor * scale).clamp_min(_EPS).detach()
        zd = z.detach()
        tangent = (h - zd * (zd * h).sum(dim=-1, keepdim=True)) / den
        z = torch.where(big, zd + tangent, z)
    return z


def complex_structure(d_osc: int) -> torch.Tensor:
    """The fixed matrix rotating each coordinate pair by 90 degrees."""
    J = torch.zeros(d_osc, d_osc)
    idx = torch.arange(0, d_osc, 2)
    J[idx + 1, idx] = 1.0
    J[idx, idx + 1] = -1.0
    return J


ANCHOR_INITS = ("ring", "harmonics", "random")

_RING_NOISE = 0.1


def anchor_geometry(kind: str, d_osc: int, T: int, num_heads: int) -> torch.Tensor:
    """The directions the anchors start from, one per head and position, a
    uniform circle ring at d_osc 2 and differing only in the planes above it."""
    assert kind in ANCHOR_INITS, \
        f"unknown anchor init {kind!r}, expected one of {ANCHOR_INITS}"
    if kind == "random":
        return F.normalize(torch.randn(num_heads, T, d_osc), dim=-1)
    theta = torch.linspace(0, 2 * math.pi, T + 1)[:-1]
    raw = torch.zeros(num_heads, T, d_osc)
    raw[:, :, 0], raw[:, :, 1] = theta.cos(), theta.sin()
    if d_osc == 2:
        return raw
    if kind == "harmonics":
        for i in range(1, d_osc // 2):
            raw[:, :, 2 * i] = torch.cos((i + 1) * theta)
            raw[:, :, 2 * i + 1] = torch.sin((i + 1) * theta)
        rest = d_osc - 2 * (d_osc // 2)
    else:
        rest = d_osc - 2
    if rest:
        raw[:, :, d_osc - rest:] = _RING_NOISE * torch.randn(num_heads, T, rest)
    return F.normalize(raw, dim=-1)


class AnchorBank(nn.Module):
    """The anchors m_j, one learned direction per head and position, read as
    the mean fields of synchronised clusters in the rotating frame."""

    def __init__(self, d_model: int, d_osc: int, max_seq_len: int, num_heads: int,
                 freeze: bool = False, init: str = "harmonics"):
        super().__init__()
        self.num_heads = num_heads
        self.d_osc = d_osc
        self.init = str(init)
        raw = anchor_geometry(self.init, d_osc, max_seq_len, num_heads)
        self.anchors = nn.Parameter(raw, requires_grad=not freeze)

    def forward(self, embeddings: torch.Tensor, T: int) -> torch.Tensor:
        m = F.normalize(self.anchors[:, :T].float(), dim=-1)
        return m.unsqueeze(0).expand(embeddings.size(0), -1, -1, -1)

    def get_anchors(self):
        """Return the learned anchor directions for inspection."""
        return F.normalize(self.anchors, dim=-1).detach()


class SoftmaxAttention(nn.Module):
    """Standard multi head attention on the fused PyTorch kernel, optionally
    with rope so a dot product depends on the offset."""

    def __init__(self, cfg):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.d_head = cfg.d_head
        self.causal = cfg.causal
        self.mask_self = bool(getattr(cfg, "mask_self", False))
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.num_heads * cfg.d_head, bias=False)
        self.rope = bool(getattr(cfg, "rope", False))
        if self.rope:
            assert self.d_head % 2 == 0, "rope needs an even head dimension"
            base = float(getattr(cfg, "rope_base", 10000.0))
            inv = base ** (-torch.arange(0, self.d_head, 2).float() / self.d_head)
            self.register_buffer("rope_inv_freq", inv, persistent=False)

    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate each coordinate pair of a (B, H, T, d_head) tensor by its position angle."""
        T = x.size(-2)
        t = torch.arange(T, device=x.device, dtype=torch.float32)
        ang = torch.outer(t, self.rope_inv_freq)
        cos, sin = ang.cos(), ang.sin()
        xf = x.float().view(*x.shape[:-1], -1, 2)
        x1, x2 = xf.unbind(-1)
        out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        return out.flatten(-2).to(x.dtype)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.num_heads, self.d_head).unbind(2)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        if self.rope:
            q = self._apply_rope(q)
            k = self._apply_rope(k)
        attn_bias = None
        if mask is not None or self.mask_self:
            attn_bias = build_bias_mask(B, T, x.device, q.dtype, self.causal, mask,
                                        self.mask_self)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.causal and attn_bias is None)
        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.d_head)
        return out, None


class ZeroFrequencyAttention(nn.Module):
    """Oscillator attention with no frequency, the base paper model. A token
    settles on its local field and the readout scores every anchor."""

    def __init__(self, cfg):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.d_head = cfg.d_head
        self.d_osc = cfg.d_osc
        self.p = cfg.p
        self.causal = cfg.causal
        self.mask_self = bool(getattr(cfg, 'mask_self', False))
        self.eps_cusp = float(getattr(cfg, 'eps_cusp', _EPS_CUSP))
        self.R_floor = float(getattr(cfg, 'R_floor', 0.0))
        self.coupling = 'softplus'
        assert self.coupling in COUPLINGS, \
            f"unknown coupling {self.coupling!r}, expected one of {COUPLINGS}"
        self.readout = str(getattr(cfg, 'readout', 'affine'))
        assert self.readout in READOUTS, \
            f"unknown readout {self.readout!r}, expected one of {READOUTS}"
        self.res_gamma = float(getattr(cfg, 'res_gamma', 0.3))
        self.res_gamma_min = float(getattr(cfg, 'res_gamma_min', 0.05))
        if self.readout == 'resonant' and bool(getattr(cfg, 'res_gamma_learnable', False)):
            g0 = max(self.res_gamma - self.res_gamma_min, 1e-3)
            raw = math.log(math.expm1(g0))
            self.res_gamma_raw = nn.Parameter(torch.full((cfg.num_heads,), raw))
        else:
            self.res_gamma_raw = None
        self.exp_beta = float(getattr(cfg, 'exp_beta', 8.0))
        self.exp_beta_min = float(getattr(cfg, 'exp_beta_min', 0.0))
        if self.readout == 'exponential' and bool(getattr(cfg, 'exp_beta_learnable', False)):
            b0 = max(self.exp_beta - self.exp_beta_min, 1e-3)
            self.exp_beta_raw = nn.Parameter(torch.full((cfg.num_heads,),
                                                        math.log(math.expm1(b0))))
        else:
            self.exp_beta_raw = None
        self.scale = math.sqrt(self.d_head)
        self.F_proj = nn.Linear(cfg.d_model, cfg.num_heads * cfg.d_head, bias=False)
        self.G_proj = nn.Linear(cfg.d_model, cfg.num_heads * cfg.d_head, bias=False)
        self.W_V = nn.Linear(cfg.d_model, cfg.num_heads * cfg.d_head, bias=False)
        self.anchor_bank = AnchorBank(cfg.d_model, self.d_osc, cfg.max_seq_len,
                                      cfg.num_heads,
                                      freeze=getattr(cfg, "freeze_anchors", False),
                                      init=getattr(cfg, "anchor_init", "harmonics"))
        if cfg.freeze_wv:
            self.W_V.weight.requires_grad_(False)

    def get_anchors(self):
        """Return the learned anchor geometry for inspection."""
        return self.anchor_bank.get_anchors()

    def _gamma2(self, dt):
        """The squared width of the resonant readout, learned per head or fixed."""
        if self.res_gamma_raw is None:
            return self.res_gamma ** 2
        g_ = self.res_gamma_min + F.softplus(self.res_gamma_raw.float())
        return (g_ * g_).view(1, -1, 1, 1).to(dt)

    def _beta(self, dt):
        """The sharpness of the exponential readout, learned per head or fixed."""
        if self.exp_beta_raw is None:
            return self.exp_beta
        b_ = self.exp_beta_min + F.softplus(self.exp_beta_raw.float())
        return b_.view(1, -1, 1, 1).to(dt)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, T, _ = x.shape
        f = self.F_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        g = self.G_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(f, g.transpose(-2, -1)) / self.scale
        dt = scores.dtype
        if dt != torch.float32:
            dt = torch.float32
            scores = scores.float()
        allowed = build_allowed(B, T, x.device, dt, self.causal, mask, self.mask_self)
        gs = {} if _GRAD_PROBE else None
        _probe(gs, "g_score", scores)
        w = coupling_weights(scores, self.coupling) * allowed
        _probe(gs, "g_w", w)

        m32 = self.anchor_bank(x, T)
        m = m32.to(dt)
        h = torch.einsum("bhtj,bhjd->bhtd", w, m).float()
        _probe(gs, "g_h", h)
        h_norm = torch.sqrt((h * h).sum(dim=-1) + self.eps_cusp)
        z_star = settled_direction(h, 0.0, m, scale=w.sum(dim=-1, keepdim=True),
                                   R_floor=self.R_floor)

        _probe(gs, "g_z", z_star)
        zm = torch.einsum("bhtd,bhjd->bhtj", z_star.to(dt), m)
        attn = readout_weights(zm, z_star, allowed, self.readout, self.p,
                               gam2=self._gamma2(dt), beta=self._beta(dt))
        _probe(gs, "g_attn", attn)

        v = self.W_V(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        out = torch.matmul(attn.to(v.dtype), v)
        _probe(gs, "g_out", out)
        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.d_head)

        if _is_compiling():
            return out, attn
        N = w.float().sum(dim=-1)
        debug = gs if gs is not None else {}
        debug.update({
            "h_norm": h_norm.detach(), "z_norm": z_star.norm(dim=-1).detach(),
            "psi": F.normalize(h, dim=-1).detach(), "z": z_star.detach(),
            "N": N.detach(),
            "coh": (h_norm / (N + _EPS)).clamp(0.0, 1.0).detach(),
            "degenerate": ((h * h).sum(dim=-1)
                           <= _EPS_DIR ** 2).float().detach(),
        })
        if not self.training:
            debug.update(coupling_stats(scores, w, allowed))
        return out, attn, debug




"""CONTROL VARIANT of attention.py (PRL, 1 Oct 2026): the only change is in FrequencyAttention, queries and
keys of width cfg.d_qk (default d_head, i.e. identical to attention.py).  diff attention.py attention_qk.py."""
class FrequencyAttention(nn.Module):
    """Oscillator attention with the frequency on, our model. A token locks
    at a phase lag or drifts and shrinks; at omega 0 it is the base model."""

    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_osc >= 2 and cfg.d_osc % 2 == 0, "d_osc must be even so coordinates pair up"
        self.num_heads = cfg.num_heads
        self.d_head = cfg.d_head
        self.d_osc = cfg.d_osc
        self.p = cfg.p
        self.causal = cfg.causal
        self.mask_self = bool(getattr(cfg, 'mask_self', False))
        self.h_floor = float(getattr(cfg, 'h_floor', 0.0))
        self.eps_cusp = float(getattr(cfg, 'eps_cusp', _EPS_CUSP))
        self.coupling = str(getattr(cfg, 'coupling', 'softplus'))
        assert self.coupling in COUPLINGS, \
            f"unknown coupling {self.coupling!r}, expected one of {COUPLINGS}"
        self.normalized_couplings = bool(getattr(cfg, 'normalized_couplings', False))
        assert not (self.normalized_couplings and self.coupling == 'none'), \
            "a coupling strength has to be split, so signed couplings cannot carry one"
        self._K_init = float(getattr(cfg, 'K_init', 0.0))
        if self.normalized_couplings and self._K_init <= 0.0:
            self._K_init = initial_N(cfg)
        self.w_bias = (nn.Parameter(torch.full((cfg.num_heads,),
                                    float(getattr(cfg, 'coupling_bias_init', 0.0))))
                       if bool(getattr(cfg, 'coupling_bias', False)) else None)
        self.readout = str(getattr(cfg, 'readout', 'affine'))
        assert self.readout in READOUTS, \
            f"unknown readout {self.readout!r}, expected one of {READOUTS}"
        self.res_gamma = float(getattr(cfg, 'res_gamma', 0.3))
        self.res_gamma_min = float(getattr(cfg, 'res_gamma_min', 0.05))
        if self.readout == 'resonant' and bool(getattr(cfg, 'res_gamma_learnable', False)):
            g0 = max(self.res_gamma - self.res_gamma_min, 1e-3)
            raw = math.log(math.expm1(g0))
            self.res_gamma_raw = nn.Parameter(torch.full((cfg.num_heads,), raw))
        else:
            self.res_gamma_raw = None
        self.exp_beta = float(getattr(cfg, 'exp_beta', 8.0))
        self.exp_beta_min = float(getattr(cfg, 'exp_beta_min', 0.0))
        if self.readout == 'exponential' and bool(getattr(cfg, 'exp_beta_learnable', False)):
            b0 = max(self.exp_beta - self.exp_beta_min, 1e-3)
            self.exp_beta_raw = nn.Parameter(torch.full((cfg.num_heads,),
                                                        math.log(math.expm1(b0))))
        else:
            self.exp_beta_raw = None
        # CONTROL VARIANT (PRL, 1 Oct 2026): queries and keys of width d_qk instead of d_head
        self.d_qk = int(getattr(cfg, "d_qk", 0)) or self.d_head
        self.scale = math.sqrt(self.d_qk)
        self.F_proj = nn.Linear(cfg.d_model, cfg.num_heads * self.d_qk, bias=False)
        self.G_proj = nn.Linear(cfg.d_model, cfg.num_heads * self.d_qk, bias=False)
        self.W_V = nn.Linear(cfg.d_model, cfg.num_heads * cfg.d_head, bias=False)
        self.anchor_bank = AnchorBank(cfg.d_model, self.d_osc, cfg.max_seq_len,
                                      cfg.num_heads,
                                      freeze=getattr(cfg, "freeze_anchors", False),
                                      init=getattr(cfg, "anchor_init", "harmonics"))
        self.register_buffer("J", complex_structure(self.d_osc))
        self.omega_proj = nn.Linear(cfg.d_model, cfg.num_heads)
        self._omega_bias_init = float(getattr(cfg, "omega_bias_init", 0.0))
        self._omega_init_std = float(getattr(cfg, "omega_init_std", 0.0))
        self.omega_bias_relative = bool(getattr(cfg, "omega_bias_relative", False))
        if self.omega_bias_relative and not bool(getattr(cfg, "rho_direct", False)):
            N = self._K_init if self.normalized_couplings else initial_N(cfg)
            h0 = N * initial_coherence(self.anchor_bank.anchors, cfg)
            self._omega_bias_init *= h0 / max(float(cfg.omega_gain_init), _EPS)
        self.K_learnable = bool(getattr(cfg, 'K_learnable', False))
        self.K_proj = (nn.Linear(cfg.d_model, cfg.num_heads)
                       if self.normalized_couplings and self.K_learnable else None)
        if self.normalized_couplings and self.K_proj is None:
            self.register_buffer("K_fixed", torch.tensor(float(self._K_init)),
                                 persistent=False)
        else:
            self.K_fixed = None
        self.reset_mechanism_parameters()
        self.omega_per_head = bool(getattr(cfg, "omega_gain_per_head", False))
        gain = float(cfg.omega_gain_init)
        self.omega_scale = nn.Parameter(torch.full((cfg.num_heads,), gain)
                                        if self.omega_per_head else torch.tensor(gain))
        self.rho_direct = bool(getattr(cfg, "rho_direct", False))
        self.rho_cap = float(getattr(cfg, "rho_cap", 0.0))
        if cfg.freeze_wv:
            self.W_V.weight.requires_grad_(False)

    def get_anchors(self):
        """Return the learned anchor geometry for inspection."""
        return self.anchor_bank.get_anchors()

    def _gamma2(self, dt):
        """The squared width of the resonant readout, learned per head or fixed."""
        if self.res_gamma_raw is None:
            return self.res_gamma ** 2
        g_ = self.res_gamma_min + F.softplus(self.res_gamma_raw.float())
        return (g_ * g_).view(1, -1, 1, 1).to(dt)

    def _beta(self, dt):
        """The sharpness of the exponential readout, learned per head or fixed."""
        if self.exp_beta_raw is None:
            return self.exp_beta
        b_ = self.exp_beta_min + F.softplus(self.exp_beta_raw.float())
        return b_.view(1, -1, 1, 1).to(dt)

    def reset_mechanism_parameters(self):
        """Start the frequency head at the prior lag. Must be called again after
        any global init sweep, which would overwrite it."""
        if self._omega_init_std > 0.0:
            nn.init.normal_(self.omega_proj.weight, std=self._omega_init_std)
        else:
            nn.init.zeros_(self.omega_proj.weight)
        nn.init.constant_(self.omega_proj.bias, self._omega_bias_init)
        if self.K_proj is not None:
            nn.init.zeros_(self.K_proj.weight)
            nn.init.constant_(self.K_proj.bias, inv_softplus(self._K_init))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, T, _ = x.shape
        f = self.F_proj(x).view(B, T, self.num_heads, self.d_qk).transpose(1, 2)
        g = self.G_proj(x).view(B, T, self.num_heads, self.d_qk).transpose(1, 2)
        scores = torch.matmul(f, g.transpose(-2, -1)) / self.scale
        dt = scores.dtype
        if dt != torch.float32:
            dt = torch.float32
            scores = scores.float()
        if self.w_bias is not None:
            scores = scores + self.w_bias.view(1, -1, 1, 1).to(scores.dtype)
        allowed = build_allowed(B, T, x.device, dt, self.causal, mask, self.mask_self)
        gs = {} if _GRAD_PROBE else None
        _probe(gs, "g_score", scores)
        w = coupling_weights(scores, self.coupling) * allowed
        _probe(gs, "g_w", w)

        m32 = self.anchor_bank(x, T)
        m = m32.to(dt)
        if not self.normalized_couplings:
            K = None
            h = torch.einsum("bhtj,bhjd->bhtd", w, m).float()
        else:
            wn = w / w.sum(dim=-1, keepdim=True).clamp_min(_EPS)
            K = (self.K_fixed if self.K_proj is None
                 else F.softplus(self.K_proj(x).float()).transpose(1, 2))
            h = torch.einsum("bhtj,bhjd->bhtd", wn, m).float() * K.unsqueeze(-1)
        _probe(gs, "g_h", h)
        h_norm = torch.sqrt((h * h).sum(dim=-1) + self.eps_cusp)
        if self.h_floor > 0.0:
            h_den = torch.sqrt(h_norm * h_norm + self.h_floor ** 2)
        else:
            h_den = h_norm + _EPS
        psi = settled_direction(h, self.h_floor, m)
        Jpsi = torch.einsum("de,bhte->bhtd", self.J.float(), psi)

        gain = self.omega_scale.float()
        if self.omega_per_head:
            gain = gain.view(1, -1, 1)
        omega = gain * self.omega_proj(x).float().transpose(1, 2)
        _probe(gs, "g_omega", omega)
        if self.rho_direct:
            rho = omega
        else:
            rho = omega / h_den
        _probe(gs, "g_rho", rho)
        if self.rho_cap > 0.0:
            rho = self.rho_cap * torch.tanh(rho / self.rho_cap)
        rho2 = rho * rho
        locked = rho2 <= 1.0

        align = torch.sqrt(torch.clamp(1.0 - rho2, min=0.0) + self.eps_cusp)
        align = torch.where(locked, align, torch.zeros_like(align))
        lag_drift = torch.sign(rho) / (rho.abs()
                                       + torch.sqrt(torch.clamp(rho2 - 1.0, min=0.0)
                                                    + self.eps_cusp))
        lag = torch.where(locked, rho, lag_drift)
        z_eff = align.unsqueeze(-1) * psi + lag.unsqueeze(-1) * Jpsi
        _probe(gs, "g_z", z_eff)

        zm = torch.einsum("bhtd,bhjd->bhtj", z_eff.to(dt), m)
        attn = readout_weights(zm, z_eff, allowed, self.readout, self.p,
                               gam2=self._gamma2(dt), beta=self._beta(dt))
        _probe(gs, "g_attn", attn)

        v = self.W_V(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        out = torch.matmul(attn.to(v.dtype), v)
        _probe(gs, "g_out", out)
        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.d_head)

        if _is_compiling():
            debug = None
        else:
            debug = gs if gs is not None else {}
            debug.update({
                "rho": rho.detach(), "omega": omega.detach(), "h_norm": h_norm.detach(),
                "align": align.detach(), "lag": lag.detach(), "z_norm": z_eff.norm(dim=-1).detach(),
                "psi": F.normalize(h, dim=-1).detach(), "z": z_eff.detach(),
                "locked_frac": locked.float().mean().detach(),
                "degenerate": ((h * h).sum(dim=-1)
                               <= _EPS_DIR ** 2).float().detach(),
            })
            N = w.float().sum(dim=-1)
            debug["N"] = N.detach()
            if K is None:
                debug["coh"] = (h_norm / (N + _EPS)).clamp(0.0, 1.0).detach()
            else:
                debug["coh"] = (h_norm / K.clamp_min(_EPS)).clamp(0.0, 1.0).detach()
                debug["K"] = (K if K.dim() else
                              K.reshape(1, 1, 1).expand(B, self.num_heads, T)).detach()
            if not self.training:
                debug.update(coupling_stats(scores, w, allowed))
        return out, attn, debug


def build_attention(cfg) -> nn.Module:
    """Pick the attention module named by cfg.attn_type."""
    if cfg.attn_type == "softmax":
        return SoftmaxAttention(cfg)
    if cfg.attn_type == "zero_frequency":
        return ZeroFrequencyAttention(cfg)
    if cfg.attn_type == "frequency":
        return FrequencyAttention(cfg)
    raise ValueError(f"unknown attn_type {cfg.attn_type!r}")
