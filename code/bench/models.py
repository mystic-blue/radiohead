"""Els models del banc (9 oct 2026).  Un sol esquelet (Model), una capa d'atenció intercanviable.

    SM   softmax + RoPE                                            la referència
    PG   Pasqualetti & Guo, TMLR 2026: LoheAttention tal com està publicada (àncores apreses per posició,
         acoblament softplus(q.k/sqrt d), punt fix normalitzat, lectura afí (1 + <x*, m_j>)^p normalitzada);
         codi oficial github.com/FabioPasqualetti/attention-by-synchronization (oscillator_attention/attention.py)
    TH   Khadraoui, TFG URV 2026: FrequencyAttention amb la configuració del seu quadern (acoblament racional amb biaix,
         acoblaments normalitzats amb K fix, àncores harmòniques apreses, h_floor, freqüència projectada, lectura
         ressonant amb gamma apresa); codi oficial github.com/aissamdev/attention-by-oscillator-synchronization
    AD   la sonda d'Adler sobre un rellotge de la síntesi (nucli 1, = M2 de l'etapa 2)
    RS   selecció per ressonància amb una freqüència per pla (radio_p, nucli 2)
    GM   reso_noq_gm de PRL_sync (el model del draft curt): estrella de Kuramoto amb envelliment, beta<v_i,v_j> a la
         freqüència, sortida = estat sincronitzat
    KM   l'antena (draft antenna_model, 9 oct): el contingut a l'estat, la posició a la freqüència, la comparació al
         guany de l'antena; sortida = estat relaxat del circuit (forma tancada de l'autoconsistència de Kuramoto)

Tots reben x (B, T, d_model) normalitzat i key_valid, i tornen (sortida, pesos (B, H, T, T) o res, diagnòstics).
La màscara (causal, i la diagonal a les tasques sintètiques) és del protocol i és la mateixa per a tots; GM i KM no
escolten mai el propi token (j < i), com a la seva definició.

Escala d'ablació comuna (cfg.qkv), a tots els models on té sentit:
    full   Q, K i V amb projeccions pròpies
    qk     sense Q: Q = K (V a part)                       (SM, AD, PG, TH)
    v      una sola projecció: Q = K = V = W_v x          (amb Q = K, la diagonal s'emmascara)
    x      cap projecció: Q = K = V = x, la llesca del capçal del corrent residual (normalitzat)
    W_o es manté sempre; les projeccions escalars de sintonia (omega de TH, alpha de KM, W_a de RS, W_c de GM) també:
    són freqüències, no consultes.  GM i RS ja no tenen Q ni K: full = v.  KM: full té Q (l'antena) i K = V.  Les equivalències amb el codi original són a tests.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, replace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

_EPS = 1e-8


@dataclass
class Cfg:
    model: str = "SM"            # SM | PG | TH | AD | AD0 | DG | RS | GM | KM
    qkv: str = "full"            # variants Q/K/V (vegeu cada model)
    d_osc: int = 0               # amplada de l'oscil·lador per capçal (0 = el valor per defecte del model)
    mask_self: bool = False      # tasques sintètiques: la diagonal emmascarada (protocol)
    task_kind: str = "lm"        # lm | seq | cls (TH: la mida inicial de K depèn de si es llegeix tota la fila)
    max_len: int = 128           # PG, TH: nombre d'àncores (una per posició entrenada)
    pos_len: int = 0             # PG, TH: àncores fins a la longitud de test més llarga (les no vistes, sense entrenar)
    # AD
    beta_init: float = 4.0
    gamma: float = 0.4
    # RS
    rs_blocks: int = 0
    rs_K: float = 0.3
    rs_gamma: float = 0.02
    rs_eps: tuple = (0.01, 0.3)
    rs_newton: int = 3
    rs_chunk: int = 8
    # PG
    pg_p: int = 1
    # TH (quadern: inducció gamma 0.4; llenguatge gamma 0.15, omega_bias 0.8 relatiu)
    th_gamma: float = 0.4
    th_gamma_min: float = 0.01
    th_omega_bias: float = 0.0
    th_omega_relative: bool = False
    th_h_floor: float = 1e-2
    # GM, KM (solucionador comú)
    K_init: float = 0.3
    lam_floor: float = 0.02      # lambda >= lam_floor * K (amplada de línia intrínseca; vegeu el draft, gradients)
    newton: int = 6
    eps_lo: float = 0.01         # ritmes d'envelliment inicials, geomètrics entre capçals (KM) o plans (GM)
    eps_hi: float = 0.3
    # KM
    an_rho: float = 0.1          # fons isòtrop inicial de l'antena (G^2 = rho + (1 - rho)|<a|v>|^2), après per capçal
    an_age: str = "head"         # head: un ritme per capçal | token: ritme del token des dels plans (com GM)
    be_tau0: float = 1.0         # BE: temperatura inicial del nucli de Planck (per capçal, apresa)
    km_init: str = "v1"          # v1: la del 9 oct (eps 0.01-0.3 entre capçals, alpha 0, K 0.3) | phys: vegeu KM.init_phys
    km_noise: bool = False       # soroll de freqüència D (après per capçal) en lloc del terra dur de lambda
    km_pow: bool = False         # antena directiva: |<a|v>|^2 -> |<a|v>|^(2n), n = exp(log_n) après per capçal (inici n = 1)
    km_delta: float = 0.0        # > 0: amplada de línia fixa D = delta K (la mateixa per a tots els capçals; no s'aprèn)
    km_explicit: int = 0         # amplada explícita: estació efectiva (cúbica amb soroll) + km_explicit passos de Newton tancats
    layer: int = 0               # capa i nombre de capes (els posa Model: la inicialització phys reparteix escales)
    n_layers: int = 1

    def to_dict(self):
        return asdict(self)


# ------------------------------------------------------------------ peces comunes

def allowed(T, device, mask_self, key_valid=None, B=1):
    ok = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
    if mask_self:
        eye = torch.eye(T, dtype=torch.bool, device=device)
        eye[0, 0] = False
        ok = ok & ~eye
    ok = ok.view(1, 1, T, T)
    if key_valid is not None:
        ok = ok & key_valid.bool().view(-1, 1, 1, T)
    return ok


def rope(v, omega, pos, sign=-1.0):
    """Rota el pla k de v (B, H, T, d) un angle sign * omega_k * pos_t."""
    B, H, T, d = v.shape
    ang = sign * pos.view(T, 1) * omega.view(1, -1)
    c, s = ang.cos().view(1, 1, T, -1), ang.sin().view(1, 1, T, -1)
    x1, x2 = v.reshape(B, H, T, d // 2, 2).unbind(-1)
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), -1).view(B, H, T, d)


def spectrum(d, base=10000.0):
    k = torch.arange(d // 2, dtype=torch.float32)
    return base ** (-2.0 * k / d)


def entropy(a, ok):
    p = a.clamp_min(1e-12)
    h = -(a * p.log()).sum(-1)
    n = ok.float().sum(-1).clamp_min(2.0).expand_as(h)
    return (h / n.log()).mean()


# ------------------------------------------------------------------ la física de la sonda (etapa 3, settle.py, literal)

LAM_MIN, CUSP = 1e-3, 1e-6


def _planes(h):
    return h.view(*h.shape[:-1], h.shape[-1] // 2, 2)


def _J(hk):
    return torch.stack((-hk[..., 1], hk[..., 0]), dim=-1)


def _solve_lambda(a, eta2, iters=10):
    with torch.no_grad():
        R2 = a.sum(-1).clamp_min(1e-30)
        a_, e_ = a / R2.unsqueeze(-1), eta2 / R2.unsqueeze(-1)
        floor = (LAM_MIN ** 2) / R2
        mu = torch.maximum(1.0 - (a_ * e_).sum(-1), floor)
        for _ in range(iters):
            den = mu.unsqueeze(-1) + e_
            N = (a_ / den).sum(-1)
            dN = (a_ / (den * den)).sum(-1)
            mu = torch.maximum(mu + N * (N - 1.0) / dN.clamp_min(1e-30), floor)
        return (mu * R2).sqrt()


def resolvent(h, eta):
    """Estat assentat de la sonda dz/dt = DeltaOmega z + (I - z z^T) h, DeltaOmega = diag(eta_k J).
    Enganxada (N(0+) >= 1): z_k = (lam h_k + eta_k J h_k) / (lam^2 + eta_k^2), N(lam) = 1.
    Deriva: la fórmula pla a pla de rope_osc.py (la mateixa que a les etapes 2-3)."""
    hk = _planes(h)
    a = (hk * hk).sum(-1)
    eta2 = eta * eta
    N0 = (a / eta2.clamp_min(1e-12)).sum(-1)
    locked = N0 >= 1.0
    lam0 = _solve_lambda(a, eta2)
    den0 = lam0.unsqueeze(-1) ** 2 + eta2
    Nl = (a / den0).sum(-1)
    dN = -2.0 * lam0 * (a / (den0 * den0)).sum(-1)
    good = locked & (dN < -1e-6)
    step = (Nl - 1.0) / torch.where(good, dN, -torch.ones_like(dN))
    lam = (lam0 - torch.where(good, step, torch.zeros_like(step))).clamp_min(LAM_MIN)
    Jh = _J(hk)
    den = (lam.unsqueeze(-1) ** 2 + eta2).unsqueeze(-1)
    z_lock = (lam.unsqueeze(-1).unsqueeze(-1) * hk + eta.unsqueeze(-1) * Jh) / den
    u = (1.0 / N0.clamp_min(1e-12)).sqrt().clamp(1.0, 1e6)
    mag = 1.0 / (u + torch.sqrt((u * u - 1.0).clamp_min(0.0) + CUSP))
    dirv = Jh / eta.abs().clamp_min(1e-6).unsqueeze(-1) * torch.sign(eta).unsqueeze(-1)
    dirv = dirv / dirv.flatten(-2).norm(dim=-1).clamp_min(1e-6).unsqueeze(-1).unsqueeze(-1)
    z_drift = mag.unsqueeze(-1).unsqueeze(-1) * dirv
    z = torch.where(locked.unsqueeze(-1).unsqueeze(-1), z_lock, z_drift).flatten(-2)
    return z, lam, locked


# ------------------------------------------------------------------ projeccions Q/K/V comunes

class QKV(nn.Module):
    """Projeccions de consulta, clau i valor segons la variant.  Retorna (q, k, v) amb forma (B, H, T, ·)."""

    def __init__(self, cfg: Cfg, d_model, H, dqk, dv):
        super().__init__()
        self.cfg, self.H, self.dqk, self.dv = cfg, H, dqk, dv
        v = cfg.qkv
        if v in ("v", "x"):                                  # escala d'ablació: una projecció (v) o cap (x)
            self.W_q = self.W_k = None
            self.W_v = nn.Linear(d_model, H * dv, bias=False) if v == "v" else None
            assert v == "v" or H * dv == d_model
            return
        if v != "q0":
            self.W_q = None if v in ("qk",) else nn.Linear(d_model, H * dqk, bias=False)
        else:
            self.W_q = None
            self.q_const = nn.Parameter(torch.randn(H, dqk) * (1.0 / math.sqrt(dqk)))
        self.W_k = nn.Linear(d_model, H * dqk, bias=False)
        if v == "kv":
            assert dqk == dv
            self.W_v = None
        elif v == "v1":
            assert H * dv == d_model
            self.W_v = None
        else:
            self.W_v = nn.Linear(d_model, H * dv, bias=False)

    def heads(self, y, d):
        B, T, _ = y.shape
        return y.view(B, T, self.H, d).transpose(1, 2)

    def forward(self, x):
        B, T, _ = x.shape
        if self.cfg.qkv in ("v", "x"):
            u = self.heads(self.W_v(x) if self.W_v is not None else x, self.dv)
            return u, u, u
        k = self.heads(self.W_k(x), self.dqk)
        if self.cfg.qkv == "qk":
            q = k
        elif self.cfg.qkv == "q0":
            q = self.q_const.view(1, self.H, 1, self.dqk).expand(B, self.H, T, self.dqk)
        else:
            q = self.heads(self.W_q(x), self.dqk)
        if self.cfg.qkv == "kv":
            v = k
        elif self.cfg.qkv == "v1":
            v = self.heads(x, self.dv)
        else:
            v = self.heads(self.W_v(x), self.dv)
        return q, k, v

    @property
    def tied_qk(self):
        return self.cfg.qkv in ("qk", "v", "x")


# ------------------------------------------------------------------ SM

class SM(nn.Module):
    def __init__(self, cfg: Cfg, d_model, H, max_len):
        super().__init__()
        self.cfg, self.H, self.dh = cfg, H, d_model // H
        self.p = QKV(cfg, d_model, H, self.dh, self.dh)
        self.W_o = nn.Linear(d_model, d_model, bias=True)
        self.register_buffer("omega", 10000.0 ** (-torch.arange(0, self.dh, 2).float() / self.dh), persistent=False)

    def forward(self, x, key_valid=None):
        B, T, _ = x.shape
        q, k, v = self.p(x)
        pos = torch.arange(T, device=x.device, dtype=torch.float32)
        q, k = rope(q, self.omega, pos, +1.0), rope(k, self.omega, pos, +1.0)
        ok = allowed(T, x.device, self.cfg.mask_self or self.p.tied_qk, key_valid)
        logits = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        a = torch.nan_to_num(torch.softmax(logits.masked_fill(~ok, float("-inf")), -1), nan=0.0)
        out = (a @ v).transpose(1, 2).reshape(B, T, -1)
        return self.W_o(out), a, {"entropy": entropy(a.detach(), ok)}


# ------------------------------------------------------------------ AD, AD0, DG

class AD(nn.Module):
    """Sonda d'Adler sobre un rellotge.  acoblament w_ij = softplus(beta q_i.k_j/sqrt(d) + b) normalitzat;
    camp h_i = sum_j w_ij r_j amb r_j = R(-omega j) e0;  la sonda s'assenta (resolvent) amb la desintonia
    eta_ik = eps_i omega_k + eta0_i;  lectura a_ij ~ (gamma^2 + |z_i - r_j|^2)^-4;  sortida sum_j a_ij v_j."""

    def __init__(self, cfg: Cfg, d_model, H, max_len):
        super().__init__()
        self.cfg, self.H, self.dh, self.d = cfg, H, d_model // H, cfg.d_osc or d_model // H
        self.p = QKV(cfg, d_model, H, self.dh, self.dh)
        self.W_o = nn.Linear(d_model, d_model, bias=True)
        self.e0 = nn.Parameter(F.normalize(torch.randn(H, self.d), dim=-1))
        self.register_buffer("omega", spectrum(self.d), persistent=False)
        self.beta = nn.Parameter(torch.full((H,), float(cfg.beta_init)))
        self.b0 = nn.Parameter(torch.zeros(H))
        self.g = cfg.gamma                                                  # amplada fixa, com a lorentz4 de l'etapa 2
        if cfg.model == "AD":
            self.eps_proj, self.eta0_proj = nn.Linear(d_model, H), nn.Linear(d_model, H)
            self.eps_gain = nn.Parameter(torch.tensor(1.0))
            self.eps_scale = 1.0 / max_len
        if cfg.model == "DG":
            self.shift_proj, self.phase_proj = nn.Linear(d_model, H), nn.Linear(d_model, H)
        self.reset_mechanism()

    def reset_mechanism(self):
        for n in ("eps_proj", "eta0_proj", "shift_proj", "phase_proj"):
            if hasattr(self, n):
                nn.init.zeros_(getattr(self, n).weight); nn.init.zeros_(getattr(self, n).bias)

    def clock(self, T, device):
        pos = torch.arange(T, device=device, dtype=torch.float32)
        e = F.normalize(self.e0, dim=-1).view(1, self.H, 1, self.d).expand(1, self.H, T, self.d)
        return rope(e, self.omega, pos)

    def settle(self, h, x):
        B, H, T, _ = h.shape
        if self.cfg.model == "AD0":
            R = h.norm(dim=-1, keepdim=True)
            return h / R.clamp_min(1e-6), {"locked": torch.ones(B, H, T, dtype=torch.bool, device=h.device)}
        if self.cfg.model == "DG":
            hn = h / h.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            ang = (self.shift_proj(x).transpose(1, 2).unsqueeze(-1) * self.omega.view(1, 1, 1, -1)
                   + self.phase_proj(x).transpose(1, 2).unsqueeze(-1))
            c, s = ang.cos(), ang.sin()
            x1, x2 = hn.view(B, H, T, -1, 2).unbind(-1)
            z = torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), -1).view(B, H, T, -1)
            return z, {"locked": torch.ones(B, H, T, dtype=torch.bool, device=h.device)}
        eps = self.eps_scale * self.eps_gain * self.eps_proj(x).float().transpose(1, 2)
        eta0 = self.eta0_proj(x).float().transpose(1, 2)
        eta = eps.unsqueeze(-1) * self.omega.view(1, 1, 1, -1) + eta0.unsqueeze(-1)
        z, lam, locked = resolvent(h, eta)
        return z, {"locked": locked, "lam": lam, "eps": eps, "eta0": eta0}

    def forward(self, x, key_valid=None):
        B, T, _ = x.shape
        q, k, v = self.p(x)
        ok = allowed(T, x.device, self.cfg.mask_self or self.p.tied_qk, key_valid)
        s = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(self.dh)
        s = self.beta.view(1, -1, 1, 1) * s + self.b0.view(1, -1, 1, 1)
        w = F.softplus(s) * ok
        w = w / w.sum(-1, keepdim=True).clamp_min(_EPS)
        r = self.clock(T, x.device)
        h = torch.einsum("bhtj,bhjd->bhtd", w, r.expand(B, -1, -1, -1))
        z, dbg = self.settle(h, x)
        zr = torch.einsum("bhtd,bhjd->bhtj", z, r.expand(B, -1, -1, -1))
        dist2 = ((z * z).sum(-1, keepdim=True) + 1.0 - 2.0 * zr).clamp_min(0.0)
        a = (1.0 / (self.g * self.g + dist2)).pow(4.0) * ok
        a = a / a.sum(-1, keepdim=True).clamp_min(_EPS)
        out = (a.to(v.dtype) @ v).transpose(1, 2).reshape(B, T, -1)
        dbg = {k_: v_.detach() for k_, v_ in dbg.items()}
        dbg["entropy"] = entropy(a.detach(), ok)
        dbg["R"] = h.norm(dim=-1).detach()
        return self.W_o(out), a, dbg


# ------------------------------------------------------------------ RS

class RS(nn.Module):
    """Selecció per ressonància (radio_p).  Per mode mu (pla o bloc de plans):
        Delta_ijmu = a_imu + c_mu (p_jmu - p_imu) - eps_mu (i - j)  [+ beta q_i.k_j/sqrt(d) amb qkv = 'dot']
        sum_j,mu p_jmu K^2 / (lam_i^2 + Delta^2) = 1   (lam_i; per sota del llindar, lam al terra gamma K)
        z_i,mu = g K sum_j v_jmu (lam_i - i Delta_ijmu) / (lam_i^2 + Delta^2)     (notació complexa per pla), després W_o.
    p_jmu = |v_jmu|^2 amb v unitari (la clau és el valor); 'k': p de W_k x unitari."""

    def __init__(self, cfg: Cfg, d_model, H, max_len):
        super().__init__()
        self.cfg, self.H = cfg, H
        self.d = cfg.d_osc or d_model // H
        self.m = self.d // 2
        nm = cfg.rs_blocks or self.m
        assert self.m % nm == 0
        self.nm = nm
        v = cfg.qkv
        if v in ("v1", "x"):                                 # 'v' = 'full': RS ja no té Q ni K
            assert H * self.d == d_model
            self.W_v = None
        else:
            self.W_v = nn.Linear(d_model, H * self.d, bias=False)
        if v == "k":
            self.W_k = nn.Linear(d_model, H * self.d, bias=False)
        if v == "dot":
            self.W_q = nn.Linear(d_model, H * self.d, bias=False)
            self.W_k = nn.Linear(d_model, H * self.d, bias=False)
            self.log_beta = nn.Parameter(torch.zeros(H))
        self.W_o = nn.Linear(H * self.d, d_model, bias=True)
        self.log_K = nn.Parameter(torch.full((H,), math.log(cfg.rs_K)))
        self.log_gain = nn.Parameter(torch.zeros(H))
        lo, hi = cfg.rs_eps
        self.radio_eps = nn.Parameter(torch.logspace(math.log10(lo), math.log10(hi), nm).repeat(H, 1))
        self.radio_c = nn.Parameter(torch.zeros(H, nm))
        if v == "q0":
            self.radio_a = nn.Parameter(torch.zeros(H, nm))
        else:
            self.W_a = nn.Linear(d_model, H * nm, bias=True)
        self.reset_mechanism()

    def reset_mechanism(self):
        if hasattr(self, "W_a"):
            nn.init.zeros_(self.W_a.weight); nn.init.zeros_(self.W_a.bias)

    def _D(self, u, w, mu):
        D = u[..., mu].unsqueeze(2) - w[..., mu].unsqueeze(3)              # (B,H,T,T,mc)
        if self._ov is not None:
            D = D + self._ov.unsqueeze(-1)
        return D

    def _G(self, u, w, p, okm, s, mu):
        K2 = self.log_K.exp().view(1, self.H, 1) ** 2
        r = 1.0 / (s.unsqueeze(-1).unsqueeze(-1) + self._D(u, w, mu).square())
        qq = p[..., mu].unsqueeze(2) * okm * r
        return K2 * qq.sum((-1, -2)), -K2 * (qq * r).sum((-1, -2))

    def _S(self, u, w, p, okm, mu):
        r = 1.0 / self._D(u, w, mu).square().clamp_min(1e-8)
        qq = p[..., mu].unsqueeze(2) * okm * r
        return qq.sum((-1, -2)), (qq * r).sum((-1, -2))

    def _read(self, u, w, okm, lam, vr, vi, mu):
        K = self.log_K.exp().view(1, self.H, 1, 1, 1)
        D = self._D(u, w, mu)
        le = lam.unsqueeze(-1).unsqueeze(-1)
        r = (K / (le * le + D * D)) * okm
        cr, ci = r * le, -r * D
        if vr.dim() == 5:
            a_r, a_i = vr[:, :, :, mu].unsqueeze(2), vi[:, :, :, mu].unsqueeze(2)
            cr, ci = cr.unsqueeze(-1), ci.unsqueeze(-1)
        else:
            a_r, a_i = vr[..., mu].unsqueeze(2), vi[..., mu].unsqueeze(2)
        return (cr * a_r - ci * a_i).sum(3), (cr * a_i + ci * a_r).sum(3)

    def _shares(self, u, w, p, okm, lam, mu):
        K2 = self.log_K.exp().view(1, self.H, 1, 1, 1) ** 2
        le = lam.unsqueeze(-1).unsqueeze(-1)
        return (K2 * p[..., mu].unsqueeze(2) * okm / (le * le + self._D(u, w, mu).square())).sum(-1)

    def power(self, y, B, T):
        H, nm, bs = self.H, self.nm, self.m // self.nm
        y = F.normalize(y.view(B, T, H, self.d).transpose(1, 2).float(), dim=-1)
        yr, yi = y[..., 0::2], y[..., 1::2]
        if nm < self.m:
            yr, yi = yr.reshape(B, H, T, nm, bs), yi.reshape(B, H, T, nm, bs)
            return y, yr, yi, (yr * yr + yi * yi).sum(-1)
        return y, yr, yi, yr * yr + yi * yi

    def forward(self, x, key_valid=None):
        B, T, _ = x.shape
        H, nm, cfg = self.H, self.nm, self.cfg
        v_src = x if self.W_v is None else self.W_v(x)
        v, vr, vi, p = self.power(v_src, B, T)
        if cfg.qkv == "k":
            _, _, _, p = self.power(self.W_k(x), B, T)
        if hasattr(self, "W_a"):
            tune = self.W_a(x).float().view(B, T, H, nm).transpose(1, 2)
        else:
            tune = self.radio_a.view(1, H, 1, nm).expand(B, H, T, nm)
        pos = torch.arange(T, device=x.device, dtype=torch.float32)
        ok = allowed(T, x.device, True, key_valid).float()                    # (B|1,1,T,T), diagonal sempre emmascarada
        okm = ok.unsqueeze(-1)
        K = self.log_K.exp().view(1, H, 1)
        c, eps = self.radio_c.view(1, H, 1, -1), self.radio_eps.view(1, H, 1, -1)
        ep = eps * pos.view(1, 1, -1, 1)
        u, w = c * p + ep, c * p - tune + ep
        self._ov = None
        if cfg.qkv == "dot":
            q = self.W_q(x).view(B, T, H, self.d).transpose(1, 2).float()
            k = self.W_k(x).view(B, T, H, self.d).transpose(1, 2).float()
            self._ov = self.log_beta.exp().view(1, H, 1, 1) * (q @ k.transpose(-1, -2)) / math.sqrt(self.d)
        chunks = [slice(s0, min(nm, s0 + cfg.rs_chunk)) for s0 in range(0, nm, cfg.rs_chunk)]
        s_min = (cfg.rs_gamma * K) ** 2 * torch.ones(B, H, T, device=x.device)
        with torch.no_grad():
            S2 = S4 = 0.0
            for mu in chunks:
                a2, a4 = self._S(u, w, p, okm, mu); S2 = S2 + a2; S4 = S4 + a4
            locked = K * K * S2 > 1.0
            s2 = torch.maximum((S2 / S4.clamp_min(1e-30)) * (K * K * S2 - 1.0), s_min)
            for _ in range(cfg.rs_newton):
                G_ = G1_ = 0.0
                for mu in chunks:
                    g0, g1 = self._G(u, w, p, okm, s2, mu); G_ = G_ + g0; G1_ = G1_ + g1
                s2 = torch.maximum(s2 + (G_ * G_ - G_) / (-G1_).clamp_min(1e-30), s_min)
        lam0 = s2.sqrt()
        G = dGs = 0.0
        for mu in chunks:
            g0, g1 = checkpoint(self._G, u, w, p, okm, lam0 * lam0, mu, use_reentrant=False)
            G = G + g0; dGs = dGs + g1
        lam = torch.where(locked, lam0 - (G - 1.0) / (2.0 * lam0 * dGs).clamp(max=-1e-12), lam0)
        lam = torch.maximum(lam, s_min.sqrt())
        zr, zi = [], []
        for mu in chunks:
            r_, i_ = checkpoint(self._read, u, w, okm, lam, vr, vi, mu, use_reentrant=False)
            zr.append(r_); zi.append(i_)
        dimc = -2 if nm < self.m else -1
        zr, zi = torch.cat(zr, dimc), torch.cat(zi, dimc)
        if nm < self.m:
            zr, zi = zr.reshape(B, H, T, self.m), zi.reshape(B, H, T, self.m)
        g = self.log_gain.exp().view(1, H, 1, 1)
        out = torch.stack((zr * g, zi * g), -1).reshape(B, H, T, self.d).transpose(1, 2).reshape(B, T, H * self.d)
        dbg = {"locked": locked.detach(), "lam": (lam / K).detach()}
        if self.training or not getattr(self, "keep_shares", False):
            a = out.new_zeros(1)
        else:
            with torch.no_grad():
                a = sum(self._shares(u, w, p, okm, lam, mu) for mu in chunks)
            dbg["entropy"] = entropy(a / a.sum(-1, keepdim=True).clamp_min(1e-12), ok.bool())
        return self.W_o(out.to(x.dtype)), a, dbg


# ------------------------------------------------------------------ PG (Pasqualetti & Guo, tal com està publicat)

class PG(nn.Module):
    """LoheAttention: W_ij = softplus(q_i.k_j / sqrt(d_h)) (causal), x_i* = normalize(sum_j W_ij m_j) amb m_j l'àncora
    apresa de la posició j a S^{d_osc-1}, a_ij = (1 + <x_i*, m_j>)^p normalitzat, sortida sum_j a_ij v_j i W_o.
    Igual que oscillator_attention.attention.LoheAttention (tests.py); l'única diferència és la màscara del protocol."""

    def __init__(self, cfg: Cfg, d_model, H, max_len):
        super().__init__()
        self.cfg, self.H, self.dh = cfg, H, d_model // H
        self.d_osc = cfg.d_osc or 8                                          # el valor per defecte de la classe oficial
        assert cfg.qkv in ("full", "qk", "v", "x")
        self.W_q = nn.Linear(d_model, H * self.dh, bias=False) if cfg.qkv == "full" else None
        self.W_k = nn.Linear(d_model, H * self.dh, bias=False) if cfg.qkv in ("full", "qk") else None
        self.W_v = nn.Linear(d_model, H * self.dh, bias=False) if cfg.qkv in ("full", "qk", "v") else None
        self.W_o = nn.Linear(H * self.dh, d_model, bias=True)
        self.anchors = nn.Parameter(F.normalize(torch.randn(H, max(cfg.max_len, cfg.pos_len), self.d_osc), dim=-1))

    def forward(self, x, key_valid=None):
        B, T, _ = x.shape
        if T > self.anchors.shape[1]:
            raise RuntimeError(f"PG: {T} posicions i només {self.anchors.shape[1]} àncores (no extrapola)")
        H, dh = self.H, self.dh
        hd = lambda y: y.view(B, T, H, dh).transpose(1, 2)
        v = hd(x if self.W_v is None else self.W_v(x))
        q = hd(self.W_q(x)) if self.W_q is not None else v
        k = hd(self.W_k(x)) if self.W_k is not None else v
        q = q if self.W_q is not None else k
        tied = self.W_q is None
        ok = allowed(T, x.device, self.cfg.mask_self or tied, key_valid)
        W = F.softplus(torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(dh)) * ok
        anc = self.anchors[:, :T]
        x_star = F.normalize(torch.einsum("bhij,hjd->bhid", W, anc), dim=-1, eps=1e-8)
        a = (1.0 + torch.einsum("bhid,hjd->bhij", x_star, anc)).clamp(min=0.0)
        if self.cfg.pg_p != 1:
            a = a ** self.cfg.pg_p
        a = a * allowed(T, x.device, self.cfg.mask_self or tied)            # com a l'original: la lectura no treu el farciment
        a = a / a.sum(-1, keepdim=True).clamp(min=1e-8)
        out = torch.einsum("bhij,bhjd->bhid", a, v).transpose(1, 2).reshape(B, T, H * dh)
        return self.W_o(out), a, {"entropy": entropy(a.detach(), ok)}


# ------------------------------------------------------------------ TH (Khadraoui, tal com està publicat)

def _th_anchors(d_osc, T, H, n=None):
    """anchor_geometry('harmonics'): un anell uniforme al primer pla, harmònics als següents.  n > T: les àncores de les
    posicions no entrenades continuen la mateixa regla (theta = 2 pi j / T), les entrenades no canvien."""
    n = n or T
    theta = 2 * math.pi * torch.arange(n, dtype=torch.float32) / T
    raw = torch.zeros(H, n, d_osc)
    raw[:, :, 0], raw[:, :, 1] = theta.cos(), theta.sin()
    for i in range(1, d_osc // 2):
        raw[:, :, 2 * i] = torch.cos((i + 1) * theta)
        raw[:, :, 2 * i + 1] = torch.sin((i + 1) * theta)
    return F.normalize(raw, dim=-1)


class TH(nn.Module):
    """FrequencyAttention amb la configuració del quadern de la tesi.  w_ij = r(f_i.g_j/sqrt(d_h) + b) (r: aproximació
    racional d'exp), normalitzats, camp h_i = K sum_j w_ij m_j (K = la massa inicial d'una fila, fix); la sonda de
    freqüència omega_i = s * (w . x_i + b0) s'enganxa amb desfasament (rho = omega/|h| <= 1) o deriva i s'encongeix;
    lectura ressonant a_ij ~ 1/(gamma^2 + |z_i - m_j|^2) normalitzada; sortida sum_j a_ij v_j.
    Igual que codes.attention.FrequencyAttention (tests.py)."""

    CUSP = 1e-6

    def __init__(self, cfg: Cfg, d_model, H, max_len):
        super().__init__()
        self.cfg, self.H, self.dh = cfg, H, d_model // H
        self.d_osc = cfg.d_osc or 2                                          # el valor per defecte de la tesi
        assert self.d_osc % 2 == 0
        assert cfg.qkv in ("full", "qk", "v", "x")
        self.F_proj = nn.Linear(d_model, H * self.dh, bias=False) if cfg.qkv == "full" else None
        self.G_proj = nn.Linear(d_model, H * self.dh, bias=False) if cfg.qkv in ("full", "qk") else None
        self.W_V = nn.Linear(d_model, H * self.dh, bias=False) if cfg.qkv in ("full", "qk", "v") else None
        self.W_o = nn.Linear(H * self.dh, d_model, bias=False)                  # W_O sense biaix, com al bloc de la tesi
        self.w_bias = nn.Parameter(torch.zeros(H))
        self.anchors = nn.Parameter(_th_anchors(self.d_osc, cfg.max_len, H, max(cfg.max_len, cfg.pos_len)))
        reads_all = cfg.task_kind != "cls"
        n = (0.5 * (cfg.max_len + 1.0) if reads_all else float(cfg.max_len)) - (1.0 if cfg.mask_self else 0.0)
        self.register_buffer("K_fixed", torch.tensor(max(n, 1.0)), persistent=False)   # x coupling_at(rational, 0) = 1
        self.omega_proj = nn.Linear(d_model, H)
        self.omega_scale = nn.Parameter(torch.tensor(1.0))
        g0 = max(cfg.th_gamma - cfg.th_gamma_min, 1e-3)
        self.res_gamma_raw = nn.Parameter(torch.full((H,), math.log(math.expm1(g0))))
        b = cfg.th_omega_bias
        if cfg.th_omega_relative:                                            # rho inicial mesurat contra el camp inicial
            b *= float(self.K_fixed) * self._initial_coherence()
        self._omega_bias_init = b
        self.reset_mechanism()

    def _initial_coherence(self):
        m = F.normalize(self.anchors.detach().float(), dim=-1)
        T, drop = m.shape[1], (1.0 if self.cfg.mask_self else 0.0)
        if self.cfg.task_kind != "cls":
            s = torch.cumsum(m, dim=1) - drop * m
            n = torch.arange(1, T + 1, dtype=m.dtype) - drop
        else:
            s = (m.sum(dim=1) - drop * m[:, -1]).unsqueeze(1)
            n = torch.full((1,), T - drop, dtype=m.dtype)
        keep = n > 0
        return float((s[:, keep].norm(dim=-1) / n[keep]).mean().clamp(1e-6, 1.0))

    def reset_mechanism(self):
        nn.init.zeros_(self.omega_proj.weight)
        nn.init.constant_(self.omega_proj.bias, self._omega_bias_init)

    def forward(self, x, key_valid=None):
        B, T, _ = x.shape
        if T > self.anchors.shape[1]:
            raise RuntimeError(f"TH: {T} posicions i només {self.anchors.shape[1]} àncores (no extrapola)")
        H, dh, cfg = self.H, self.dh, self.cfg
        hd = lambda y: y.view(B, T, H, dh).transpose(1, 2)
        v = hd(x if self.W_V is None else self.W_V(x))
        f = hd(self.F_proj(x)) if self.F_proj is not None else v
        g = hd(self.G_proj(x)) if self.G_proj is not None else v
        f = f if self.F_proj is not None else g
        s = (f @ g.transpose(-2, -1)).float() / math.sqrt(dh) + self.w_bias.view(1, -1, 1, 1)
        ok = allowed(T, x.device, cfg.mask_self or self.F_proj is None, key_valid).float()
        w = torch.where(s >= 0.0, s + 1.0, 1.0 / (1.0 - s + 0.5 * s * s)) * ok
        m = F.normalize(self.anchors[:, :T].float(), dim=-1).unsqueeze(0)
        wn = w / w.sum(-1, keepdim=True).clamp_min(_EPS)
        h = torch.einsum("bhtj,bhjd->bhtd", wn, m.expand(B, -1, -1, -1)) * self.K_fixed
        n2 = (h * h).sum(-1, keepdim=True)
        h_norm = torch.sqrt(n2.squeeze(-1) + self.CUSP)
        hf = cfg.th_h_floor
        h_den = torch.sqrt(h_norm * h_norm + hf ** 2) if hf > 0 else h_norm + _EPS
        psi = h / torch.sqrt(n2 + hf ** 2) if hf > 0 else F.normalize(h, dim=-1)
        Jpsi = torch.stack((-psi[..., 1::2], psi[..., 0::2]), -1).flatten(-2)
        omega = self.omega_scale * self.omega_proj(x).float().transpose(1, 2)
        rho = omega / h_den
        rho2 = rho * rho
        locked = rho2 <= 1.0
        align = torch.where(locked, torch.sqrt(torch.clamp(1.0 - rho2, min=0.0) + self.CUSP), torch.zeros_like(rho))
        lag = torch.where(locked, rho, torch.sign(rho) / (rho.abs() + torch.sqrt(torch.clamp(rho2 - 1.0, min=0.0) + self.CUSP)))
        z = align.unsqueeze(-1) * psi + lag.unsqueeze(-1) * Jpsi
        zm = torch.einsum("bhtd,bhjd->bhtj", z, m.expand(B, -1, -1, -1))
        gam = cfg.th_gamma_min + F.softplus(self.res_gamma_raw.float())
        a = 1.0 / ((gam * gam).view(1, -1, 1, 1) + ((z * z).sum(-1, keepdim=True) + 1.0 - 2.0 * zm).clamp(min=0.0))
        a = a * ok
        a = a / (a.sum(-1, keepdim=True) + 1e-8)
        out = (a.to(v.dtype) @ v).transpose(1, 2).reshape(B, T, H * dh)
        return self.W_o(out), a, {"locked": locked.detach(), "rho": rho.detach(), "entropy": entropy(a.detach(), ok.bool())}


# ------------------------------------------------------------------ solucionador comú de GM i KM

def solve_lambda(K2m, D2, K, lam_floor, newton):
    """sum_j K2m_ij / (lam_i^2 + D2_ij) = 1 per a lam_i >= lam_floor K (enganxat), o lam_i = lam_floor K (deriva).
    Newton sobre 1/G(s), s = lam^2, des del terra: monòton cap a l'arrel; després un pas diferenciable (derivada
    implícita).  És el solucionador de reso_noq_gm (PRL_sync), literal."""
    s_min = (lam_floor * K.view(1, -1, 1)) ** 2
    Gmin = (K2m / (s_min.unsqueeze(-1) + D2)).sum(-1)
    locked = Gmin > 1.0
    with torch.no_grad():
        s2 = s_min.expand_as(Gmin).clone()
        for _ in range(newton):
            den = s2.unsqueeze(-1) + D2
            G_, G1_ = (K2m / den).sum(-1), (K2m / den.square()).sum(-1)
            s2 = torch.maximum(s2 + (G_ * G_ - G_) / G1_.clamp_min(1e-30), s_min)
    lam0 = s2.sqrt()
    den = lam0.unsqueeze(-1) ** 2 + D2
    G = (K2m / den).sum(-1)
    dG = (-2.0 * lam0.unsqueeze(-1) * K2m / den.square()).sum(-1)
    lam = torch.where(locked, lam0 - (G - 1.0) / dG.clamp(max=-1e-12), lam0)
    return torch.maximum(lam, s_min.sqrt().expand_as(lam)), locked


def solve_mu_noise(K2m, D2, Dn, iters=24):
    """Amb soroll de freqüència D (per capçal):  D = mu (1 - sum_j K2m_ij / (mu^2 + D2_ij)),  mu = lambda + D.
    Per D > 0 la solució és única i g(mu) = mu (1 - S(mu)) hi creix estrictament: la derivada implícita és finita a tot
    arreu (no cal cap terra).  Bisecció sense gradients (el predicat g(mu) < D és monòton), després un pas de Newton
    diferenciable (teorema de la funció implícita)."""
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
            below = g(m)[0] < Dd
            lo, hi = torch.where(below, m, lo), torch.where(below, hi, m)
        mu0 = 0.5 * (lo + hi)
    gv, dg = g(mu0)
    return mu0 - (gv - Dv) / dg.clamp_min(1e-12)


def explicit_mu(K2m, D2, Dn, steps=1):
    """Amplada explícita (draft radio_head, Sec. VI i App. E), sense cap bucle.
    Estació efectiva amb les mateixes T = sum K2m/D2 i U = sum K2m/D2^2: desintonia^2 = T/U, acoblament^2 = T^2/U;
    s'hi enganxa amb amplada de línia D: (mu - D)(mu^2 + Db^2) = KG2 mu, la cúbica amb una sola arrel > D (Cardano,
    l'arrel real més gran); després steps passos de Newton tancats sobre ln S(mu) = ln(1 - D/mu) en ln mu.
    Tot en unitats de sqrt(sum K2m) per condicionar la cúbica.  Dn (H,) > 0."""
    Dv = Dn.view(1, -1, 1)
    # (B,H,T): escala de freqüències; terra relatiu a D (dd = D/sc <= 1e3): un receptor que gairebé no sent res
    # (antena directiva, G^2 ~ 1e-20) dona mu -> D sense desbordar la cúbica en float32
    sc = torch.maximum(K2m.sum(-1), (1e-3 * Dv).square()).clamp_min(1e-24).sqrt()
    k2 = K2m / sc.unsqueeze(-1) ** 2
    d2 = (D2 / sc.unsqueeze(-1) ** 2).clamp_min(1e-6)                         # |Delta| >= 1e-3 sc: sense desbordar 1/Delta^4 en float32
    dd = Dv / sc
    T = (k2 / d2).sum(-1)
    U = (k2 / d2.square()).sum(-1).clamp_min(1e-30)
    Db2, KG2 = T / U, T.square() / U                                           # estació efectiva
    c2, c1, c0 = -dd, Db2 - KG2, -dd * Db2                                     # x^3 + c2 x^2 + c1 x + c0 = 0
    p = c1 - c2.square() / 3
    q = 2 * c2 ** 3 / 27 - c2 * c1 / 3 + c0
    disc = (q / 2).square() + (p / 3) ** 3
    one = disc > 0                                    # cada branca rep valors segurs on no s'usa (gradients finits)
    sq = torch.where(one, disc, torch.ones_like(disc)).sqrt()
    cbrt = lambda x: torch.sign(x) * (x.abs() + 1e-12) ** (1.0 / 3)
    t1 = cbrt(-q / 2 + sq) + cbrt(-q / 2 - sq)                                 # una arrel real
    pm = torch.where(one, torch.ones_like(p), (-p / 3).clamp_min(1e-12))
    arg = (-q / 2 / pm ** 1.5).clamp(-1 + 1e-6, 1 - 1e-6)
    t3 = 2 * pm.sqrt() * torch.cos(torch.arccos(arg) / 3)                      # tres reals: la més gran
    x = torch.where(one, t1, t3) - c2 / 3
    x = torch.maximum(x, dd * (1 + 1e-4))
    for _ in range(steps):                            # passos tancats: ln S(mu) = ln(1 - D/mu)
        den = x.unsqueeze(-1) ** 2 + d2
        a = k2 / den
        S = a.sum(-1).clamp_min(1e-30)
        c = (a * x.unsqueeze(-1) ** 2 / den).sum(-1) / S
        gap = (x - dd).clamp_min(1e-6 * dd)
        phi = torch.log(S) - torch.log(gap / x)
        x = torch.maximum(x * torch.exp(phi / (2 * c + dd / gap)), dd * (1 + 1e-6))
    return torch.maximum(x, dd * (1 + 1e-6)) * sc


def _complex_read(yr, yi, v):
    """sum_j (yr + i yi)_ij v_j, amb i -> J pla a pla.  v (B,H,T,d) -> (B,H,T,d)."""
    vr, vi = v[..., 0::2], v[..., 1::2]
    zr = yr @ vr - yi @ vi
    zi = yr @ vi + yi @ vr
    return torch.stack((zr, zi), -1).flatten(-2)


# ------------------------------------------------------------------ GM (reso_noq_gm)

class GM(nn.Module):
    """Token j: oscil·lador de Lohe en l'estat del seu valor unitari v_j; la freqüència amb què el sent el nou i és
    beta <v_i, v_j> - chi_j (i - j), chi_j = sum_mu chi_mu |v_j,mu|^2.  El nou: freqüència c_i = W_c x_i, acoblament K.
    Delta_ij = beta <v_i,v_j> - c_i - chi_j (i - j);  y_ij = K / (lam_i + i Delta_ij);  sum_j |y_ij|^2 = 1;
    sortida g sum_j y_ij v_j (i -> J), W_o.  Igual que ResonanceLoheAttention(reso_noq_gm) de PRL_sync (tests.py)."""

    def __init__(self, cfg: Cfg, d_model, H, max_len):
        super().__init__()
        self.cfg, self.H = cfg, H
        self.d = cfg.d_osc or d_model // H
        self.m = self.d // 2
        assert cfg.qkv in ("full", "v", "x"), "GM: full = v (K = V, sense Q); x = sense W_v"
        self.W_v = None if cfg.qkv == "x" else nn.Linear(d_model, H * self.d, bias=False)
        assert self.W_v is not None or H * self.d == d_model
        self.W_o = nn.Linear(H * self.d, d_model, bias=True)
        self.W_c = nn.Linear(d_model, H, bias=True)
        self.log_beta = nn.Parameter(torch.zeros(H))
        self.raw_chi_pl = nn.Parameter(torch.logspace(math.log10(cfg.eps_lo), math.log10(cfg.eps_hi), self.m).repeat(H, 1))
        self.log_gain = nn.Parameter(torch.zeros(H))
        self.log_K = nn.Parameter(torch.full((H,), math.log(cfg.K_init)))
        self.reset_mechanism()

    def reset_mechanism(self):
        nn.init.zeros_(self.W_c.weight); nn.init.zeros_(self.W_c.bias)

    def forward(self, x, key_valid=None):
        B, T, _ = x.shape
        H, d, cfg = self.H, self.d, self.cfg
        v = F.normalize((x if self.W_v is None else self.W_v(x)).view(B, T, H, d).transpose(1, 2).float(), dim=-1)
        c = self.W_c(x).float().transpose(1, 2)                                        # (B,H,T)
        en = v[..., 0::2] ** 2 + v[..., 1::2] ** 2
        chi = (en * self.raw_chi_pl.view(1, H, 1, self.m)).sum(-1)                     # (B,H,T)
        pos = torch.arange(T, device=x.device, dtype=torch.float32)
        age = (pos.view(T, 1) - pos.view(1, T)).clamp_min(0.0)
        S = self.log_beta.exp().view(1, H, 1, 1) * (v @ v.transpose(-1, -2))
        D = S - c.unsqueeze(-1) - chi.unsqueeze(-2) * age
        ok = allowed(T, x.device, True, key_valid).to(D.dtype)
        K = self.log_K.exp()
        K2m = K.view(1, H, 1, 1) ** 2 * ok
        D2 = D.square()
        lam, locked = solve_lambda(K2m, D2, K, cfg.lam_floor, cfg.newton)
        le = lam.unsqueeze(-1)
        den = le ** 2 + D2
        Kv = K.view(1, H, 1, 1)
        yr, yi = Kv * le / den * ok, -Kv * D / den * ok
        z = _complex_read(yr, yi, v) * self.log_gain.exp().view(1, H, 1, 1)
        out = z.transpose(1, 2).reshape(B, T, H * d)
        a = K2m / den
        dbg = {"locked": locked.detach(), "lam": (lam / K.view(1, H, 1)).detach(),
               "KoverKc": (K2m / D2.clamp_min(1e-8)).sum(-1).sqrt().detach(),
               "entropy": entropy((a / a.sum(-1, keepdim=True).clamp_min(1e-12)).detach(), ok.bool())}
        return self.W_o(out.to(x.dtype)), a.detach(), dbg


# ------------------------------------------------------------------ KM (l'antena)

class KM(nn.Module):
    """KURAMOTOR: l'antena (draft antenna_model).  Per capçal, d = d_osc parells de coordenades (C^{d/2}), J = gir de 90 graus.

    Emissors (j < i): l'estat és el valor unitari v_j = W_v x_j / |.|; la freqüència és l'edat,
        nu_j(i) = nu - eps_h (i - j)           (an_age = 'token': eps_j = sum_mu chi_h,mu |v_j,mu|^2, com GM).
    Receptor i: antena a_i = W_q x_i / |.| (qkv 'self': a_i = v_i); guany de Malus amb fons isòtrop,
        G_ij^2 = rho_h + (1 - rho_h) |<a_i|v_j>|^2,   |<a|v>|^2 = <a,v>^2 + <a,Jv>^2;
        sintonia alpha_i = W_alpha x_i (desintonia Delta_ij = alpha_i - eps (i - j)).
    Autoconsistència:  sum_j K^2 G_ij^2 / (lam_i^2 + Delta_ij^2) = 1  (o lam al terra: per sota de K_c, silenci).
    Sortida (l'estat relaxat del circuit):  z_i = g sum_j w_ij v_j,  w_ij = K G_ij / (lam_i + i Delta_ij),  i -> J;  W_o.

    Escala d'ablació (cfg.qkv): full (antena W_q, K = V = W_v) | v (antena = el propi valor: una sola projecció) |
    x (cap projecció: antena i valor = la llesca del capçal del corrent residual).  Més: omni (G = 1: cap contingut,
    el control de posició pura) | rho0 (rho = 0 fix: antena pura) | self (= v) | v1 (sense W_v, amb W_q)."""

    def __init__(self, cfg: Cfg, d_model, H, max_len):
        super().__init__()
        self.cfg, self.H = cfg, H
        self.d = cfg.d_osc or d_model // H
        self.m = self.d // 2
        v = cfg.qkv
        if v in ("v1", "x"):
            assert H * self.d == d_model
            self.W_v = None
        else:
            self.W_v = nn.Linear(d_model, H * self.d, bias=False)
        self.W_q = None if v in ("self", "v", "x", "omni") else nn.Linear(d_model, H * self.d, bias=False)
        self.W_k = nn.Linear(d_model, H * self.d, bias=False) if v == "qkv" else None   # qkv: claus separades (11 oct)
        self.W_o = nn.Linear(H * self.d, d_model, bias=True)
        self.W_alpha = nn.Linear(d_model, H, bias=True)
        eps0 = torch.logspace(math.log10(cfg.eps_lo), math.log10(cfg.eps_hi), H)
        if cfg.an_age == "token":
            self.chi_pl = nn.Parameter(eps0.view(H, 1).repeat(1, self.m))
        else:
            self.log_eps = nn.Parameter(eps0.log())
        if v not in ("omni", "rho0"):
            self.raw_rho = nn.Parameter(torch.full((H,), math.log(cfg.an_rho / (1.0 - cfg.an_rho))))
        self.log_K = nn.Parameter(torch.full((H,), math.log(cfg.K_init)))
        self.log_gain = nn.Parameter(torch.zeros(H))
        assert not cfg.km_explicit or cfg.km_delta > 0 or cfg.km_noise, "KM explícit: cal amplada de línia (--delta)"
        if cfg.km_noise and cfg.km_delta <= 0:
            self.log_D = nn.Parameter(torch.full((H,), math.log(0.1 * cfg.K_init)))
        if cfg.km_pow:
            self.log_n = nn.Parameter(torch.zeros(H))
        self._alpha0 = torch.zeros(H)
        if cfg.km_init in ("phys", "head"):
            self.init_phys(cfg, H, max_len)
        self.reset_mechanism()

    def init_phys(self, cfg, H, T):
        """Inicialització física (9 oct).  Les L*H unitats (capa, capçal) reben ritmes d'envelliment geomètrics del
        ràpid (eps = 1: el token anterior, una edat per unitat de desintonia) al lent (eps = 1/T: tot el context dins
        una desintonia 1), de la primera capa a l'última; amb una sola unitat, la mitjana geomètrica 1/sqrt(T).  Cada
        unitat escolta l'edat 1.5 (entre el token anterior i el següent: cap ressonància exacta), alpha = 1.5 eps, i
        comença a K/K_c = 1.5 per a un context típic: K_c = eps / (g sqrt(S)), S = sum_{L=1}^{T-1} (L - 1.5)^-2,
        g^2 = rho + (1 - rho) 2/d el guany mitjà d'una clau aleatòria.  D (si n'hi ha) = 0.1 K."""
        U = cfg.n_layers * H
        u = torch.arange(H, dtype=torch.float32) + cfg.layer * H
        if cfg.km_init == "head":                           # 11 oct: every layer starts alike; the H heads of a layer
            if H == 1:                                      # spread from eps = 1 to 1/T (one head: 1/sqrt(T))
                eps = torch.full((H,), T ** -0.5)
            else:
                eps = torch.exp(math.log(1.0 / T) * torch.arange(H, dtype=torch.float32) / (H - 1))
        elif U == 1:
            eps = torch.full((H,), T ** -0.5)
        else:
            eps = torch.exp(math.log(1.0) + (math.log(1.0 / T) - math.log(1.0)) * u / (U - 1))
        S = sum((L - 1.5) ** -2 for L in range(1, T))
        rho = cfg.an_rho if cfg.qkv not in ("omni", "rho0") else (1.0 if cfg.qkv == "omni" else 0.0)
        g = math.sqrt(rho + (1 - rho) * 2.0 / self.d)
        K = 1.5 * eps / (g * math.sqrt(S))
        with torch.no_grad():
            if hasattr(self, "log_eps"):
                self.log_eps.copy_(eps.log())
            else:
                self.chi_pl.copy_(eps.view(H, 1).repeat(1, self.m))
            self.log_K.copy_(K.log())
            if hasattr(self, "log_D"):
                self.log_D.copy_((0.1 * K).log())
        self._alpha0 = 1.5 * eps

    def reset_mechanism(self):
        nn.init.zeros_(self.W_alpha.weight)
        with torch.no_grad():
            self.W_alpha.bias.copy_(self._alpha0)

    def _heads(self, y, B, T):
        return F.normalize(y.view(B, T, self.H, self.d).transpose(1, 2).float(), dim=-1)

    def gain2(self, a, v):
        """G^2 (B,H,T,T): rho + (1 - rho) (<a_i,v_j>^2 + <a_i,J v_j>^2)."""
        if self.cfg.qkv == "omni":
            return None
        Jv = torch.stack((-v[..., 1::2], v[..., 0::2]), -1).flatten(-2)
        g2 = (a @ v.transpose(-1, -2)).square() + (a @ Jv.transpose(-1, -2)).square()
        if self.cfg.km_pow:                                   # directivitat: lòbul |<a|v>|^(2n)
            g2 = g2.clamp(1e-12, 1.0).pow(self.log_n.exp().view(1, self.H, 1, 1))
        if self.cfg.qkv == "rho0":
            return g2
        rho = torch.sigmoid(self.raw_rho).view(1, self.H, 1, 1)
        return rho + (1.0 - rho) * g2

    def forward(self, x, key_valid=None):
        B, T, _ = x.shape
        H, cfg = self.H, self.cfg
        v = self._heads(x if self.W_v is None else self.W_v(x), B, T)
        a = v if self.W_q is None else self._heads(self.W_q(x), B, T)
        k = v if self.W_k is None else self._heads(self.W_k(x), B, T)                # qkv: l'antena llegeix la clau
        G2 = self.gain2(a, k)
        alpha = self.W_alpha(x).float().transpose(1, 2)                               # (B,H,T)
        pos = torch.arange(T, device=x.device, dtype=torch.float32)
        age = (pos.view(T, 1) - pos.view(1, T)).clamp_min(0.0)
        if cfg.an_age == "token":
            en = v[..., 0::2] ** 2 + v[..., 1::2] ** 2
            eps = (en * self.chi_pl.view(1, H, 1, self.m)).sum(-1).unsqueeze(-2)        # (B,H,1,T): ritme de l'emissor
        else:
            eps = self.log_eps.exp().view(1, H, 1, 1)
        D = alpha.unsqueeze(-1) - eps * age
        # the token's own station (age 0) follows the task's rule, as for the other layers; it stays masked when the
        # antenna is the content itself (overlap one)
        ok = allowed(T, x.device, cfg.mask_self or self.W_q is None or cfg.km_init != "head", key_valid).to(D.dtype)
        K = self.log_K.exp()
        Kv = K.view(1, H, 1, 1)
        G2m = ok if G2 is None else G2 * ok
        K2m = Kv ** 2 * G2m
        D2 = D.square()
        if cfg.km_noise or cfg.km_delta > 0:
            Dn = cfg.km_delta * K if cfg.km_delta > 0 else self.log_D.exp()
            mu = explicit_mu(K2m, D2, Dn, int(cfg.km_explicit)) if cfg.km_explicit else solve_mu_noise(K2m, D2, Dn)
            lam, locked = mu - Dn.view(1, H, 1), None
            le = mu.unsqueeze(-1)
        else:
            lam, locked = solve_lambda(K2m, D2, K, cfg.lam_floor, cfg.newton)
            le = lam.unsqueeze(-1)
        den = le ** 2 + D2
        G = G2m.clamp_min(0.0).add(1e-12).sqrt() * ok
        yr, yi = Kv * G * le / den, -Kv * G * D / den
        z = _complex_read(yr, yi, v) * self.log_gain.exp().view(1, H, 1, 1)
        out = z.transpose(1, 2).reshape(B, T, H * self.d)
        w2 = K2m / den                                                                 # |w_ij|^2
        if locked is None:
            locked = w2.sum(-1) > 0.5
        dbg = {"locked": locked.detach(), "lam": (lam / K.view(1, H, 1)).detach(),
               "KoverKc": (K2m / D2.clamp_min(1e-8)).sum(-1).sqrt().detach(),
               "wsum": w2.sum(-1).detach(),
               "entropy": entropy((w2 / w2.sum(-1, keepdim=True).clamp_min(1e-12)).detach(), ok.bool())}
        return self.W_o(out.to(x.dtype)), w2.detach(), dbg


def planck(x, tau):
    """Planck kernel B_tau(x) = 1 / (tau (e^{x/tau} - 1)): 1/x for x << tau (Lorentzian, Rayleigh-Jeans), e^{-x/tau}/tau
    for x >> tau (Boltzmann).  x >= 0; the exponent is capped for float range."""
    y = (x / tau).clamp(max=40.0)                                         # cap: e^-40 is negligible; squares stay finite
    return 1.0 / (tau * 2.0 * torch.sinh(0.5 * y) * torch.exp(0.5 * y))  # = expm1(y); MPS expm1 is exp - 1 (0 for small y)


class BE(KM):
    NEWTON_STEPS = 3
    """Bose-Einstein attention (bose_attention.tex, 11 Oct 2026): the Kuramotor with the Lorentzian kernel of its
    weights replaced by the Planck kernel, a_ij = K^2 G_ij^2 B_tau(mu_i + Delta_ij^2), the solution of a budgeted allocation
    with the Bose entropy.  mu_i >= 0 from sum_j a_ij = 1, or mu_i = 0 (condensate: silence) when the sum at mu = 0 is
    below one.  tau per head, learned (log), initial be_tau0.  tau -> infinity: the Kuramotor; tau -> 0: Boltzmann
    weights (softmax) that may sum to less than one.  Readout as the Kuramotor: modulus sqrt(a), phase lag
    arctan(Delta / lambda), lambda = sqrt(mu + D^2), D = delta K (the linewidth only regularizes the phase)."""

    def __init__(self, cfg, d_model, H, max_len):
        super().__init__(cfg, d_model, H, max_len)
        self.log_tau = nn.Parameter(torch.full((H,), math.log(cfg.be_tau0)))

    def forward(self, x, key_valid=None):
        B, T, _ = x.shape
        H, cfg = self.H, self.cfg
        v = self._heads(x if self.W_v is None else self.W_v(x), B, T)
        a = v if self.W_q is None else self._heads(self.W_q(x), B, T)
        k = v if self.W_k is None else self._heads(self.W_k(x), B, T)
        G2 = self.gain2(a, k)
        alpha = self.W_alpha(x).float().transpose(1, 2)
        pos = torch.arange(T, device=x.device, dtype=torch.float32)
        age = (pos.view(T, 1) - pos.view(1, T)).clamp_min(0.0)
        eps = self.log_eps.exp().view(1, H, 1, 1)
        D = alpha.unsqueeze(-1) - eps * age
        ok = allowed(T, x.device, cfg.mask_self or self.W_q is None or cfg.km_init != "head", key_valid)
        K = self.log_K.exp(); Kv = K.view(1, H, 1, 1); tau = self.log_tau.exp().view(1, H, 1, 1)
        K2m = torch.where(ok, Kv ** 2 * (torch.ones_like(D) if G2 is None else G2), torch.zeros_like(D))
        D2 = D.square()
        S = lambda mu: torch.where(ok, K2m * planck((mu.unsqueeze(-1) + D2).clamp_min(1e-12), tau), torch.zeros_like(D2)).sum(-1)
        def dS_(mu):                                                          # dS/dmu, with B'(x) = -1/(4 tau^2 sinh^2(x/2tau))
            u = ((mu.unsqueeze(-1) + D2).clamp_min(1e-12) / tau).clamp(max=40.0)
            return torch.where(ok, -K2m / (4.0 * tau ** 2 * torch.sinh(0.5 * u).square()), torch.zeros_like(D2)).sum(-1)
        with torch.no_grad():
            # explicit width of the effective mode (exact for one station; tau -> inf: the explicit Kuramotor),
            # mu = tau ln(1 + T (T - 1) / (tau U)), T = S(0), U = -S'(0); then NEWTON_STEPS Newton steps on 1/S - 1
            zero = torch.zeros(B, H, T, device=x.device)
            S0 = S(zero); U0 = (-dS_(zero)).clamp_min(1e-30)
            mu0 = tau.view(1, H, 1) * torch.log1p((S0 * (S0 - 1.0)).clamp_min(0.0) / (tau.view(1, H, 1) * U0))
            for _ in range(self.NEWTON_STEPS):
                Sm = S(mu0).clamp_min(1e-30)
                g = 1.0 / Sm - 1.0
                gp = (-dS_(mu0) / Sm.square()).clamp_min(1e-30)
                mu0 = torch.where(S0 > 1.0, (mu0 - g / gp).clamp_min(1e-12), zero)
        # one differentiable Newton step (implicit function theorem); condensate: mu = 0
        dS = dS_(mu0)
        mu = torch.where(S0 > 1.0, mu0 - (S(mu0) - 1.0) / dS.clamp(max=-1e-12), torch.zeros_like(mu0)).clamp_min(0.0)
        w2 = torch.where(ok, K2m * planck((mu.unsqueeze(-1) + D2).clamp_min(1e-12), tau), torch.zeros_like(D2))
        Dn = cfg.km_delta * K if cfg.km_delta > 0 else 1e-3 * K
        lam = (mu + Dn.view(1, H, 1) ** 2).sqrt().unsqueeze(-1)
        rad = (lam ** 2 + D2).sqrt()
        amp = w2.clamp_min(0.0).add(1e-24).sqrt() * ok
        yr, yi = amp * lam / rad, -amp * D / rad
        z = _complex_read(yr, yi, v) * self.log_gain.exp().view(1, H, 1, 1)
        out = z.transpose(1, 2).reshape(B, T, H * self.d)
        dbg = {"locked": (S0 > 1.0).detach(), "lam": (lam.squeeze(-1) / K.view(1, H, 1)).detach(),
               "KoverKc": S0.clamp_min(0).sqrt().detach(), "wsum": w2.sum(-1).detach(),
               "tau": self.log_tau.exp().detach(),
               "entropy": entropy((w2 / w2.sum(-1, keepdim=True).clamp_min(1e-12)).detach(), ok.bool())}
        return self.W_o(out.to(x.dtype)), w2.detach(), dbg


ATTN = {"SM": SM, "PG": PG, "TH": TH, "AD": AD, "AD0": AD, "DG": AD, "RS": RS, "GM": GM, "KM": KM, "BE": BE}


# ------------------------------------------------------------------ esquelet comú (el del protocol de la síntesi)

@dataclass
class Net:
    task: str = "lm"             # lm | cls (una posició) | seq (moltes posicions etiquetades)
    vocab: int = 8004
    n_out: int = 8004
    max_len: int = 128
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 512
    dropout: float = 0.1
    embed_scale: bool = False
    init_std: float = 0.0
    padding_idx: object = 0
    pe: str = "none"             # none | sinusoidal (codi de posició absolut a l'entrada: PG i TH tal com es publiquen)


def sinusoidal(n, d):
    pos = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


class Block(nn.Module):
    def __init__(self, net: Net, cfg: Cfg):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(net.d_model), nn.LayerNorm(net.d_model)
        self.attn = ATTN[cfg.model](cfg, net.d_model, net.n_heads, net.max_len)
        self.ffn = nn.Sequential(nn.Linear(net.d_model, net.d_ff), nn.GELU(), nn.Dropout(net.dropout),
                                 nn.Linear(net.d_ff, net.d_model), nn.Dropout(net.dropout))
        self.keep, self.last = False, None

    def forward(self, x, key_valid=None):
        out, a, dbg = self.attn(self.n1(x), key_valid)
        if self.keep:
            self.last = (a.detach(), dbg)
        x = x + out
        return x + self.ffn(self.n2(x))


class Model(nn.Module):
    def __init__(self, net: Net, cfg: Cfg):
        super().__init__()
        self.net, self.cfg = net, cfg
        self.embed = nn.Embedding(net.vocab, net.d_model, padding_idx=net.padding_idx)
        if net.pe == "sinusoidal":
            self.register_buffer("pe", sinusoidal(8192, net.d_model), persistent=False)
        self.drop = nn.Dropout(net.dropout)
        self.blocks = nn.ModuleList([Block(net, replace(cfg, layer=l, n_layers=net.n_layers)) for l in range(net.n_layers)])
        self.norm = nn.LayerNorm(net.d_model)
        self.head = nn.Linear(net.d_model, net.n_out)
        if net.init_std > 0:
            for mod in self.modules():
                if isinstance(mod, (nn.Linear, nn.Embedding)):
                    nn.init.normal_(mod.weight, std=net.init_std)
                    if getattr(mod, "bias", None) is not None:
                        nn.init.zeros_(mod.bias)
            for mod in self.modules():
                if hasattr(mod, "reset_mechanism"):
                    mod.reset_mechanism()

    def forward(self, x, key_valid=None, index=None):
        h = self.embed(x)
        if self.net.embed_scale:
            h = h * math.sqrt(self.net.d_model)
        if self.net.pe == "sinusoidal":
            h = h + self.pe[: x.size(1)]
        h = self.drop(h)
        for b in self.blocks:
            h = b(h, key_valid)
        h = self.norm(h)
        if self.net.task == "cls":
            h = h[torch.arange(h.size(0), device=h.device), index]
        return self.head(h)

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def attn_params(self):
        return sum(p.numel() for n, p in self.named_parameters() if ".attn." in n and p.requires_grad)
