"""Checks of the lab layers.  python3 tests.py  (CPU, float32 as in training: KM computes in float32).

 1. KM and PG equal bench/models.py (the code of every run so far; PG there is checked against the official code).
 2. SM equals bench SM; the fast path (SDPA) equals the explicit one, for SM and GATED.
 3. GATED with a zero gate projection is SM times 1/2 (the gate is the only change).
 4. SIGMOID: ALiBi slopes as published; weights in (0, 1), not normalized; b = -log(max_len).
 5. Causality, for every layer and both KM solvers: changing token t never changes the output before t.
 6. KM explicit (2 steps) agrees with the implicit solver; KM-norm has sum |w|^2 = 1 and equals KM when locked;
    KM-noage does not see positions (a permutation of the past leaves the last output unchanged).
 7. Finite gradients for every layer.
"""
from __future__ import annotations

import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from layers import Cfg, LAYERS, alibi_slopes, explicit_mu, solve_mu        # noqa: E402
from net import Net, Model                                                  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(HERE), "bench"))
import models as BN                                                         # noqa: E402

torch.manual_seed(0)
FAIL = []


def check(name, ok, info=""):
    print(("ok   " if ok else "FAIL ") + name + (f"  ({info})" if info else ""))
    if not ok:
        FAIL.append(name)


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


B, T, d, H = 3, 24, 32, 4
x = torch.randn(B, T, d)

# 1. KM and PG against bench
for mask_self in (False, True):
    for expl in (0, 2):
        bc = BN.Cfg(model="KM", qkv="full", d_osc=d // H, mask_self=mask_self, max_len=T, km_init="head", km_pow=True,
                    an_rho=1e-8, km_delta=1e-3, km_explicit=expl)
        old = BN.KM(bc, d, H, T)
        new = LAYERS["KM"](Cfg(model="KM", mask_self=mask_self, max_len=T, km_explicit=expl), d, H)
        with torch.no_grad():                                  # random trained-like parameters, copied by name
            for n_, p in old.named_parameters():
                p.copy_(p + 0.3 * torch.randn_like(p))
        missing = new.load_state_dict(old.state_dict(), strict=True)
        o1, w1, _ = old(x)
        o2, _, _ = new(x)
        check(f"KM == bench KM (mask_self {mask_self}, explicit {expl})", rel(o2, o1) < 1e-5, f"rel {rel(o2, o1):.1e}")
bc = BN.Cfg(model="PG", qkv="full", d_osc=32, max_len=T)
old = BN.PG(bc, d, H, T)
new = LAYERS["PG"](Cfg(model="PG", max_len=T), d, H)
new.load_state_dict(old.state_dict(), strict=True)
check("PG == bench PG", rel(new(x)[0], old(x)[0]) < 1e-5)

# 2. SM against bench, fast path against explicit
old = BN.SM(BN.Cfg(model="SM", qkv="full"), d, H, T)
new = LAYERS["SM"](Cfg(model="SM", max_len=T), d, H)
sd = {k.replace("p.", ""): v for k, v in old.state_dict().items()}
new.load_state_dict(sd, strict=True)
new.need_weights = True
check("SM == bench SM (explicit path)", rel(new(x)[0], old(x)[0]) < 1e-5)
new.need_weights = False
check("SM == bench SM (SDPA path)", rel(new(x)[0], old(x)[0]) < 1e-5)
g = LAYERS["GATED"](Cfg(model="GATED", max_len=T), d, H)
fast = g(x)[0]
g.need_weights = True
check("GATED: SDPA path == explicit path", rel(fast, g(x)[0]) < 1e-5)

# 3. GATED with a zero gate is SM / 2
with torch.no_grad():
    g.W_q.weight.view(H, 2, d // H, d)[:, 0].copy_(new.W_q.weight.view(H, d // H, d))
    g.W_q.weight.view(H, 2, d // H, d)[:, 1].zero_()
    for nm in ("W_k", "W_v"):
        getattr(g, nm).weight.copy_(getattr(new, nm).weight)
    g.W_o.weight.copy_(new.W_o.weight); g.W_o.bias.zero_(); new.W_o.bias.zero_()
check("GATED(zero gate) == SM / 2", rel(g(x)[0], 0.5 * new(x)[0]) < 1e-5)

# 4. SIGMOID
s4 = alibi_slopes(4)
check("ALiBi slopes, 4 heads = 2^-2, 2^-4, 2^-6, 2^-8", torch.allclose(s4, torch.tensor([2.0 ** -2, 2.0 ** -4, 2.0 ** -6, 2.0 ** -8])))
check("ALiBi slopes, 8 heads = 2^-1 ... 2^-8", torch.allclose(alibi_slopes(8), 2.0 ** -torch.arange(1, 9).float()))
sg = LAYERS["SIGMOID"](Cfg(model="SIGMOID", max_len=T), d, H)
sg.need_weights = True
_, a, _ = sg(x)
ok = torch.tril(torch.ones(T, T, dtype=torch.bool))
check("SIGMOID: weights in (0,1) on visible keys, 0 elsewhere", bool((a[..., ok] > 0).all() and (a[..., ok] < 1).all() and (a[..., ~ok] == 0).all()))
check("SIGMOID: b = -log(max_len)", abs(sg.b + math.log(T)) < 1e-5)

# 5. causality (whole model, every layer)
for model in ("SM", "GATED", "SIGMOID", "PG", "KM", "KMe", "KMnorm", "KMnoage", "KMnoant"):
    cfg = Cfg(model="KM" if model.startswith("KM") else model, max_len=T, km_explicit=2 if model == "KMe" else 0,
              km_norm=model == "KMnorm", km_noage=model == "KMnoage", km_noant=model == "KMnoant")
    net = Net(vocab=50, n_out=50, d_model=d, n_heads=H, n_layers=2, d_ff=2 * d, dropout=0.0,
              pe="sinusoidal" if model == "PG" else "none")
    m = Model(net, cfg).eval()
    tok = torch.randint(1, 50, (2, T))
    t = 15
    tok2 = tok.clone(); tok2[:, t] = (tok[:, t] % 49) + 1
    with torch.no_grad():
        y1, y2 = m(tok), m(tok2)
    leak = float((y1[:, :t] - y2[:, :t]).abs().max())
    check(f"causal: {model}", leak < 1e-5, f"max change before t {leak:.1e}")

# 6. KM solvers and controls
K2m = torch.rand(2, H, T, T) * torch.tril(torch.ones(T, T)) * 0.05
D2 = (torch.randn(2, H, T, T) * 0.5).square() + 1e-3
Dn = torch.full((H,), 1e-3 * 0.3)
mi, me = solve_mu(K2m, D2, Dn), explicit_mu(K2m, D2, Dn, 2)
r = ((me - mi).abs() / mi).flatten()
check("KM explicit(2) vs implicit width", float(r.median()) < 1e-3, f"median rel {float(r.median()):.1e}, max {float(r.max()):.1e}")
kn = LAYERS["KM"](Cfg(model="KM", max_len=T, km_norm=True), d, H)
k0 = LAYERS["KM"](Cfg(model="KM", max_len=T), d, H)
k0.load_state_dict(kn.state_dict())
kn.need_weights = k0.need_weights = True
_, wn, dn = kn(x)
check("KM-norm: sum |w|^2 = 1", float((wn.sum(-1)[:, :, 1:] - 1).abs().max()) < 1e-5)
with torch.no_grad():
    k0.log_K += 3.0; kn.log_K += 3.0                      # far above K_c: KM is already normalized
o0, w0, d0 = k0(x)
on, _, _ = kn(x)
bound = float(d0["wsum"].min().rsqrt() - 1)              # locked: sum |w|^2 = 1 - D/mu, so the rescaling is <= this
dz = float((on - o0).abs().max()) / float((o0 - k0.W_o.bias).abs().max())
check("KM-norm == KM up to the line width where KM is locked", float(d0["wsum"].min()) > 0.99 and dz <= 1.5 * bound,
      f"min sum|w|^2 {float(d0['wsum'].min()):.4f}, change {dz:.1e} <= {bound:.1e}")
kg = LAYERS["KM"](Cfg(model="KM", max_len=T, km_noage=True), d, H)
perm = torch.cat([torch.randperm(T - 1), torch.tensor([T - 1])])
with torch.no_grad():
    kg.W_alpha.weight.normal_(0, 0.1)
    y1, y2 = kg(x)[0][:, -1], kg(x[:, perm])[0][:, -1]
check("KM-noage: blind to the order of the past", rel(y2, y1) < 1e-5, f"rel {rel(y2, y1):.1e}")
k_age = LAYERS["KM"](Cfg(model="KM", max_len=T), d, H)
with torch.no_grad():
    y1, y2 = k_age(x)[0][:, -1], k_age(x[:, perm])[0][:, -1]
check("KM (with ageing): sees the order", rel(y2, y1) > 1e-3, f"rel {rel(y2, y1):.1e}")

ka = LAYERS["KM"](Cfg(model="KM", max_len=T, km_noant=True), d, H)
ka.need_weights = True
_, wa, _ = ka(x)
_, wb, _ = ka(torch.randn(B, T, d))
check("KM-noant: weights independent of content", rel(wb, wa) < 1e-5, f"rel {rel(wb, wa):.1e}")

# 7. gradients
for model in ("SM", "GATED", "SIGMOID", "PG", "KM", "KMe", "KMnorm", "KMnoage", "KMnoant"):
    cfg = Cfg(model="KM" if model.startswith("KM") else model, max_len=T, km_explicit=2 if model == "KMe" else 0,
              km_norm=model == "KMnorm", km_noage=model == "KMnoage", km_noant=model == "KMnoant")
    lay = LAYERS[cfg.model](cfg, d, H)
    out = lay(torch.randn(B, T, d))[0]
    out.square().mean().backward()
    bad = [n_ for n_, p in lay.named_parameters() if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
    check(f"finite gradients: {model}", not bad, ", ".join(bad))

print("\nTOT BÉ" if not FAIL else f"\n{len(FAIL)} FALLADES: " + "; ".join(FAIL))
