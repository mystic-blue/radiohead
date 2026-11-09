"""27 Oct 2026: does the output of an unresolved receiver follow the closed form?  Receivers of the second layer of the
language model of Fig. 3(d) (KM-same D256 h4 f0.1 seed 0, m = 32), every head (slowest first), at random positions of the
validation stream and at the TRAINED coupling, with no resolution filter (most have eps < K_c, the slowest Kr/eps ~ 70).
Noiseless Lohe equation (as sim_real.py, time in units of 1/K_c), weights demodulated by least squares over the second
half (wls), and compared with the closed form w_j = K G_j / (K r + i Delta_j):
  err_w = |w_dyn - w_cf| / |w_cf|   (all weights),   err_o = |o_dyn - o_cf| / |o_cf|,  o = sum_j w_j v_j  (the head output),
  r_dyn against r_cf.
Usage (from code/): python3 derivation/sim_slow.py <head rank, 0 = slowest> [n_receivers]; writes derivation/sim_slow_k<rank>.npz"""
import os, sys
for _v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):    # small matrices: threaded BLAS
    os.environ.setdefault(_v, "1")                                                # is ~100x slower under load
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); CODE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(CODE, "figures")); sys.path.insert(0, HERE)
import dump_lm_lab as D                                                          # noqa: E402
from oscfamily.data import tinystories                                           # noqa: E402
import theory as th                                                              # noqa: E402
torch.set_num_threads(2)
RANK = int(sys.argv[1]); NREC = int(sys.argv[2]) if len(sys.argv) > 2 else 6
rng = np.random.default_rng(7 + RANK)
NSTART, CUT, SMAX_CAP = 2, 20.0, float(os.environ.get("SMAX_CAP", "4000"))
m_ = D.load_lab(os.path.join(CODE, "lab/results/lm/KM-same/L2-D256-h4-f0.1/lr0.001-e6-w0.02-dc0.2/s0.pt"))
att = m_.blocks[1].attn; cap = {}
att.register_forward_pre_hook(lambda mod, args: cap.__setitem__("x", args[0].detach()))
val = tinystories._packed()["val"].long()
with torch.no_grad():
    tok = val[:8, :-1]; m_(tok, key_valid=tok != 0)
    x = cap["x"]; B, T, _ = x.shape; H = att.H
    unit = lambda y: torch.nn.functional.normalize(att.split(y).float(), dim=-1)
    v = unit(att.W_v(x)); a = unit(att.W_q(x))
    Jv = torch.stack((-v[..., 1::2], v[..., 0::2]), -1).flatten(-2)
    ov2 = (a @ v.transpose(-1, -2)).square() + (a @ Jv.transpose(-1, -2)).square()
    rho = torch.sigmoid(att.raw_rho).view(1, H, 1, 1)
    G2 = rho + (1 - rho) * ov2.clamp(1e-12, 1).pow(att.log_n.exp().view(1, H, 1, 1))
    alpha = att.W_alpha(x).float().transpose(1, 2)
    eps = att.eps().numpy(); Ktr = att.log_K.exp().numpy()
hh = int(np.argsort(eps)[RANK])
vc = (v[..., 0::2] + 1j * v[..., 1::2]).numpy(); m = vc.shape[-1]
out = {k: [] for k in ("err_w", "err_o", "r_dyn", "r_cf", "Kc", "K", "eps", "N", "Kr_eps")}
rec = 0
while rec < NREC:
    b, i = rng.integers(B), int(rng.integers(64, T))
    j = np.arange(i + 1); j = j[(tok[b, : i + 1] != 0).numpy()]
    G = np.sqrt(G2[b, hh, i, j].numpy().astype(float)); Dl = alpha[b, hh, i].item() - eps[hh] * (i - j)
    Kc = th.sums(G, Dl)[2]
    if Ktr[hh] <= 1.05 * Kc:                                     # silent or at the edge: the comparison is for locked receivers
        continue
    keep = np.abs(Dl) < CUT * Kc; G, Dl, V = G[keep], Dl[keep] / Kc, vc[b, hh, j[keep]]
    K1 = Ktr[hh] / Kc * th.sums(G, Dl)[2]                       # trained coupling in units of K_c of the kept tokens
    rcf = th.implicit(G, Dl, K1); wcf = K1 * G / (K1 * rcf + 1j * Dl)
    z = rng.normal(size=(NSTART, m)) + 1j * rng.normal(size=(NSTART, m)); z /= np.linalg.norm(z, axis=1, keepdims=True)
    GV = G[:, None] * V
    f = lambda z, s: K1 * ((np.exp(1j * Dl * s) @ GV)[None, :] - np.real(z @ (np.exp(1j * Dl * s) @ GV).conj())[:, None] * z)
    DT = min(0.01, 0.12 / max(np.abs(Dl).max(), K1))
    sp = np.diff(np.sort(Dl)); sp = sp[sp > 0].min() if (sp > 0).any() else 1.0
    TMAX = min(SMAX_CAP, max(300.0, 2 * 10 * 2 * np.pi / sp))      # the window resolves the closest pair of tokens
    steps = int(TMAX / DT); s = 0.0; racc = np.zeros(NSTART); zacc = np.zeros((NSTART, len(G)), complex); n = 0
    for k in range(steps):
        k1 = f(z, s); k2 = f(z + 0.5 * DT * k1, s + 0.5 * DT); k3 = f(z + 0.5 * DT * k2, s + 0.5 * DT); k4 = f(z + DT * k3, s + DT)
        z = z + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4); z /= np.linalg.norm(z, axis=1, keepdims=True); s += DT
        if k >= steps // 2:
            racc += np.real(z @ (np.exp(1j * Dl * s) @ GV).conj()); zacc += np.exp(-1j * Dl * s)[None, :] * (z @ V.conj().T); n += 1
    k0 = steps // 2; om = Dl[None, :] - Dl[:, None]; xx = np.exp(1j * om * DT)
    S = np.where(np.abs(om) * DT < 1e-12, float(n), xx ** (k0 + 1) * (1 - xx ** n) / np.where(np.abs(1 - xx) < 1e-15, 1, 1 - xx))
    A = (V.conj() @ V.T) * S
    wd = np.linalg.solve(A + 1e-9 * np.eye(len(G)) * n, zacc.T).T                  # (NSTART, N), as sim_real.py (wls)
    ocf = wcf @ V; od = wd @ V
    out["err_w"].append(np.linalg.norm(wd - wcf, axis=1) / np.linalg.norm(wcf))
    out["err_o"].append(np.linalg.norm(od - ocf, axis=1) / np.linalg.norm(ocf))
    out["r_dyn"].append(racc / n); out["r_cf"].append(rcf); out["Kc"].append(Kc); out["K"].append(K1)
    out["eps"].append(eps[hh] / Kc); out["N"].append(len(G)); out["Kr_eps"].append(K1 * rcf / (eps[hh] / Kc))
    rec += 1
    print(f"head eps={eps[hh]:.3g} rec {rec - 1}: N={len(G)} K/Kc={K1:.2f} Kr/eps={out['Kr_eps'][-1]:.1f} steps={steps} "
          f"err_w={out['err_w'][-1].round(3)} err_o={out['err_o'][-1].round(3)} r {out['r_dyn'][-1].round(3)} vs {rcf:.3f}", flush=True)
    np.savez(os.path.join(HERE, f"sim_slow_k{RANK}.npz"), m=m, eps_head=eps[hh], **{k: np.array(x) for k, x in out.items()})
