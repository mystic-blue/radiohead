"""Fig. 3(a) of the letter (26 Oct 2026): the integrated receiver against the closed form, on real contexts of trained
language models.  As sim_fig2.py, but the gains G_j, contents v_j (in C^m) and detunings Delta_j of every context are
those of a trained radio head (TinyStories, d_model 128 split into 8, 4, 2, 1 heads: m = 8, 16, 32, 64; second layer,
lab KM seed 1), for receivers at random positions of the validation stream whose tokens the receiver resolves: the closure
averages the beats between tokens, which needs their spacing eps to exceed the band, eps >= K_c (EPSMIN).  For each receiver the coupling is scanned
over K/K_c (the trained one is one point of the scan); time in units of 1/K_c; tokens with |Delta_j| > 20 K_c dropped
(weight < 1/400 of a resonant one).  Noiseless Lohe equation, RK4 renormalized, time averages over the second half; r, the total weight sum_j |<zeta_j>|^2, and the collapsed order parameter
r_hat = (1 - F(Kr)/F(0))^(1/2), F(mu) = sum_j G_j^2/(mu^2 + Delta_j^2): the implicit root gives r_hat = (1 - K_c^2/K^2)^(1/2)
exactly, for every context (one token: Adler's law); and the time-averaged amplitude <zeta_j> of every token
(zeta, padded to 512 like Gpad, Dpad), and the amplitudes w_j demodulated by least squares against
all the tokens (wls), which separates tokens whose detunings differ by less than the inverse averaging time ( time and frequencies in units of K_c, so K = XS).
Usage (from code/): python3 derivation/sim_real.py <h> [n_receivers] [d_model]; writes derivation/sim_real_h<h>.npz
(d_model 128) or sim_real_D<d_model>h<h>.npz."""
import os, sys
for _v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):    # small matrices: threaded BLAS
    os.environ.setdefault(_v, "1")                                                # is ~100x slower under load
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); CODE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(CODE, "figures")); sys.path.insert(0, HERE)
import dump_lm_lab as D                                                          # noqa: E402
from oscfamily.data import tinystories                                           # noqa: E402
from lab.layers import allowed                                                   # noqa: E402
import theory as th                                                              # noqa: E402
torch.set_num_threads(2)
h = int(sys.argv[1]); NREC = int(sys.argv[2]) if len(sys.argv) > 2 else 10
DM = int(sys.argv[3]) if len(sys.argv) > 3 else 128; TAG = f"h{h}" if DM == 128 else f"D{DM}h{h}"
rng = np.random.default_rng(h)
XS = np.array([0.5, 0.7, 0.85, 0.95, 1.05, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0, 2.5])
NSTART, SMAX, CUT = 2, 300.0, 20.0
SMAX_CAP = float(os.environ.get("SMAX_CAP", "6000"))
EPSMIN = float(os.environ.get("EPSMIN", "1"))       # 26 Oct: only receivers whose tokens are resolved, eps >= K_c (Sec. S2)
if EPSMIN == 0:                                      # every receiver, resolved or not: sim_real_<tag>_all.npz
    TAG += "_all"; rng = np.random.default_rng(100 + h)
SEED = os.environ.get("SEED")                         # 27 Oct: further receivers, sim_real_<tag>_s<SEED>.npz
if SEED:
    TAG += f"_s{SEED}"; rng = np.random.default_rng(1000 * int(SEED) + h + (100 if EPSMIN == 0 else 0))
F1 = os.environ.get("F1") == "1"                      # 3 Nov: the full-data protocol (KM --same, all TinyStories, one epoch, seed 0)
OUTDIR = os.path.join(HERE, "sim_f1") if F1 else HERE; os.makedirs(OUTDIR, exist_ok=True)
CKPT = f"lab/results/lm/KM-same/L2-D{DM}-h{h}-f1/lr0.001-e1-w0.02-dc0.2/s0.pt" if F1 else f"lab/results/lm/KM/L2-D{DM}-h{h}-f0.1/lr0.001-e6-w0.02-dc0.2/s1.pt"
m_ = D.load_lab(os.path.join(CODE, CKPT))
att = m_.blocks[1].attn
cap = {}
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
    ok = allowed(T, x.device, False, tok != 0)[:, 0] if allowed(T, x.device, False, tok != 0).dim() == 4 else None
vc = (v[..., 0::2] + 1j * v[..., 1::2]).numpy()                                   # (B,H,T,m)
m = vc.shape[-1]
out = {k: [] for k in ("r", "rimp", "ws", "rhat", "zeta", "wls", "G", "D", "Kc", "Ktr", "N")}
rec = 0
while rec < NREC:
    b, hh, i = rng.integers(B), rng.integers(H), int(rng.integers(16, 64))
    j = np.arange(i + 1); j = j[(tok[b, : i + 1] != 0).numpy()]
    G = np.sqrt(G2[b, hh, i, j].numpy().astype(float)); Dl = alpha[b, hh, i].item() - eps[hh] * (i - j)
    Kc = th.sums(G, Dl)[2]
    if eps[hh] < EPSMIN * Kc:                          # tokens not resolved by the receiver (beats slower than the band)
        continue
    rec += 1
    keep = np.abs(Dl) < CUT * Kc; G, Dl, V = G[keep], Dl[keep] / Kc, vc[b, hh, j[keep]]          # time in units of 1/K_c
    Kc1 = th.sums(G, Dl)[2]
    Bt = len(XS) * NSTART; Kb = np.repeat((XS * Kc1)[:, None], NSTART, 0)
    z = rng.normal(size=(Bt, m)) + 1j * rng.normal(size=(Bt, m)); z /= np.linalg.norm(z, axis=1, keepdims=True)
    GV = G[:, None] * V
    f = lambda z, s: Kb * ((np.exp(1j * Dl * s) @ GV)[None, :] - np.real(z @ (np.exp(1j * Dl * s) @ GV).conj())[:, None] * z)
    DT = min(0.01, 0.12 / np.abs(Dl).max())
    TMAX = min(SMAX_CAP, max(SMAX, 2 * 10 * 2 * np.pi / max(np.abs(Dl).min(), 1e-3)))   # average >= 10 periods of the slowest token
    steps = int(TMAX / DT); s = 0.0; racc = np.zeros(Bt); zacc = np.zeros((Bt, len(G)), complex); n = 0
    for k in range(steps):
        k1 = f(z, s); k2 = f(z + 0.5 * DT * k1, s + 0.5 * DT); k3 = f(z + 0.5 * DT * k2, s + 0.5 * DT); k4 = f(z + DT * k3, s + DT)
        z = z + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4); z /= np.linalg.norm(z, axis=1, keepdims=True); s += DT
        if k >= steps // 2:
            racc += np.real(z @ (np.exp(1j * Dl * s) @ GV).conj()); zacc += np.exp(-1j * Dl * s)[None, :] * (z @ V.conj().T); n += 1
    rr = (racc / n).reshape(len(XS), NSTART); out["r"].append(rr)
    Fm = lambda mu: (G ** 2 / (mu[..., None] ** 2 + Dl ** 2)).sum(-1)               # collapsed order parameter (1 - F(Kr)/F(0))^(1/2)
    # lock-in by least squares: z = sum_l w_l v_l e^{i Delta_l s} over the averaging window, b = A w with
    # A_jl = <v_j|v_l> sum_s e^{i(Delta_l - Delta_j) s}; separates tokens whose detunings differ by less than 1/window
    k0 = steps // 2; om = Dl[None, :] - Dl[:, None]; x = np.exp(1j * om * DT)
    S = np.where(np.abs(om) * DT < 1e-12, float(n), x ** (k0 + 1) * (1 - x ** n) / np.where(np.abs(1 - x) < 1e-15, 1, 1 - x))
    A = (V.conj() @ V.T) * S
    wls = np.linalg.solve(A + 1e-9 * np.eye(len(G)) * n, zacc.T).T                                     # (Bt, N)
    out["wls"].append(np.pad(wls.reshape(len(XS), NSTART, -1), ((0, 0), (0, 0), (0, 512 - len(G)))))
    out["rhat"].append(np.sqrt(np.clip(1 - Fm((XS * Kc1)[:, None] * np.abs(rr)) / Fm(np.zeros(1)), 0, None))); out["ws"].append((np.abs(zacc / n) ** 2).sum(1).reshape(len(XS), NSTART)); out["zeta"].append(np.pad((zacc / n).reshape(len(XS), NSTART, -1), ((0, 0), (0, 0), (0, 512 - len(G))))); out["rimp"].append([th.implicit(G, Dl, x_ * Kc1) for x_ in XS])
    out["G"].append(G); out["D"].append(Dl); out["Kc"].append(Kc); out["Ktr"].append(Ktr[hh] / Kc); out["N"].append(len(G)); out.setdefault("eps", []).append(eps[hh] / Kc)
    print(f"h={h} m={m} receiver {rec - 1}: N={len(G)} trained K/K_c={Ktr[hh] / Kc:.2f} steps={steps}", flush=True)
    np.savez(os.path.join(OUTDIR, f"sim_real_{TAG}.npz"), XS=XS, m=m, r=np.array(out["r"]), ws=np.array(out["ws"]), rhat=np.array(out["rhat"]), zeta=np.array(out["zeta"]), wls=np.array(out["wls"]), rimp=np.array(out["rimp"]),
             Kc=np.array(out["Kc"]), Ktr=np.array(out["Ktr"]), eps=np.array(out["eps"]), N=np.array(out["N"]),
             rinf=np.array([th.sums(g, d)[3] for g, d in zip(out["G"], out["D"])]),
             Gpad=np.array([np.pad(g, (0, 512 - len(g))) for g in out["G"]]), Dpad=np.array([np.pad(d, (0, 512 - len(d)), constant_values=1e9) for d in out["D"]]))
