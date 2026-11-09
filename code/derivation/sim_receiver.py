"""Receiver of Solvable model of a radio head: exact dynamics against the closed forms.
Contexts: N tokens on an ageing chain (Delta_j = alpha - eps*L, L = 1..N, eps = 1, alpha non-integer), random unit
contents in C^m, random unit query, dipole gains G_j = |<q|v_j>| (n = 1).  For each context and coupling K = x K_c, the
noiseless equation dz/ds = K(h - <h,z> z) is integrated (RK4, renormalized) from several random starts; time averages
over the second half give the order parameter r and the demodulated amplitudes zeta_j.  Writes sim_receiver.npz."""
import numpy as np, sys
rng = np.random.default_rng(2026)
SCAN = len(sys.argv) > 2                      # sim_receiver.py m N: scan run (fewer couplings and starts)
N, NCTX, NSTART = (int(sys.argv[2]), 6, 4) if SCAN else (16, 8, 6)
XS = np.array([0.6, 1.0, 1.5, 3.0]) if SCAN else np.array([0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0])
MS = [int(sys.argv[1])] if sys.argv[1:] else [4, 8, 16, 32, 64]
SMAX = 500.0

def context(m):
    V = rng.normal(size=(N, m)) + 1j * rng.normal(size=(N, m)); V /= np.linalg.norm(V, axis=1, keepdims=True)
    q = rng.normal(size=m) + 1j * rng.normal(size=m); q /= np.linalg.norm(q)
    G = np.abs(V.conj() @ q)
    alpha = rng.uniform(2.2, 5.8); Dl = alpha - np.arange(1, N + 1) * 1.0
    return V, G, Dl

out = {}
for m in MS:
    R_all, Z_all, G_all, D_all = [], [], [], []
    for c in range(NCTX):
        V, G, Dl = context(m)
        Kc = 1 / np.sqrt((G ** 2 / Dl ** 2).sum())
        K = (XS * Kc)[:, None]                                            # (nx,1)
        B = len(XS) * NSTART
        Kb = np.repeat(K, NSTART, 0)                                       # (B,1)
        z = rng.normal(size=(B, m)) + 1j * rng.normal(size=(B, m)); z /= np.linalg.norm(z, axis=1, keepdims=True)
        GV = G[:, None] * V                                                # (N,m)
        def f(z, s):
            h = np.exp(1j * Dl * s) @ GV                                   # (m,)
            r = np.real(z @ h.conj())[:, None]                             # <h,z>, (B,1)
            return Kb * (h[None, :] - r * z), r
        DT = min(0.01, 0.12 / np.abs(Dl).max())                           # RK4 step resolves the fastest beat
        steps = int(SMAX / DT); s = 0.0; racc = np.zeros(B); zacc = np.zeros((B, N), complex); n = 0
        for k in range(steps):
            k1, r = f(z, s); k2, _ = f(z + 0.5 * DT * k1, s + 0.5 * DT); k3, _ = f(z + 0.5 * DT * k2, s + 0.5 * DT)
            k4, _ = f(z + DT * k3, s + DT)
            z = z + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4); z /= np.linalg.norm(z, axis=1, keepdims=True); s += DT
            if k >= steps // 2:
                r = np.real(z @ (np.exp(1j * Dl * s) @ GV).conj())
                racc += r; zacc += np.exp(-1j * Dl * s)[None, :] * (z @ V.conj().T); n += 1
        R_all.append((racc / n).reshape(len(XS), NSTART)); Z_all.append((zacc / n).reshape(len(XS), NSTART, N))
        G_all.append(G); D_all.append(Dl)
        print(f"m={m} context {c} done", flush=True)
    out[f"r_{m}"], out[f"zeta_{m}"] = np.array(R_all), np.array(Z_all)
    out[f"G_{m}"], out[f"D_{m}"] = np.array(G_all), np.array(D_all)
    np.savez(f"sim_scan_m{m}_N{N}.npz" if SCAN else f"sim_receiver{'_' + '_'.join(map(str, MS)) if sys.argv[1:] else ''}.npz", XS=XS, **out)
