"""Fig. 2 of the letter (physics only, no training): the integrated receiver against the closed forms.
Same contexts and integrator as sim_receiver.py (N tokens on an ageing chain, random unit contents in C^m, random
query, dipole gains, noiseless Lohe equation, RK4 renormalized, time averages over the second half), on a dense grid
of K/K_c around the transition.  One fixed context for every m: the gain profile G_j = g_j sqrt(4/m) (g_j drawn once, from a random context at m = 4) and
the dial alpha = 3.5 are the same in every run, so the closed forms give one curve in units of K_c and G-bar for all m;
each run draws its own contents v_j = G_j e^{i phi_j} q + sqrt(1 - G_j^2) u_j (u_j random, orthogonal to q), so only
the cross-talk between tokens, of order 1/m, changes with m.
Usage: python3 sim_fig2.py m [N]; writes sim_fig2_m<m>_N<N>.npz."""
import numpy as np, sys
m = int(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 16
rng = np.random.default_rng(1000 + m)
NCTX, NSTART, SMAX = 10, 3, 600.0
XS = np.array([0.5, 0.7, 0.85, 0.95, 1.02, 1.05, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0])

r0 = np.random.default_rng(7); V0 = r0.normal(size=(N, 4)) + 1j * r0.normal(size=(N, 4)); V0 /= np.linalg.norm(V0, axis=1, keepdims=True)
q0 = r0.normal(size=4) + 1j * r0.normal(size=4); q0 /= np.linalg.norm(q0)
G0 = np.abs(V0.conj() @ q0) * np.sqrt(4 / m); D0 = 3.5 - np.arange(1, N + 1) * 1.0

def context():
    q = rng.normal(size=m) + 1j * rng.normal(size=m); q /= np.linalg.norm(q)
    U = rng.normal(size=(N, m)) + 1j * rng.normal(size=(N, m)); U -= np.outer(U @ q.conj(), q)
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    V = (G0 * np.exp(2j * np.pi * rng.random(N)))[:, None] * q[None, :] + np.sqrt(1 - G0 ** 2)[:, None] * U
    return V, np.abs(V.conj() @ q), D0

R_all, Z_all, G_all, D_all = [], [], [], []
for c in range(NCTX):
    V, G, Dl = context()
    Kc = 1 / np.sqrt((G ** 2 / Dl ** 2).sum())
    B = len(XS) * NSTART; Kb = np.repeat((XS * Kc)[:, None], NSTART, 0)
    z = rng.normal(size=(B, m)) + 1j * rng.normal(size=(B, m)); z /= np.linalg.norm(z, axis=1, keepdims=True)
    GV = G[:, None] * V
    def f(z, s):
        h = np.exp(1j * Dl * s) @ GV
        return Kb * (h[None, :] - np.real(z @ h.conj())[:, None] * z)
    DT = min(0.01, 0.12 / np.abs(Dl).max()); steps = int(SMAX / DT); s = 0.0
    racc = np.zeros(B); zacc = np.zeros((B, N), complex); n = 0
    for k in range(steps):
        k1 = f(z, s); k2 = f(z + 0.5 * DT * k1, s + 0.5 * DT); k3 = f(z + 0.5 * DT * k2, s + 0.5 * DT); k4 = f(z + DT * k3, s + DT)
        z = z + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4); z /= np.linalg.norm(z, axis=1, keepdims=True); s += DT
        if k >= steps // 2:
            racc += np.real(z @ (np.exp(1j * Dl * s) @ GV).conj()); zacc += np.exp(-1j * Dl * s)[None, :] * (z @ V.conj().T); n += 1
    R_all.append((racc / n).reshape(len(XS), NSTART)); Z_all.append((zacc / n).reshape(len(XS), NSTART, N))
    G_all.append(G); D_all.append(Dl); print(f"m={m} context {c} done", flush=True)
    np.savez(f"sim_fig2_m{m}_N{N}.npz", XS=XS, r=np.array(R_all), zeta=np.array(Z_all), G=np.array(G_all), D=np.array(D_all))
