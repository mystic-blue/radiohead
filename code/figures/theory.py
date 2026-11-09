"""Closed form of the antenna model and shared plotting style.

    weights   a_j = K^2 G_j^2 / (lam^2 + D_j^2),   sum_j a_j = 1  (or lam = 0 below K_c)
    K_c^{-2}  = sum_j G_j^2 / D_j^2
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID = "#1f1f1e", "#8a8984", "#e6e5e0"
BLUE, ORANGE, GREEN, GOLD, PINK = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.labelcolor": INK, "font.family": "serif", "mathtext.fontset": "cm",
                     "axes.titlesize": 8.5, "legend.frameon": False, "legend.fontsize": 7.5})


def style(a, tag=None, grid=True):
    if grid:
        a.grid(color=GRID, lw=0.5); a.set_axisbelow(True)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    if tag:
        a.text(-0.2, 1.04, tag, transform=a.transAxes, fontsize=9, fontweight="bold", color=INK)


def solve_lambda(G2, D, K):
    """G2, D: (..., N).  Returns lam (...,), 0 below threshold."""
    G2, D = np.broadcast_arrays(np.asarray(G2, float), np.asarray(D, float))
    K = np.broadcast_to(np.asarray(K, float), G2.shape[:-1])
    f = lambda l: (K[..., None] ** 2 * G2 / (l[..., None] ** 2 + D ** 2 + 1e-300)).sum(-1) - 1
    on = f(np.zeros(G2.shape[:-1])) > 0
    lo, hi = np.zeros(G2.shape[:-1]), K * np.sqrt(G2.sum(-1)) + 1e-9
    for _ in range(80):
        m = (lo + hi) / 2
        up = f(m) > 0
        lo, hi = np.where(up, m, lo), np.where(up, hi, m)
    return np.where(on, (lo + hi) / 2, 0.0)


def weights(G2, D, K):
    lam = solve_lambda(G2, D, K)
    a = np.asarray(K)[..., None] ** 2 * G2 / (lam[..., None] ** 2 + D ** 2 + 1e-300)
    return lam, a


def Kc(G2, D):
    return 1 / np.sqrt((G2 / (D ** 2 + 1e-300)).sum(-1))


def malus(rng, m, shape):
    """|<a|v>|^2 for random unit a, v in C^m: the antenna gain of a random key (mean 1/m = 2/d)."""
    v = rng.normal(size=shape + (m,)) + 1j * rng.normal(size=shape + (m,))
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    return np.abs(v[..., 0]) ** 2          # antenna along the first complex axis
