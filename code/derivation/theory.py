"""Closed forms of Solvable model of a radio head (noiseless and with linewidth), for one receiver.
G, Dl: gains and detunings (arrays); K: coupling.  All return the bandwidth mu = K r (+ D) and the weights a_j."""
import numpy as np
from scipy.optimize import brentq

def sums(G, Dl):
    s1, s2 = (G ** 2 / Dl ** 2).sum(), (G ** 2 / Dl ** 4).sum()
    return s1, s2, 1 / np.sqrt(s1), s1 / np.sqrt(s2)            # sigma1, sigma2, K_c, r_inf

def S(mu, G, Dl, K):
    return (K ** 2 * G ** 2 / (mu ** 2 + Dl ** 2)).sum()

def implicit(G, Dl, K):
    """Root of sum_j a_j = 1 (locked) or r = 0 (drifting).  Returns r."""
    s1 = (G ** 2 / Dl ** 2).sum()
    if K ** 2 * s1 <= 1 + 1e-9: return 0.0
    hi = K * np.sqrt((G ** 2).sum()) + 1e-12
    return brentq(lambda mu: S(mu, G, Dl, K) - 1, 1e-15, hi) / K

def explicit(G, Dl, K, step=False):
    """r = r_inf sqrt(1 - K_c^2/K^2), optionally one Newton step on ln S = 0 in ln(K r)."""
    s1, s2, Kc, rinf = sums(G, Dl)
    if K <= Kc: return 0.0
    r = rinf * np.sqrt(1 - Kc ** 2 / K ** 2)
    if step:
        mu = K * r; a = K ** 2 * G ** 2 / (mu ** 2 + Dl ** 2); Sv = a.sum()
        c = (a * mu ** 2 / (mu ** 2 + Dl ** 2)).sum() / Sv
        r = r * Sv ** (1 / (2 * c))
    return r

def weights(G, Dl, K, r, D=0.0):
    return K ** 2 * G ** 2 / ((K * r + D) ** 2 + Dl ** 2)

def implicit_noise(G, Dl, K, D):
    """Root mu of mu (1 - S(mu)) = D.  Returns mu."""
    return brentq(lambda mu: mu * (1 - S(mu, G, Dl, K)) - D, D * (1 + 1e-12), D + K * np.sqrt((G ** 2).sum()) + 1e-12)

def explicit_noise(G, Dl, K, D, step=True):
    """Largest root of (mu - D)(mu^2 + Db^2) = K^2 Gb^2 mu, then one Newton step on ln S - ln(1 - D/mu) = 0."""
    s1, s2, Kc, rinf = sums(G, Dl)
    Db2, KG2 = s1 / s2, K ** 2 * rinf ** 2
    mu = np.roots([1, -D, Db2 - KG2, -D * Db2]); mu = mu[np.abs(mu.imag) < 1e-9].real.max()
    if step:
        a = K ** 2 * G ** 2 / (mu ** 2 + Dl ** 2); Sv = a.sum(); c = (a * mu ** 2 / (mu ** 2 + Dl ** 2)).sum() / Sv
        mu = mu * np.exp((np.log(Sv) - np.log(1 - D / mu)) / (2 * c + D / (mu - D)))
    return mu
