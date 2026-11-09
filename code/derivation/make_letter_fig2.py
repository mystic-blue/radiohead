"""Letter, Fig. 2: the receiver.  (a) one station, exactly: in-phase amplitude and power against |Delta|/KG, with the band
of time averages below threshold (centre to Adler's drift); (b) many stations: order parameter and total weight against
K/K_c, simulation (m = 16, N = 16) against the implicit and explicit closed forms; (c) error of the closed form against
the simulation, against m, locked and silent.  Reads sim_receiver_*.npz and sim_scan_*.npz (this folder)."""
import glob, os
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import theory as th

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "papers", "letter")
INK, MUTED, BLUE, ORANGE, GOLD = "#1f1f1e", "#8a8984", "#2a78d6", "#eb6834", "#eda100"
plt.rcParams.update({"font.size": 8, "font.family": "serif", "mathtext.fontset": "cm", "axes.linewidth": 0.6,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
                     "legend.frameon": False})
fig, (A, B, C) = plt.subplots(1, 3, figsize=(7.0, 2.05), gridspec_kw=dict(wspace=0.42, left=0.07, right=0.99,
                                                                           top=0.9, bottom=0.21))

# (a) one station
x = np.linspace(0, 3, 600); lk = x <= 1
A.plot(x[lk], np.ones(lk.sum()), color=BLUE, lw=1.3, label=r"power $|\bar\zeta|^2$")
A.plot(x[lk], np.sqrt(1 - x[lk] ** 2), color=ORANGE, lw=1.3, label=r"in phase, $r/G$")
A.plot(x[~lk], np.zeros((~lk).sum()), color=ORANGE, lw=1.3)
xd = x[~lk]; centre, adler = 1 / xd ** 2, (xd - np.sqrt(xd ** 2 - 1)) ** 2
A.fill_between(xd, adler, centre, color=BLUE, alpha=0.15, lw=0)
A.plot(xd, centre, color=BLUE, lw=1.3); A.plot(xd, adler, "--", color=BLUE, lw=0.9)
A.text(2.05, 0.29, "centre", fontsize=6.3, color=BLUE); A.text(2.05, 0.115, "Adler drift", fontsize=6.3, color=BLUE)
A.axvline(1, color=MUTED, lw=0.5, ls=":")
A.set_xlabel(r"$|\Delta|/KG$"); A.set_ylim(-0.03, 1.08); A.set_xlim(0, 3); A.legend(fontsize=6.2, loc="center right",
                                                                                       bbox_to_anchor=(1.0, 0.78))

# (b) many stations, m = 16
d = np.load("sim_receiver_16.npz"); XS = d["XS"]; G, Dl, R, Z = d["G_16"], d["D_16"], d["r_16"], d["zeta_16"]
XF = np.linspace(0.3, 3.2, 240); nc = len(G)
rs = np.array([R[c] / np.sqrt((G[c] ** 2).sum()) for c in range(nc)])
ws = np.array([(np.abs(Z[c]) ** 2).sum(-1) for c in range(nc)])
def curves(kind):
    rr, ww = [], []
    for c in range(nc):
        s1, s2, Kc, _ = th.sums(G[c], Dl[c]); rmax = np.sqrt((G[c] ** 2).sum()); r1, w1 = [], []
        for xx in XF:
            K = xx * Kc
            r = th.implicit(G[c], Dl[c], K) if kind == "imp" else th.explicit(G[c], Dl[c], K)
            r1.append(r / rmax); w1.append((K ** 2 * G[c] ** 2 / ((K * r) ** 2 + Dl[c] ** 2)).sum())
        rr.append(r1); ww.append(w1)
    return np.median(rr, 0), np.median(ww, 0)
ri, wi = curves("imp"); re_, _ = curves("exp")
B.plot(XF, wi, color=BLUE, lw=1.3, label=r"total weight $\sum_j a_j$")
B.plot(XF, ri, color=ORANGE, lw=1.3, label=r"order parameter $r$")
B.plot(XF, re_, "--", color=ORANGE, lw=0.9, label=r"$\bar G\sqrt{1-K_c^2/K^2}$")
for sim, col in ((ws, BLUE), (rs, ORANGE)):
    q1, med, q3 = np.percentile(sim.transpose(1, 0, 2).reshape(len(XS), -1), [25, 50, 75], 1)
    B.errorbar(XS, med, yerr=[med - q1, q3 - med], fmt="o", ms=2.6, mfc="white", mec=col, ecolor=col, lw=0.7,
               capsize=1.2, zorder=3)
B.axvline(1, color=MUTED, lw=0.5, ls=":"); B.set_xlim(0.3, 3.2); B.set_ylim(-0.03, 1.16)
B.set_xlabel(r"$K/K_c$"); B.legend(fontsize=6.0, loc="center right", bbox_to_anchor=(1.0, 0.64))
B.text(0.4, 1.06, "silent", fontsize=6.5, color=MUTED); B.text(1.9, 1.06, "locked", fontsize=6.5, color=MUTED)

# (c) error against m
files = {m: np.load(f"sim_receiver_{m}.npz") for m in (4, 8, 16, 32, 64) if os.path.exists(f"sim_receiver_{m}.npz")}
for f in glob.glob("sim_scan_m*_N16.npz"):
    files[int(f.split("_m")[1].split("_")[0])] = np.load(f)
def err(d, m, xx):
    i = int(np.argmin(np.abs(d["XS"] - xx)))
    if abs(d["XS"][i] - xx) > 1e-9:
        return None
    G, Dl, Z = d[f"G_{m}"], d[f"D_{m}"], d[f"zeta_{m}"]; e = []
    for c in range(len(G)):
        Kc = th.sums(G[c], Dl[c])[2]; K = xx * Kc; r = th.implicit(G[c], Dl[c], K)
        a = K ** 2 * G[c] ** 2 / ((K * r) ** 2 + Dl[c] ** 2)
        e += [np.abs(np.abs(Z[c, i, s]) ** 2 - a).sum() / a.sum() for s in range(Z.shape[2])]
    return np.median(e)
for xx, col, lab in ((3.0, INK, r"locked, $K=3K_c$"), (1.5, BLUE, r"locked, $K=1.5K_c$"), (0.6, ORANGE, r"silent, $K=0.6K_c$")):
    pts = sorted((m, err(files[m], m, xx)) for m in files); pts = [(m, e) for m, e in pts if e is not None]
    mm, ee = np.array(pts).T
    C.plot(mm, ee, "o-" if xx > 1 else "o--", ms=2.6, lw=0.9, color=col, label=lab)
mm = np.array([4, 1024]); C.plot(mm, 0.5 / mm, ":", color=MUTED, lw=0.8, label=r"$\propto 1/m$")
C.set_xscale("log", base=2); C.set_yscale("log"); C.set_xticks([4, 16, 64, 256, 1024]); C.set_xticklabels([4, 16, 64, 256, 1024])
C.minorticks_off(); C.set_xlabel(r"dimension $m=d/2$"); C.set_ylabel("error of the weights", labelpad=1)
C.legend(fontsize=5.8, loc="lower left"); C.set_ylim(2e-4, 1)

for P, t in zip((A, B, C), "abc"):
    P.text(-0.22, 1.04, t, transform=P.transAxes, fontsize=9, fontweight="bold", color=INK)
    P.spines["top"].set_visible(False); P.spines["right"].set_visible(False)
    P.grid(color="#e6e5e0", lw=0.5); P.set_axisbelow(True)
fig.savefig(os.path.join(OUT, "fig2_receiver.pdf")); fig.savefig(os.path.join(OUT, "fig2_receiver.png"), dpi=220)
