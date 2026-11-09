"""Figure of Solvable model of a radio head: exact dynamics against the closed forms.
(a) order parameter, (b) total weight, against K/K_c, at m = 16 (simulation: median and quartiles over contexts and
starts); (c) error of the implicit closed form against the simulation as a function of m; (d) error of the explicit
forms against the implicit one.  Reads sim_receiver_<m>.npz; writes fig_receiver.pdf/png."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, theory as th
import os
MS = [m for m in (4, 8, 16, 32, 64) if os.path.exists(f"sim_receiver_{m}.npz")]
D = {m: np.load(f"sim_receiver_{m}.npz") for m in MS}
XS = D[MS[0]]["XS"]; XF = np.linspace(0.3, 3.2, 300)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "legend.frameon": False, "mathtext.fontset": "cm"})
C = {"sim": "#222222", "imp": "#1f5fb4", "exp": "#d1495b", "step": "#e09f3e", "noise": "#7a7a7a"}
fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.9)); A, B, Dd, Cc, Ee, Ff = ax.ravel()

def theory_curves(G, Dl, xs, kind, D_=0.0):
    s1, s2, Kc, rinf = th.sums(G, Dl); rmax = np.sqrt((G ** 2).sum()); out_r, out_w = [], []
    for x in xs:
        K = x * Kc
        if kind == "imp": r = th.implicit(G, Dl, K); mu = K * r
        elif kind == "exp": r = th.explicit(G, Dl, K); mu = K * r
        elif kind == "step": r = th.explicit(G, Dl, K, step=True); mu = K * r
        elif kind == "noise": mu = th.implicit_noise(G, Dl, K, D_ * K); r = (mu - D_ * K) / K
        a = K ** 2 * G ** 2 / (mu ** 2 + Dl ** 2)
        out_r.append(r / rmax); out_w.append(a)
    return np.array(out_r), np.array(out_w)

m0 = 16 if 16 in MS else MS[-1]; d = D[m0]; G, Dl, R, Z = d[f"G_{m0}"], d[f"D_{m0}"], d[f"r_{m0}"], d[f"zeta_{m0}"]
nctx = len(G)
rs = np.array([R[c] / np.sqrt((G[c] ** 2).sum()) for c in range(nctx)])            # (ctx, x, start)
ws = np.array([(np.abs(Z[c]) ** 2).sum(-1) for c in range(nctx)])                   # total weight (ctx, x, start)
for P, sim, lab in ((A, rs, r"$r/r_{\max}$"), (B, ws, r"total weight")):
    q1, med, q3 = np.percentile(sim.reshape(nctx, len(XS), -1).transpose(1, 0, 2).reshape(len(XS), -1), [25, 50, 75], 1)
    for kind, ls, lw in (("imp", "-", 1.3), ("exp", "--", 1.0), ("step", ":", 1.2), ("noise", "-", 0.7)):
        cur = np.array([theory_curves(G[c], Dl[c], XF, kind, 0.05)[0 if P is A else 1] for c in range(nctx)])
        y = np.median(cur if P is A else cur.sum(-1), 0)
        P.plot(XF, y, ls, color=C[kind], lw=lw, zorder=2,
               label={"imp": "implicit", "exp": r"explicit, $\bar G\sqrt{1-K_c^2/K^2}$", "step": "explicit + one step",
                      "noise": r"implicit, linewidth $\delta=0.05$"}[kind])
    P.errorbar(XS, med, yerr=[med - q1, q3 - med], fmt="o", ms=3.2, color=C["sim"], lw=0.8, capsize=1.5, zorder=3,
               label="simulation")
    P.axvline(1, color="0.6", lw=0.5, ls=":"); P.set_xlabel(r"$K/K_c$"); P.set_ylabel(lab); P.set_xlim(0.3, 3.2)
A.set_ylim(-0.03, 1.0); B.set_ylim(0, 1.12); A.legend(fontsize=6.3, loc="upper left")
B.plot(XF, np.minimum(XF ** 2, 1), color="0.75", lw=3, zorder=1, alpha=0.6, label=r"$\min(K^2/K_c^2,1)$")
B.legend(fontsize=6.3, loc="lower right", handles=[B.get_lines()[-1]])

# (d,e) error of the implicit closed form against the simulation, against m (N = 16) and against N (m = 32)
import glob
def err_of(d, m, x):
    G, Dl, Z = d[f"G_{m}"], d[f"D_{m}"], d[f"zeta_{m}"]; i = int(np.argmin(np.abs(d["XS"] - x))); errs = []
    if abs(d["XS"][i] - x) > 1e-9: return None
    for c in range(len(G)):
        a_th = theory_curves(G[c], Dl[c], [x], "imp")[1][0]
        errs += [np.abs(np.abs(Z[c, i, st]) ** 2 - a_th).sum() / a_th.sum() for st in range(Z.shape[2])]
    return np.percentile(errs, [25, 50, 75])
mfiles = {m: D[m] for m in MS}
for f in sorted(glob.glob("sim_scan_m*_N16.npz")):
    mm = int(f.split("_m")[1].split("_")[0]); mfiles[mm] = np.load(f)
nfiles = {16: D[32]} if 32 in D else {}
for f in sorted(glob.glob("sim_scan_m32_N*.npz")):
    nn = int(f.split("_N")[1].split(".")[0]); nfiles[nn] = np.load(f)
cols = {0.6: "#7fb3d5", 1.0: "#2e86c1", 1.5: "#1b4f72", 3.0: "#000000"}
for P, files, key, xl in ((Cc, mfiles, "m", r"$m$ ($N=16$)"), (Ee, nfiles, "N", r"$N$ ($m=32$)")):
    ks = sorted(files)
    for x, col in cols.items():
        pts = [(k, err_of(files[k], k if key == "m" else 32, x)) for k in ks]
        pts = [(k, e) for k, e in pts if e is not None]
        if not pts: continue
        kk = np.array([p[0] for p in pts]); e = np.array([p[1] for p in pts])
        P.errorbar(kk, e[:, 1], yerr=[e[:, 1] - e[:, 0], e[:, 2] - e[:, 1]], fmt="o" + ("--" if x < 1 else "-"), ms=3,
                   lw=0.9, capsize=1.5, color=col, label=fr"$K/K_c={x:g}$" + (" (drifting)" if x < 1 else ""))
    P.set_xscale("log", base=2); P.set_yscale("log"); tk = [k for k in ks if k in (4, 16, 64, 256, 1024)] if key == "m" else ks; P.set_xticks(tk); P.set_xticklabels(tk, fontsize=6.5); P.minorticks_off()
    P.set_xlabel(xl); P.set_ylabel("error vs simulation")
Ee.legend(fontsize=5.8, loc="lower left", ncol=2); Ee.set_ylim(2e-4, 0.6)

# (d) explicit forms against the implicit one (no simulation), all m pooled
for kind, col, lab in (("exp", C["exp"], "explicit"), ("step", C["step"], "explicit + one step")):
    E = []
    for m in MS:
        d = D[m]
        for c in range(len(d[f"G_{m}"])):
            G, Dl = d[f"G_{m}"][c], d[f"D_{m}"][c]
            a_i = theory_curves(G, Dl, XF, "imp")[1]; a_e = theory_curves(G, Dl, XF, kind)[1]
            E.append(np.abs(a_e - a_i).sum(-1) / a_i.sum(-1))
    E = np.array(E); q1, med, q3 = np.percentile(E, [25, 50, 75], 0)
    Dd.plot(XF, med, color=col, lw=1.2, label=lab); Dd.fill_between(XF, q1, q3, color=col, alpha=0.2, lw=0)
Dd.axvline(1, color="0.6", lw=0.5, ls=":"); Dd.set_yscale("log"); Dd.set_ylim(1e-5, 1); Dd.set_xlim(0.3, 3.2)
Dd.set_xlabel(r"$K/K_c$"); Dd.set_ylabel("error vs implicit"); Dd.legend(fontsize=6.3, loc="lower right")
Dd.text(0.66, 3e-4, "identical\nbelow $K_c$\n($r=0$)", fontsize=6.3, ha="center", color="0.4")

# (f) drifting branch: simulated total weight relative to K^2/K_c^2, for every m
for m, col in zip(MS, plt.cm.viridis(np.linspace(0, 0.85, len(MS)))):
    d = D[m]; G, Dl, Z = d[f"G_{m}"], d[f"D_{m}"], d[f"zeta_{m}"]; sel = XS < 1
    ratio = np.array([[(np.abs(Z[c, i]) ** 2).sum(-1) / XS[i] ** 2 for i in np.where(sel)[0]] for c in range(len(G))])
    med = np.median(ratio.transpose(1, 0, 2).reshape(sel.sum(), -1), 1)
    Ff.plot(XS[sel], med, "o-", ms=3, lw=0.9, color=col, label=f"$m={m}$")
Ff.axhline(1, color="0.6", lw=0.6, ls=":"); Ff.set_ylim(0.6, 1.05); Ff.set_xlim(0.3, 1.0)
Ff.set_xlabel(r"$K/K_c$"); Ff.set_ylabel(r"total weight / $(K/K_c)^2$, simulation"); Ff.legend(fontsize=5.8, ncol=2, loc="lower left")
for P, t in zip([A, B, Dd, Cc, Ee, Ff], "abcdef"):
    P.text(-0.30, 1.0, f"({t})", transform=P.transAxes, fontsize=9)
    P.spines["top"].set_visible(False); P.spines["right"].set_visible(False)
fig.tight_layout(); fig.savefig("fig_receiver.pdf"); fig.savefig("fig_receiver.png", dpi=200)
