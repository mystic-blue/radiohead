"""Fig. 2 of the letter (26 Oct 2026; merges the former Figs. 2 and 3): the receiver against the closed form, on the contexts
of trained language models (main panels) and on random contexts (insets), and the performance of the trained layer, all
against the dimension m of a head (d = 2m).
(a) order parameter r_hat = (1 - F(Kr)/F(0))^(1/2), F(mu) = sum_j G_j^2/(mu^2 + Delta_j^2), against K/K_c: the root of
    Eq. (norm) gives sqrt(1 - K_c^2/K^2) for every context.  Main: real contexts (../derivation/sim_real.py, TinyStories,
    d_model 128 split into 8 to 1 heads, second layer, seed 1; m = 128: one head in d_model 256).  Inset: random contexts
    (../derivation/sim_fig2.py).
(b) every weight of the locked receivers (K >= 1.2 K_c) against its detuning in units of the simulated width Kr, on the
    Lorentzian 1/(1+x^2).  Main real, inset random.
(c) error of the closed-form weights, sum_j |a_sim - a|/sum_j a, against m: locked (K = 2 K_c, root of Eq. norm) and
    silent (K = 0.7 K_c, a_j = K^2 G_j^2/Delta_j^2).  Main real (median over receivers and starts), inset random.
(d, e) test next-token accuracy relative to softmax with rotary position, against m: (d) d_model 128 split into 8 to 1
    heads; (e) four heads, d_model = 8m; both on all of TinyStories, one epoch, same initial ageing (2 Nov;
    lab/plans/lm_E.txt, lm_F.txt).  PG, the radio head (root) and one effective token (one Newton step).
Colour = m, the same scale in (a)-(c); insets on their own range of m.  Writes fig2_combined.{pdf,png}."""
import glob, importlib.util, json, os, sys
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from theory import BLUE, GREEN, INK, MUTED, plt
from theory import style as _style
DER = os.path.join(os.path.dirname(HERE), "derivation")
spec = importlib.util.spec_from_file_location("dtheory", os.path.join(DER, "theory.py")); th = importlib.util.module_from_spec(spec); spec.loader.exec_module(th)
plt.rcParams.update({"font.size": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.labelsize": 8})
LIGHT, DARK = "#bcd6f4", "#123f78"
CMAP = LinearSegmentedColormap.from_list("m", [LIGHT, DARK])
KUR = lambda x: np.sqrt(np.clip(1 - np.asarray(x, float) ** -2.0, 0, None))


def style(a, tag=None):
    _style(a, tag, grid=False)


def mini(a):
    a.tick_params(labelsize=5, length=1.5, pad=1); a.patch.set_alpha(0)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)


def rhat(G, Dl, K, r):
    F = lambda mu: (G ** 2 / (mu ** 2 + Dl ** 2)).sum()
    return np.sqrt(max(0.0, 1 - F(K * abs(r)) / F(0.0)))


# ---------------------------------------------------------------- data
REAL, ALL = {}, {}                  # resolved receivers (eps >= K_c), and receivers drawn without condition
SIMDIR = os.path.join(DER, "sim_f1") if os.environ.get("F1") == "1" else DER     # 3 Nov: F1=1 reads the full-data models
OUT = "fig2_combined_f1" if os.environ.get("F1") == "1" else "fig2_combined"
for f in sorted(glob.glob(os.path.join(SIMDIR, "sim_real_*.npz"))):
    try:
        d = dict(np.load(f))                                   # read at once: the simulations rewrite these files
    except Exception:
        continue
    if "eps" not in d: continue
    S = ALL if "_all" in os.path.basename(f) else REAL; m_ = int(d["m"])     # 27 Oct: several files (seeds) per m, merged
    if m_ not in S:
        S[m_] = d
    else:
        for k in ("r", "ws", "rhat", "zeta", "wls", "rimp", "Kc", "Ktr", "eps", "N", "rinf", "Gpad", "Dpad"):
            S[m_][k] = np.concatenate([S[m_][k], d[k]])
RAND = {int(f.split("_m")[1].split("_")[0]): dict(np.load(f)) for f in glob.glob(os.path.join(DER, "sim_fig2_m*_N16.npz"))}
MR, MQ = sorted(REAL), sorted(RAND)
LM = (3, 7)                                                     # colour scale: log2 m from 8 to 128, shared by (a)-(c)
col = lambda m: CMAP(np.clip((np.log2(m) - LM[0]) / (LM[1] - LM[0]), 0, 1))
colq = lambda m: CMAP((np.log2(m) - np.log2(MQ[0])) / (np.log2(MQ[-1]) - np.log2(MQ[0])))
rng = np.random.default_rng(0)


def real_tokens(d, c):
    ok = d["Dpad"][c] < 1e8
    return d["Gpad"][c][ok], d["Dpad"][c][ok], ok


fig = plt.figure(figsize=(3.4, 4.2))   # 27 Oct: more compact; 3 Nov: 4.65 -> 4.2 (PRL length)
gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1], hspace=0.62, wspace=0.42, left=0.13, right=0.98, top=0.98, bottom=0.075)
A = fig.add_subplot(gs[0, :]); B = fig.add_subplot(gs[1, 0]); C = fig.add_subplot(gs[1, 1])
Dp = fig.add_subplot(gs[2, 0]); E = fig.add_subplot(gs[2, 1], sharey=Dp)

# (a) collapsed order parameter
XF = np.geomspace(0.45, 7, 300)
for m in sorted(set(MR) | set(ALL)):                      # every receiver; open circles: tokens not resolved (eps < K_c)
    for S in (REAL, ALL):
        if m not in S: continue
        d = S[m]; y = d["rhat"]; xj = d["XS"][None, :, None] * np.exp(rng.normal(0, 0.012, y.shape))
        res = np.broadcast_to((d["eps"] >= 1)[:, None, None], y.shape)
        A.scatter(xj[res], y[res], s=6, color=col(m), alpha=0.85, lw=0, zorder=3)
        A.scatter(xj[~res], y[~res], s=6, facecolor="none", edgecolor=col(m), alpha=0.8, lw=0.5, zorder=2)
    y = np.concatenate([S[m]["rhat"][S[m]["eps"] >= 1] for S in (REAL, ALL) if m in S]); XSm = (REAL.get(m) or ALL.get(m))["XS"]
    print(f"(a) m={m}: {len(y)} resolved receivers, median |r_hat - sqrt(1-Kc^2/K^2)| (K >= 1.2 Kc) = "
          f"{np.median(np.abs(y - KUR(XSm)[None, :, None])[:, XSm >= 1.2]):.3f}")
A.plot(XF, KUR(XF), "-", color=INK, lw=1.0, zorder=10, label=r"$\sqrt{1-K_c^2/K^2}$")
A.set_xscale("log"); A.set_xticks([0.5, 1, 2, 3]); A.set_xticklabels(["0.5", "1", "2", "3"]); A.minorticks_off(); A.set_xlim(0.45, 3.3)   # 27 Oct: to 3
A.set_ylim(-0.04, 1.08); A.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1]); A.set_xlabel(r"coupling $K/K_c$", labelpad=1); A.set_ylabel(r"order parameter $\hat r$", labelpad=1)
_hl = [A.scatter([], [], s=10, color=col(m)) for m in MR] + [A.get_lines()[-1]]
A.legend(_hl, [f"$m={m}$" for m in MR] + [r"$\sqrt{1-K_c^2/K^2}$"], fontsize=6, loc="upper left", bbox_to_anchor=(0.0, 1.0),
         handletextpad=0.3, handlelength=1.4, labelspacing=0.25, frameon=False, borderaxespad=0.1)   # 27 Oct: one column, the root below
style(A, "a")
ia = A.inset_axes([0.725, 0.12, 0.265, 0.4])   # 3 Nov: a bit wider, like the main panel   # 27 Oct: squarer, higher   # 26 Oct: lower right, below the curve   # 26 Oct: the empty corner (K < K_c, r > 0)
for m in MQ:
    d = RAND[m]
    for c in range(len(d["G"])):
        G, Dl = d["G"][c], d["D"][c]; Kc = th.sums(G, Dl)[2]
        y = np.array([[rhat(G, Dl, x * Kc, d["r"][c, i, s]) for s in range(d["r"].shape[2])] for i, x in enumerate(d["XS"])])
        ia.scatter(np.repeat(d["XS"], y.shape[1]), y.ravel(), s=2.5, color=colq(m), alpha=0.7, lw=0, zorder=3)
ia.plot(XF, KUR(XF), "-", color=INK, lw=0.7, zorder=10)
ia.set_xscale("log"); ia.set_xlim(0.45, 4.5); ia.set_xticks([1, 4]); ia.set_xticklabels(["1", "4"]); ia.minorticks_off()
ia.set_ylim(-0.04, 1.05); ia.set_yticks([0, 1]); mini(ia); ia.set_title("random contexts", fontsize=5.3, pad=1.5)

# (b) Lorentzian collapse
x = np.linspace(-12, 12, 400)
for m in MR:
    d = REAL[m]
    for c in range(d["rhat"].shape[0]):
        G, Dl, ok = real_tokens(d, c)
        for i in np.where(d["XS"] >= 1.2)[0]:
            for s in range(d["r"].shape[2]):
                K = d["XS"][i]; mu = K * d["r"][c, i, s]
                if mu <= 0: continue
                a = np.abs(d["zeta"][c, i, s][ok]) ** 2
                B.scatter(Dl / mu, a * mu ** 2 / (K * G) ** 2, s=1.8, color=col(m), alpha=0.5, lw=0, rasterized=True)
B.plot(x, 1 / (1 + x ** 2), color=INK, lw=0.6, zorder=5, label=r"$1/(1+x^2)$")
B.set_yscale("log"); B.set_xlim(-12, 12); B.set_ylim(4e-3, 2.2); B.set_xticks([-10, 0, 10]); B.set_yticks([1e-2, 1]); B.minorticks_off()
B.set_xlabel(r"$\Delta_j/Kr$", labelpad=1); B.set_ylabel(r"$|w_j|^2(Kr)^2/(KG_j)^2$", labelpad=1)
B.legend(fontsize=5.5, loc="upper left", bbox_to_anchor=(0.02, 1.02), frameon=False, handlelength=1.6, handletextpad=0.4, borderaxespad=0); style(B, "b")


# (c) error of the weights against m
def err_real(d, x, locked):
    i = int(np.argmin(np.abs(d["XS"] - x))); e = []
    for c in range(d["rhat"].shape[0]):
        G, Dl, ok = real_tokens(d, c); K = d["XS"][i]
        r = th.implicit(G, Dl, K) if locked else 0.0
        if locked and r <= 0: continue
        a = K ** 2 * G ** 2 / ((K * r) ** 2 + Dl ** 2)
        e += [np.abs(np.abs(d["zeta"][c, i, s][ok]) ** 2 - a).sum() / a.sum() for s in range(d["r"].shape[2])]
    return np.median(e)


def err_rand(d, x, locked):
    i = int(np.argmin(np.abs(d["XS"] - x))); e = []
    for c in range(len(d["G"])):
        G, Dl = d["G"][c], d["D"][c]; K = x * th.sums(G, Dl)[2]
        r = th.implicit(G, Dl, K) if locked else 0.0
        a = K ** 2 * G ** 2 / ((K * r) ** 2 + Dl ** 2)
        e += [np.abs(np.abs(d["zeta"][c, i, s]) ** 2 - a).sum() / a.sum() for s in range(d["zeta"].shape[2])]
    return np.median(e)


RES = []                              # (m, eps/Kr, error) of every receiver at every locked coupling K >= 1.2 K_c (27 Oct)
for S in (REAL, ALL):
    for m, d in S.items():
        IL = np.where(d["XS"] >= 1.2)[0]; pick = np.random.default_rng(m).choice(IL, d["rhat"].shape[0])   # 27 Oct: one point
        for c in range(d["rhat"].shape[0]):                     # per receiver, at a coupling K >= 1.2 K_c drawn at random
            for i in [pick[c]]:
                K = d["XS"][i]
                G, Dl, ok = real_tokens(d, c); r = th.implicit(G, Dl, K)
                if r <= 0 or ok.sum() < 2: continue              # one token: exact, nothing to test
                a = K ** 2 * G ** 2 / ((K * r) ** 2 + Dl ** 2)
                e = np.median([np.abs(np.abs(d["zeta"][c, i, s][ok]) ** 2 - a).sum() / a.sum() for s in range(d["r"].shape[2])])
                RES.append((m, d["eps"][c] / (K * r), e))
RES = np.array(RES); RX = RES[:, 0] * RES[:, 1]; rs = RES[:, 1] >= 1      # x = m eps / Kr; resolved: eps >= Kr
for m in sorted(set(RES[:, 0])):                # 26 Oct: the leak of the beats, error ~ Kr/(m eps): one law for every receiver
    s_ = RES[:, 0] == m
    C.scatter(RX[s_ & rs], RES[s_ & rs, 2], s=8, color=col(m), alpha=0.85, lw=0, zorder=3)
    C.scatter(RX[s_ & ~rs], RES[s_ & ~rs, 2], s=8, facecolor="none", edgecolor=col(m), alpha=0.85, lw=0.5, zorder=3)
QX, QE = [], []                                  # random contexts (detunings spaced by 1), median per m
for m in MQ:
    d = RAND[m]; i = int(np.argmin(np.abs(d["XS"] - 2.0))); xs, es = [], []
    for c in range(len(d["G"])):
        G, Dl = d["G"][c], d["D"][c]; K = 2.0 * th.sums(G, Dl)[2]; r = th.implicit(G, Dl, K)
        a = K ** 2 * G ** 2 / ((K * r) ** 2 + Dl ** 2)
        xs.append(m / (K * r)); es.append(np.median([np.abs(np.abs(d["zeta"][c, i, s]) ** 2 - a).sum() / a.sum() for s in range(d["zeta"].shape[2])]))
    QX.append(np.median(xs)); QE.append(np.median(es))
QX, QE = np.array(QX), np.array(QE)
ct, cq = np.exp(np.median(np.log(RES[rs, 2] * RX[rs]))), np.exp(np.median(np.log(QE * QX)))
xx = np.geomspace(RX.min() / 3, RX.max() * 3, 50)                                # 26 Oct: the fit alone, in grey, beyond the data
C.plot(xx, ct / xx, "-", color=MUTED, lw=0.9, zorder=1, label=r"$\propto Kr/m\varepsilon$")
sl = np.polyfit(np.log(RX[rs]), np.log(RES[rs, 2]), 1)[0]; slq = np.polyfit(np.log(QX), np.log(QE), 1)[0]
print(f"(c) {len(RES)} trained receivers, {rs.sum()} resolved: slope {sl:.2f} (trained, resolved), {slq:.2f} (random); "
      f"prefactors {ct:.2f}, {cq:.2f}; median error resolved {np.median(RES[rs, 2]):.3f}, unresolved {np.median(RES[~rs, 2]):.3f}")
for m in MR:                                     # 28 Oct: numbers of Sec. S2, at 2 K_c
    print(f"(c) m={m}: median error at 2K_c, resolved {err_real(REAL[m], 2.0, True):.4f}")
_u = [e for m, d in ALL.items() for e in [err_real({**d, **{k: d[k][d["eps"] < 1] for k in ("rhat", "zeta", "Gpad", "Dpad", "r")}}, 2.0, True)] if (d["eps"] < 1).any()]
print(f"(c) unresolved (eps < K_c), median error at 2K_c per m: {np.round(_u, 3)}")
_j = np.linalg.lstsq(np.c_[np.ones(rs.sum()), np.log(RES[rs, 1]), np.log(RES[rs, 0])], np.log(RES[rs, 2]), rcond=None)[0]
print(f"(c) joint fit, resolved: error ~ (eps/Kr)^{_j[1]:.2f} m^{_j[2]:.2f}")
C.set_xscale("log"); C.set_yscale("log"); C.minorticks_off()
C.set_xlim(RX.min() / 3, RX.max() * 3); C.set_ylim(RES[:, 2].min() / 3, RES[:, 2].max() * 3)   # 27 Oct: every point inside
C.set_xlabel(r"$m\varepsilon/Kr$", labelpad=1); C.set_ylabel("error of the weights", labelpad=1)
C.legend(fontsize=5.5, loc="lower left", frameon=False, handlelength=1.6, borderaxespad=0.1); style(C, "c")

# (d, e) performance relative to softmax
LAB = os.path.join(os.path.dirname(HERE), "lab", "results", "lm")
BENCH = os.path.join(os.path.dirname(HERE), "bench", "results", "lm", "L2")
SCH_A, BSCH, SCH_B = "lr0.001-e6-w0.02-dc0.2", "lr0.001-f0.1-st6p100-w0.02-dc0.2", "lr0.001-e1-w0.02-dc0.2"
MODELS = (("SM", ["SM"], "SM-full-D128-h{h}", None, None, None),
          ("PG", ["PG"], "PG-full-D128-h{h}", GREEN, "o-", "PG"),
          ("KM", ["KM-same", "KM"], "KM-full-D128-h{h}-ihead-pow-ar1e-08-delta0.001", BLUE, "o-", "radio head"),
          ("KMe", ["KMe1-same", "KMe1"], "KM-full-D128-h{h}-ihead-pow-ar1e-08-delta0.001-expl", "#82b4ea", "s--", "one effective token"))
acc = lambda f: json.load(open(f))["test"]["acc"] if os.path.exists(f) else None


def runs_a(labs, btag, h):
    """2 Nov: (d) on the protocol of (e), all of TinyStories, one epoch, same initial ageing (lab/plans/lm_F.txt)"""
    out = {}
    for s in (0, 1, 2):
        for lb in labs:
            a = acc(os.path.join(LAB, lb, f"L2-D128-h{h}-f1", SCH_B, f"s{s}.json"))
            if a is not None:
                out[s] = a; break
    return out


def runs_b(labs, d):
    out = {}
    for s in (0, 1, 2):
        for lb in labs:
            a = acc(os.path.join(LAB, lb, f"L2-D{4 * d}-h4-f1", SCH_B, f"s{s}.json"))
            if a is not None:
                out[s] = a; break
    return out


def ratios(m, sm):
    common = sorted(set(m) & set(sm))
    if common:
        return [m[s] / sm[s] for s in common]
    return [np.mean(list(m.values())) / np.mean(list(sm.values()))] if m and sm else []


MSP = [8, 16, 32, 64]; MSPE = MSP + [128]   # 29 Oct: (e) to m = 128 (D1024 runs); (d) stays at m <= 64; 3 Nov: (e) briefly cut at 64, restored
got_e = False
for key, labs, btag, c, ls, name in MODELS[1:]:
    for ax, get, get_sm in ((Dp, lambda d: runs_a(labs, btag, 128 // d), lambda d: runs_a(MODELS[0][1], MODELS[0][2], 128 // d)),
                            (E, lambda d: runs_b(labs, d), lambda d: runs_b(MODELS[0][1], d))):
        pts = [(mm, ratios(get(2 * mm), get_sm(2 * mm))) for mm in (MSP if ax is Dp else MSPE)]
        pts = [(mm, v) for mm, v in pts if v]
        print("d" if ax is Dp else "e", name, {mm: [round(v_, 4) for v_ in v] for mm, v in pts})
        if not pts:
            continue
        got_e |= ax is E
        zo = 2 if "effective" in name else 3          # 3 Nov: the effective token under the radio head
        ax.plot([mm for mm, _ in pts], [np.mean(v) for _, v in pts], ls, color=c, ms=3.4, lw=1.0, label=name, zorder=zo)
        for mm, v in pts:
            if len(v) > 1:
                ax.scatter([mm] * len(v), v, s=2.5, color=c, lw=0, alpha=0.6, zorder=zo + 1)
for ax, tag, title, ms_, xr in ((Dp, "d", r"$d_{\rm model}=128$", MSP, 78), (E, "e", "four heads", MSPE, 156)):
    ax.axhline(1, color="#7a7a7a", lw=0.8, zorder=1)
    ax.set_xscale("log", base=2); ax.set_xticks(ms_); ax.set_xticklabels(map(str, ms_)); ax.minorticks_off()
    ax.set_xlim(6.5, xr); ax.set_xlabel(r"dimension $m$", labelpad=1); ax.set_title(title, fontsize=6.5, pad=2, color=INK); style(ax, tag)
plt.setp(E.get_yticklabels(), visible=False)
Dp.text(7, 1.003, "softmax", fontsize=6, color="#7a7a7a", ha="left", va="bottom")
Dp.set_ylabel("accuracy / softmax", labelpad=1); Dp.set_ylim(0.9, 1.035)
h_, l_ = Dp.get_legend_handles_labels(); Dp.legend(h_, l_, fontsize=5.5, frameon=False, loc="lower right", handlelength=1.6, borderaxespad=0.1)
if not got_e:
    E.text(0.5, 0.62, "running", transform=E.transAxes, ha="center", va="center", fontsize=6.5, color=MUTED)
fig.savefig(os.path.join(HERE, OUT + ".pdf"), bbox_inches="tight", pad_inches=0.01)
fig.savefig(os.path.join(HERE, OUT + ".png"), dpi=250, bbox_inches="tight", pad_inches=0.01)
