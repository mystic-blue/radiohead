"""MQAR (lab runs, 19 Oct 2026), one column, v2.  Top: the task (16 key-value pairs, then 16 queries, 20% of them for an
absent key, whose answer is the null class).  Two layers, one head of width d_h, 80 epochs, learning rate of each model
chosen on validation at d_h 64; markers = mean over seeds, bars = range.
(a) accuracy on present keys against d_h; (b) at d_h 64, accuracy against the number of pairs at test (trained on 16);
(c) the transition, query by query, in the radio head at d_h 64 (last layer): order parameter r / r_inf,
r_inf = (sum_j G_ij^2)^(1/2) its strong-coupling value, against K/K_c, present keys blue, absent pink; dashed, the
one-station law sqrt(1 - (K_c/K)^2), no parameter.  Inset: total weight sum_j |w_ij|^2, log-log; dashed,
min[(K/K_c)^2, 1] (linear response below threshold, normalized above).
Reads ../lab/results/mqar; the radio head is probed once and cached in dumps/mqar_transition.json.
Writes fig_mqar.{pdf,png}."""
import json, os, sys
import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FixedLocator, NullLocator
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from theory import BLUE, GOLD, GREEN, INK, MUTED, ORANGE, PINK, plt, style
LAB = os.path.join(os.path.dirname(HERE), "lab")
sys.path.insert(0, LAB)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "bench"))
from mqar_table import chosen_lr, load                                     # noqa: E402

COL = {"SM": "#7a7a7a", "GATED": GREEN, "SIGMOID": GOLD, "PG": ORANGE, "KM": BLUE, "KMe2": "#82b4ea"}
NAME = {"SM": "softmax + RoPE", "GATED": "gated", "SIGMOID": "sigmoid + ALiBi", "PG": "Pasqualetti–Guo",
        "KM": "radio head", "KMe2": "radio head, explicit"}
RIVALS = ["SM", "GATED", "SIGMOID", "PG", "KMe2", "KM"]
DH, P = [16, 32, 64, 128], [16, 32, 64, 128]
KEY, VAL = "#5b5a56", "#b3b0a6"

rows = load()
lr = chosen_lr(rows)
rows = [r for r in rows if lr.get(r["model"]) == r["lr"]]


def stat(sel, f):
    v = np.array([f(r) for r in sel], dtype=float)
    return (v.mean(), v.min(), v.max()) if len(v) else (np.nan,) * 3


def curve(A, x, ys, m):
    mu, lo, hi = map(np.array, zip(*ys))
    ok = ~np.isnan(mu)
    A.errorbar(np.array(x)[ok], mu[ok], yerr=[(mu - lo)[ok], (hi - mu)[ok]], fmt="o", ls="-", ms=2.2,
               lw=1.2 if m == "KM" else 0.8, elinewidth=0.5, capsize=0, color=COL[m], label=NAME[m], zorder=3 if m == "KM" else 2)


def log2axis(A, ticks):
    A.set_xscale("log", base=2); A.xaxis.set_major_locator(FixedLocator(ticks)); A.set_xticklabels([str(t) for t in ticks])
    A.xaxis.set_minor_locator(NullLocator()); A.set_ylim(-0.03, 1.05)


plt.rcParams.update({"font.size": 7, "xtick.labelsize": 6.3, "ytick.labelsize": 6.3, "axes.labelsize": 7})
fig = plt.figure(figsize=(3.4, 4.2))
gs = fig.add_gridspec(4, 2, height_ratios=[0.36, 1.0, 0.16, 1.0], hspace=0.5, wspace=0.45, left=0.15, right=0.97, top=0.98, bottom=0.08)

# the task
S = fig.add_subplot(gs[0, :]); S.set_position(S.get_position().translated(-0.05, 0)); S.axis("off"); S.set_xlim(0, 1); S.set_ylim(0, 1)


def boxes(y, items, x0, w=0.052, gap=0.008):
    x = x0
    for txt, col in items:
        S.add_patch(FancyBboxPatch((x, y), w, 0.28, boxstyle="round,pad=0.004,rounding_size=0.012", fc=col, ec="none"))
        S.text(x + w / 2, y + 0.14, txt, ha="center", va="center", fontsize=6.2, color="white")
        x += w + gap
    return x


x = boxes(0.62, [("k₁", KEY), ("v₁", VAL), ("k₂", KEY), ("v₂", VAL), ("k₃", KEY), ("v₃", VAL)], 0.0)
S.text(x + 0.005, 0.76, "…", fontsize=8, va="center", color=INK)
S.text(x + 0.06, 0.76, "16 pairs, then 16 queries", fontsize=6.3, color=MUTED, va="center")
x = boxes(0.12, [("k₂", BLUE)], 0.0); S.text(x + 0.005, 0.26, r"$\to$ v₂   key present", fontsize=6.5, va="center", color=INK)
x = boxes(0.12, [("k₉", PINK)], 0.42); S.text(x + 0.005, 0.26, r"$\to\varnothing$   key absent (20 %)", fontsize=6.5, va="center", color=INK)

# (a) extrapolation, every model at d_h 64
A = fig.add_subplot(gs[1, 0])
get = lambda p: (lambda r: r["test"]["acc"]) if p == 16 else (lambda r: r["extra"].get(f"P{p}", {}).get("acc", np.nan))
for m in RIVALS:
    sel = [r for r in rows if r["model"] == m and r["d"] == 64]
    if sel:
        curve(A, P, [stat(sel, get(p)) for p in P], m)
A.axhline(0.2, color=MUTED, lw=0.5, ls=":"); A.axvspan(13.5, 19, color="#f1f0eb", zorder=0)
log2axis(A, P); A.set_xlabel("pairs at test"); A.set_ylabel(r"accuracy ($d_h=64$)"); style(A, "a")
A.text(16.2, 0.05, "train", fontsize=5, color=MUTED, ha="center")
h, l = A.get_legend_handles_labels()
LG = fig.add_subplot(gs[2, :]); LG.axis("off")
LG.legend(h, l, fontsize=5.4, loc="center", ncol=3, handlelength=1.5, columnspacing=1.0, borderaxespad=0)

# the radio head, query by query: for each d_h the KM run with the best validation (seed chosen on validation only)
CACHE = os.path.join(HERE, "dumps", "mqar_queries.json")


def probe(f):
    from layers import Cfg
    from net import Net, Model
    from tasks import mqar
    s = torch.load(f, map_location="cpu", weights_only=False)
    cfg = Cfg(**s["cfg"])
    m = Model(Net(**s["net"]), cfg); m.load_state_dict(s["best"]); m.eval(); m.keep(True)
    _, _, te, extra, _ = mqar(batch_size=256)
    out = {}
    for nm, ld in {"P16": te, **extra}.items():
        o = {k: [] for k in ("kk", "r", "ws", "pres", "ok")}
        with torch.no_grad():
            for b in ld:
                lo = m(b["x"]); dbg = m.blocks[-1].last[1]
                q = b["y"] != -100
                o["kk"] += dbg["KoverKc"][:, 0][q].tolist()
                o["r"] += ((dbg["mu"][:, 0] - cfg.km_delta) / dbg["G2sum"][:, 0].sqrt())[q].tolist()
                o["ws"] += dbg["wsum"][:, 0][q].tolist()
                o["pres"] += b["present"][q].int().tolist(); o["ok"] += (lo.argmax(-1) == b["y"])[q].int().tolist()
        out[nm] = o
    return out


import glob
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
for d in DH:
    km = [r for r in rows if r["model"] == "KM" and r["d"] == d]
    if not km:
        continue
    best = max(km, key=lambda r: (r["val"]["acc"], -r["val"]["loss"]))
    key = f"d{d}s{best['seed']}"
    if key not in cache:
        fs = glob.glob(os.path.join(LAB, "results", "mqar", "KM", f"L2-D{d}-h1", f"lr{lr['KM']:g}-*", f"s{best['seed']}.pt"))
        cache[key] = probe(fs[0]); os.makedirs(os.path.dirname(CACHE), exist_ok=True); json.dump(cache, open(CACHE, "w"))
    cache.setdefault("pick", {})[str(d)] = key
Q = {int(d): {p: {k: np.array(v) for k, v in cache[key][p].items()} for p in cache[key]} for d, key in cache["pick"].items()}
xx = np.logspace(-3, 3, 400)
law = lambda x: np.sqrt(np.clip(1 - np.asarray(x, float) ** -2.0, 0, None))

# (b) the transition at the training length, d_h 64
B = fig.add_subplot(gs[1, 1])
ins = B.inset_axes([0.22, 0.45, 0.36, 0.36])
e = Q[64]["P16"]
for k, c, s_ in (("absent", PINK, e["pres"] == 0), ("present", BLUE, e["pres"] == 1)):
    B.scatter(e["kk"][s_], e["r"][s_], s=1.6, color=c, alpha=0.5, lw=0, rasterized=True, zorder=3, label=f"key {k}")
    ins.scatter(e["kk"][s_], e["ws"][s_], s=0.6, color=c, alpha=0.5, lw=0, rasterized=True, zorder=3)
B.plot(xx, law(xx), color=INK, lw=0.7, ls="--", zorder=4, label=r"$\sqrt{1-(K_c/K)^2}$")
ins.plot(xx, np.minimum(xx ** 2, 1.0), color=INK, lw=0.6, ls="--", zorder=4)
for a_ in (B, ins):
    a_.axvline(1, color=MUTED, lw=0.5, ls=":"); a_.set_xscale("log"); a_.set_xlim(1e-2, 30)
B.set_ylim(-0.03, 1.05); B.set_xlabel(r"$K/K_c$"); B.set_ylabel(r"order parameter $r/r_\infty$", labelpad=1)
B.set_xticks([1e-2, 1e-1, 1, 10]); B.set_xticklabels(["0.01", "0.1", "1", "10"]); B.minorticks_off()
style(B, "b")
ins.set_yscale("log"); ins.set_ylim(5e-5, 3); ins.tick_params(labelsize=4.5, length=1.5, pad=1)
ins.set_yticks([1e-4, 1e-2, 1]); ins.set_xticks([1e-2, 1]); ins.set_xticklabels(["0.01", "1"]); ins.minorticks_off()
ins.set_title(r"$\sum_j|w_{ij}|^2$", fontsize=5, pad=1, color=INK); ins.patch.set_alpha(0.0)
for sp in ("top", "right"):
    ins.spines[sp].set_visible(False)
B.text(0.97, 0.05, "16 pairs", transform=B.transAxes, ha="right", fontsize=5.5, color=MUTED)

# (c) width: which side of the threshold each query falls on, against the context at test
SH = {16: "#bcd6f4", 32: "#82b4ea", 64: BLUE, 128: "#123f78"}
C = fig.add_subplot(gs[3, 0])
for d in sorted(Q):
    q = [Q[d][f"P{p}"] for p in P]
    C.plot(P, [((e["kk"] > 1) & (e["pres"] == 0)).sum() / (e["pres"] == 0).sum() for e in q], "-", color=SH[d], lw=1.0,
           marker="o", ms=2.2, label=rf"$d_h={d}$")
    C.plot(P, [((e["kk"] < 1) & (e["pres"] == 1)).sum() / (e["pres"] == 1).sum() for e in q], "--", color=SH[d], lw=1.0,
           marker="s", ms=1.8)
log2axis(C, P); C.set_xlabel("pairs at test"); C.set_ylabel("fraction on the wrong side", labelpad=1); style(C, "c")
C.plot([], [], "-", color=MUTED, lw=1.0, label=r"absent, $K>K_c$"); C.plot([], [], "--", color=MUTED, lw=1.0, label=r"present, $K<K_c$")
C.legend(fontsize=5, loc="upper left", handlelength=1.8, labelspacing=0.25, borderaxespad=0.2)

# (d) where the one-station law fails: d_h 64 at 64 pairs, wrong answers marked
D_ = fig.add_subplot(gs[3, 1])
e = Q[64]["P64"]
for c, s_ in ((PINK, e["pres"] == 0), (BLUE, e["pres"] == 1)):
    D_.scatter(e["kk"][s_], e["r"][s_], s=1.6, color=c, alpha=0.45, lw=0, rasterized=True, zorder=3)
w = e["ok"] == 0
D_.scatter(e["kk"][w], e["r"][w], s=3, marker="x", color=INK, lw=0.3, alpha=0.35, rasterized=True, zorder=4, label="wrong")
D_.plot(xx, law(xx), color=INK, lw=0.7, ls="--", zorder=5)
D_.axvline(1, color=MUTED, lw=0.5, ls=":"); D_.set_xscale("log"); D_.set_xlim(1e-2, 30); D_.set_ylim(-0.03, 1.05)
D_.set_xticks([1e-2, 1e-1, 1, 10]); D_.set_xticklabels(["0.01", "0.1", "1", "10"]); D_.minorticks_off()
D_.set_xlabel(r"$K/K_c$"); D_.set_ylabel(r"$r/r_\infty$", labelpad=1); style(D_, "d")
lg = D_.legend(fontsize=5, loc="upper left", markerscale=1.5, handletextpad=0.2, borderaxespad=0.2)
lg.legend_handles[0].set_alpha(1)
D_.text(0.03, 0.78, r"64 pairs, $d_h=64$", transform=D_.transAxes, ha="left", fontsize=5.5, color=MUTED)

fig.savefig(os.path.join(HERE, "fig_mqar.pdf")); fig.savefig(os.path.join(HERE, "fig_mqar.png"), dpi=300)
print("lr:", lr, "picked:", cache["pick"])
