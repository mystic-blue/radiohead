"""MQAR phase B, the radio head query by query (analysis, not a letter figure).  KM, every seed that trained, every d_h, test with 16..128 pairs.  Reads dumps/mqar_where.json
(last layer: K/K_c, total weight, weight on each pair, answer right or wrong).  Writes mqar_analysis2.png."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from theory import BLUE, INK, MUTED, PINK, plt, style
W = {k: {kk: np.array(v) for kk, v in d.items()} for k, d in json.load(open(os.path.join(HERE, "dumps", "mqar_where.json"))).items()}
DH, P = [16, 32, 64, 128], [16, 32, 64, 128]
SH = {16: "#bcd6f4", 32: "#82b4ea", 64: BLUE, 128: "#123f78"}
MK = {"KM": ("o", "-"), "KM-norm": ("s", "--")}
seeds = lambda lab, d: sorted({int(k.split("|")[2]) for k in W if k.startswith(f"{lab}|{d}|")})


def per(lab, d, f):
    """mean and range over seeds of f(query arrays) at each test length"""
    v = np.array([[f(W[f"{lab}|{d}|{s}|P{p}"]) for p in P] for s in seeds(lab, d)])
    return v.mean(0), v.min(0), v.max(0)


plt.rcParams.update({"font.size": 7, "xtick.labelsize": 6, "ytick.labelsize": 6})
fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.6), gridspec_kw=dict(hspace=0.5, wspace=0.38, left=0.07, right=0.99, top=0.93, bottom=0.1))
ab = lambda e: e["pres"] == 0
pr = lambda e: e["pres"] == 1
panels = [
    (ax[0, 0], "KM", lambda e: np.median(e["wsum"][ab(e)]), r"absent: median $\sum_j|w_{ij}|^2$", "a", "KM: the silence fills up"),
    (ax[0, 1], None, lambda e: np.mean(e["kk"][ab(e)] > 1), r"absent: fraction with $K>K_c$", "b", "absent keys cross threshold"),
    (ax[0, 2], None, lambda e: np.mean(e["ok"][ab(e)]), "absent: accuracy", "c", "answer 'absent'"),
    (ax[1, 0], None, lambda e: np.mean(e["kk"][pr(e)] < 1), r"present: fraction with $K<K_c$", "d", "present keys fall silent"),
    (ax[1, 1], None, lambda e: np.median(e["match"][pr(e)]), "present: weight on the right pair", "e", "present keys: reading the right pair"),
]
for a, only, f, yl, tag, title in panels:
    for lab in ("KM",):
        if only and lab != only: continue
        for d in DH:
            if not seeds(lab, d): continue
            mu, lo, hi = per(lab, d, f)
            a.errorbar(P, mu, yerr=[np.clip(mu - lo, 0, None), np.clip(hi - mu, 0, None)], fmt=MK[lab][0], ls=MK[lab][1], ms=2.2, lw=0.9, elinewidth=0.5,
                       capsize=0, color=SH[d], label=rf"$d_h={d}$" if lab == "KM" else None)
    a.set_xscale("log", base=2); a.set_xticks(P); a.set_xticklabels(map(str, P)); a.minorticks_off(); a.set_ylim(-0.03, 1.05)
    a.set_xlabel("pairs at test"); a.set_ylabel(yl, labelpad=1); a.set_title(title, fontsize=6.5); style(a, tag)
ax[0, 0].legend(fontsize=5.5, loc="upper left", handlelength=1.5)
# (f) the threshold predicts the error: one point per (model, d_h, seed, pairs), absent and present
a = ax[1, 2]
for k, e in W.items():
    lab, d, s, p = k.split("|")
    if lab != "KM": continue
    for side, sel, wrong, c in (("absent", ab(e), lambda e, s_: np.mean(e["kk"][s_] > 1), PINK), ("present", pr(e), lambda e, s_: np.mean(e["kk"][s_] < 1), BLUE)):
        a.scatter(wrong(e, sel), 1 - np.mean(e["ok"][sel]), s=7, marker=MK[lab][0], facecolor=c if lab == "KM" else "none",
                  edgecolor=c, lw=0.6, alpha=0.8)
a.plot([0, 1], [0, 1], color=MUTED, lw=0.5, ls=":")
a.scatter([], [], s=7, color=PINK, label="absent keys"); a.scatter([], [], s=7, color=BLUE, label="present keys")
a.legend(fontsize=5.5, loc="upper left", handletextpad=0.2)
a.set_xlim(-0.03, 1.03); a.set_ylim(-0.03, 1.03); a.set_xlabel("fraction on the wrong side of $K_c$"); a.set_ylabel("error rate", labelpad=1)
a.set_title("threshold vs error", fontsize=6.5); style(a, "f")
fig.savefig(os.path.join(HERE, "mqar_analysis2.png"), dpi=220)
