"""Fig. 2 (24 Oct 2026, lab data): the physics of the radio head in the four tasks, one column each.
Row 1, the law that the trained receiver obeys; row 2, the same law beyond training (accuracy against context length).
Mechanism panels (b, c): lab KM weights, lr 1e-3 (fixed for every model, as the accuracy panels), seed 0 (dumps/ci_lab_lr0.003_s0.json from
dump_ci_lab.py; falls back to the bench draft dumps/ci_draft_lr1e-3.json if missing).  The lr of each mechanism panel
can be changed with the environment variables FIG2_MECH_LR_COUNT / FIG2_MECH_LR_INDUCTION.
Accuracy panels (e, f, g): lab runs, every available seed (mean, min-max bars), lr of each model chosen on validation
at seed 0 (lab/plans/ci_lr_chosen.json for counting and induction; mqar_table.chosen_lr for recall).
Language (d, h): dumps/lab_lm_column_s*.json (dump_lm_lab.py; KM = KM-same) if present, (h) averaged over seeds with
the range; else the bench dumps/lm_column.json.  Bench version of this script: make_fig2_physics_bench.py.
Writes fig2_physics.{pdf,png}."""
import json, os, sys
import numpy as np
from matplotlib.patches import FancyBboxPatch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from theory import BLUE, GREEN, INK, MUTED, ORANGE, PINK, plt, style
from theory import style as _style


def style(a, tag=None):                                       # 25 Oct: no grids in this figure
    _style(a, tag, grid=False)


DUMP = os.path.join(HERE, "dumps")
DUMP_LM = os.path.join(HERE, "dumps_f1")    # 2 Nov: language (d, h) on all of TinyStories, one epoch (dump_lm_lab.py, dump_lm_recv.py)
import glob, math
LAB = os.path.join(os.path.dirname(HERE), "lab")
LABRES = os.path.join(LAB, "results")
# 25 Oct: the authors fixed lr 1e-3 for every model in counting and induction (no 3e-3 runs), mechanism and accuracy
MECH_LR = {t: float(os.environ.get(f"FIG2_MECH_LR_{t.upper()}", "0.001")) for t in ("count", "induction")}


def mech(task):
    f = os.path.join(DUMP, f"ci_lab_lr{MECH_LR[task]:g}_s0.json")
    if os.path.exists(f) and task in json.load(open(f)):
        print(f"[{task} mechanism] lab: {os.path.basename(f)}"); return json.load(open(f))[task]
    print(f"[{task} mechanism] lab dump missing ({os.path.basename(f)}): BENCH draft ci_draft_lr1e-3.json")
    return json.load(open(os.path.join(DUMP, "ci_draft_lr1e-3.json")))[task]


CI = {t: mech(t) for t in ("count", "induction")}
LR_CHOSEN = {k: 0.001 for k in json.load(open(os.path.join(LAB, "plans", "ci_lr_chosen.json")))}     # all at 1e-3,
LR_CHOSEN["induction|SM"] = 3e-4          # except softmax in induction: it does not learn at 1e-3 (5 seeds), 3e-4 is its validation choice


def lab_curves(task, m, Ts=(32, 64, 128, 256, 512), learned=0.99):
    """(n_seeds, len(Ts)) test accuracy against context length, lab runs at lr 1e-3.  Per seed, the longest schedule
    available (softmax induction also has 240-epoch runs); only seeds that learned the task, validation accuracy at the
    trained length >= `learned`, are kept (chosen on validation, never on test), unless none did; the count is printed."""
    lr = LR_CHOSEN[f"{task}|{m}"]
    by_seed = {}
    for f in glob.glob(os.path.join(LABRES, task, m, "L2-D64-h1", f"lr{lr:g}-e*-w0.02-dc0.2", "s*.json")):
        ep = int(f.split(f"lr{lr:g}-e")[1].split("-")[0]); sd = os.path.basename(f)[1:-5]
        if ep >= by_seed.get(sd, (0, None))[0]:
            by_seed[sd] = (ep, f)
    v, kept = [], []
    for sd, (ep, f) in sorted(by_seed.items()):
        r = json.load(open(f))
        if r["val"]["acc"] < learned and any(json.load(open(g))["val"]["acc"] >= learned for _, g in by_seed.values()):
            continue                                       # a model that never learns is shown as it is
        kept.append(f"{sd}(e{ep})" + ("" if r["val"]["acc"] >= learned else "!"))
        v.append([r["test"]["acc"]] + [r["extra"].get(f"T{T}", {}).get("acc", np.nan) for T in Ts[1:]])
    print(f"[{task} accuracy] {m}: lr {lr:g}, shown {len(kept)}/{len(by_seed)} (! = did not learn): {kept}")
    LEARNED[f"{task}|{m}"] = (len(kept), len(by_seed))
    return np.array(v, float) if v else np.full((1, len(Ts)), np.nan)


LEARNED = {}


def range_plot(a, Ts, v, c, zorder=None):
    mu = np.nanmean(v, 0)
    a.errorbar(Ts, mu, yerr=[mu - np.nanmin(v, 0), np.nanmax(v, 0) - mu], fmt="o-", color=c, ms=2.4,
               lw=1.1 if c == BLUE else 0.9, elinewidth=0.5, capsize=0, zorder=zorder or (3 if c == BLUE else 2))


MQ = json.load(open(os.path.join(DUMP, "mqar_queries.json")))
WH = {k: {kk: np.array(v) for kk, v in d.items()} for k, d in json.load(open(os.path.join(DUMP, "mqar_where.json"))).items() if k.startswith("KM|")}
KEY, VAL, OTHER = "#5b5a56", "#b3b0a6", "#d9d7d0"
SH = {16: "#bcd6f4", 32: "#82b4ea", 64: BLUE, 128: "#123f78"}
plt.rcParams.update({"font.size": 7, "xtick.labelsize": 6.3, "ytick.labelsize": 6.3, "axes.labelsize": 7})
fig = plt.figure(figsize=(7.0, 3.5))
gs = fig.add_gridspec(2, 4, hspace=0.48, wspace=0.36, left=0.055, right=0.965, top=0.85, bottom=0.11)


PHASE = {"recall": "locked + silent", "counting": "silent", "induction": "locked",
         "next-token prediction": "all"}


def cartoon(col, title, rows):
    p = gs[0, col].get_position(fig)
    S = fig.add_axes([p.x0 - 0.03, 0.895, 0.235, 0.055]); S.axis("off"); S.set_xlim(0, 1); S.set_ylim(0, 1)
    t_ = S.text(0.0, 1.05, title, fontsize=7.5, color=INK, fontweight="bold", va="bottom")
    bb = t_.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(S.transAxes.inverted())
    S.text(bb.x1 + 0.03, 1.05, PHASE[title], fontsize=5.8, color=MUTED, va="bottom", ha="left")
    for y, items in rows:
        x = 0.0
        for txt, c in items:
            if c is None:
                S.text(x + 0.01, y + 0.25, txt, fontsize=6.3, va="center", color=INK); x += 0.045 + 0.024 * len(txt); continue
            bw = 0.075 + 0.035 * max(0, len(txt) - 1)
            S.add_patch(FancyBboxPatch((x, y), bw, 0.5, boxstyle="round,pad=0.004,rounding_size=0.012", fc=c, ec="none"))
            S.text(x + bw / 2, y + 0.25, txt, ha="center", va="center", fontsize=6, color="white" if c not in (OTHER,) else INK)
            x += bw + 0.012


cartoon(2, "induction", [(0.1, [("x", OTHER), ("A", KEY), ("B", VAL), ("y", OTHER), ("z", OTHER), ("w", OTHER), ("A", BLUE), ("→ B", None)])])
cartoon(0, "recall", [(0.1, [("k₁", KEY), ("v₁", VAL), ("k₂", KEY), ("v₂", VAL), ("k₂", BLUE), ("→ v₂", None), ("k₉", PINK), (r"$\to\varnothing$", None)])])
cartoon(1, "counting", [(0.1, [("A", BLUE), ("x", OTHER), ("A", BLUE), ("y", OTHER), ("z", OTHER), ("A", BLUE), ("→ 2", None)])])
cartoon(3, "next-token prediction", [(0.1, [("a", OTHER), ("cat", OTHER), ("sat", OTHER), ("on", OTHER), ("the", BLUE), ("→ mat", None)])])

TT = [32, 64, 128, 256, 512]


def silent_side(a):                                          # 27 Oct: the silent phase, K < K_c, shaded; no labels
    a.axvspan(1e-3, 1, color="#edf1f6", lw=0, zorder=0)


def kk_axis(a, lo=0.1, hi=100):                              # 26 Oct: 0.1-100 in all four
    """the mechanism panels share the axis K/K_c, log (a-c: 0.1-10; language, d: 0.01-100)"""
    a.set_xscale("log"); a.set_xlim(lo, hi); t = [v for v in (1e-2, 1e-1, 1, 10, 100) if lo <= v <= hi]
    a.set_xticks(t); a.set_xticklabels([f"{v:g}" for v in t]); a.minorticks_off()


def ctx_axis(a, train, x0, floor, hi=512):
    """x from the first point of the panel to hi, y from the chance floor of the task (0 for induction)"""
    TTh = [t for t in TT + [1024] if t <= hi]
    a.set_xscale("log", base=2); a.set_xticks([t for t in TTh if t >= x0] if x0 in TTh else [x0] + [t for t in TTh if t > 1.5 * x0])
    a.set_xticklabels(map(str, a.get_xticks().astype(int))); a.minorticks_off()
    a.set_xlim(x0 / 1.12, hi * 1.03 * 1.1); a.set_ylim(floor - 0.03, 1.03); a.axvspan(x0 / 1.12, train * 1.12, color="#f1f0eb", zorder=0)
    a.set_xlabel("context length at test" if getattr(a, "_xl", False) else "")


# ---------------- induction
I = {k: np.array(v) for k, v in CI["induction"]["Q"].items()}
x = I["K"] * I["G"] / np.abs(I["Delta"])                     # locking of the answer: K/K_c of a single station
A = fig.add_subplot(gs[0, 2])
TC = {32: "#123f78", 64: BLUE, 128: "#82b4ea", 256: "#bcd6f4"}     # 25 Oct: dark = short contexts, as "as trained" in (b)
for T_, c in reversed(list(TC.items())):
    sel = (I["T"] == T_) if T_ < 256 else (I["T"] >= 256)
    A.scatter(x[sel], I["w"][sel], s=3, color=c, alpha=0.6, lw=0, rasterized=True, zorder=2,
              label=(f"T={T_}" + (" (train)" if T_ == 32 else "")) if T_ < 256 else r"T$\geq$256")
xt = np.logspace(-2, 2, 600)
A.plot(xt, np.minimum(1, xt ** 2.0), color=INK, lw=0.8, ls="--", zorder=4, label=r"$\min[1,(KG/\Delta)^2]$")
edges = np.logspace(-1, 2, 25)
mid = np.sqrt(edges[1:] * edges[:-1]); acc = [I["ok"][(x >= a) & (x < b)].mean() if ((x >= a) & (x < b)).sum() >= 5 else np.nan for a, b in zip(edges[:-1], edges[1:])]
A.plot(mid, acc, color=ORANGE, lw=1.1, zorder=5, label="accuracy")
A.axvline(1, color=MUTED, lw=0.6, ls=":"); silent_side(A); kk_axis(A); A.set_ylim(-0.03, 1.1)
A.set_xlabel(r"$K/K_c$ of the answer, $KG/|\Delta|$"); A.set_ylabel(r"$|w_j|^2$ of the answer", labelpad=1)
_h, _l = A.get_legend_handles_labels(); _o = [_l.index(r"$\min[1,(KG/\Delta)^2]$"), _l.index("accuracy")] + [i for i in reversed(range(len(_l))) if _l[i].startswith("T")]
lg = A.legend([_h[i] for i in _o], [_l[i] for i in _o], fontsize=5.6, loc="center left", bbox_to_anchor=(0.45, 0.45), handlelength=1.2, markerscale=3, labelspacing=0.15, borderpad=0.2, frameon=False)   # 26 Oct: theory first, then T
for h_ in lg.legend_handles: h_.set_alpha(1)
style(A, "c")
B = fig.add_subplot(gs[1, 2])
Ts = [32, 64, 128, 256, 512, 1024]                            # 27 Oct: to 1024, as language
for m, c in (("SM", "#7a7a7a"), ("PG", GREEN), ("KM", BLUE)):
    range_plot(B, Ts, lab_curves("induction", m, Ts), c)
_ins = []                                                      # fraction of answers inside the locking range, every seed
for _f in sorted(glob.glob(os.path.join(DUMP, f"ci_lab_lr{MECH_LR['induction']:g}_s*.json"))):
    _d = json.load(open(_f)).get("induction")
    if _d is None: continue
    _q = {k: np.array(v, float) for k, v in _d["Q"].items()}; _x = np.abs(_q["Delta"]) / (_q["K"] * _q["G"])
    _ins.append([np.mean(_x[_q["T"] == T] < 1) for T in Ts])
print(f"[induction] locking-range prediction from {len(_ins)} seeds")
_ins = np.array(_ins); B.plot(Ts, _ins.mean(0), "s--", color=INK, ms=2.2, lw=0.8, mfc="none", zorder=4)
ctx_axis(B, 32, 32, 0.0, 1024); B.set_ylabel("accuracy", labelpad=1)
for xx_, yy_, t_, c_, ha_ in ((75, 0.2, "softmax", "#7a7a7a", "left"), (140, 0.1, "PG", GREEN, "left"), (1000, 0.86, "radio head", BLUE, "right"),
                              (1000, 0.97, r"dashed: $|\Delta|<KG$", INK, "right")):   # 26 Oct: clear of the curves
    B.text(xx_, yy_, t_, fontsize=5.5, color=c_, ha=ha_, va="center")
style(B, "g")

# ---------------- recall (MQAR, lab)
e = {k: np.array(v) for k, v in MQ["d64s0"]["P16"].items()}
xx = np.logspace(-3, 3, 400)
C = fig.add_subplot(gs[0, 0]); ins = C.inset_axes([0.56, 0.22, 0.42, 0.46])   # 27 Oct: the same inset in (a), (b), (d)
for c, s_ in ((PINK, e["pres"] == 0), (BLUE, e["pres"] == 1)):
    C.scatter(e["kk"][s_], e["r"][s_], s=5, color=c, alpha=0.5, lw=0, rasterized=True, zorder=3)
    ins.scatter(e["kk"][s_], e["ws"][s_], s=1.6, color=c, alpha=0.5, lw=0, rasterized=True, zorder=3)
C.plot(xx, np.sqrt(np.clip(1 - xx ** -2.0, 0, None)), color=INK, lw=0.7, ls="--", zorder=4)
ins.plot(xx, np.minimum(xx ** 2, 1), color=INK, lw=0.6, ls="--", zorder=4)
for a_ in (C, ins):
    a_.axvline(1, color=MUTED, lw=0.5, ls=":"); a_.set_xscale("log"); a_.set_xlim(1e-2, 1e2)
kk_axis(C); C.set_ylim(-0.03, 1.05)
C.set_xlabel(r"$K/K_c$"); C.set_ylabel(r"$r/r_\infty$", labelpad=1); style(C, "a")
silent_side(C)
ins.set_yscale("log"); ins.set_ylim(5e-5, 3); ins.tick_params(labelsize=4.5, length=1.5, pad=1); ins.set_yticks([1e-4, 1e-2, 1])
ins.set_xticks([1e-2, 1, 1e2]); ins.set_xticklabels(["0.01", "1", "100"]); ins.minorticks_off(); ins.set_title(r"$\Sigma_j|w_j|^2$", fontsize=5, pad=1)
ins.patch.set_alpha(0)
for sp in ("top", "right"): ins.spines[sp].set_visible(False)
D_ = fig.add_subplot(gs[1, 0]); P = [16, 32, 64, 128]
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "lab"))
from mqar_table import chosen_lr, load                                     # noqa: E402
_rows = load(); _lr = chosen_lr(_rows); _rows = [r for r in _rows if _lr.get(r["model"]) == r["lr"] and r["d"] == 64]
_acc = lambda r, p: r["test"]["acc"] if p == 16 else r["extra"][f"P{p}"]["acc"]
PL = [16, 32, 64, 128, 256, 496]; TL = [2 * p + 16 for p in PL]
LONG = os.path.join(os.path.dirname(HERE), "lab", "results", "mqar_long")
for m, c in (("SM", "#7a7a7a"), ("PG", GREEN), ("KM", BLUE)):
    runs = [r for r in _rows if r["model"] == m]
    v = []
    for r in runs:
        lj = json.load(open(os.path.join(LONG, f"{m}-D64-lr{r['lr']:g}-s{r['seed']}.json")))["extra"]
        v.append([_acc(r, p) for p in PL[:4]] + [lj[f"P{p}"]["acc"] for p in PL[4:]])
    v = np.array(v); mu = v.mean(0)
    D_.errorbar(TL, mu, yerr=[mu - v.min(0), v.max(0) - mu], fmt="o-", color=c, ms=2.4, lw=1.1 if c == BLUE else 0.9,
                elinewidth=0.5, capsize=0, zorder=3 if c == BLUE else 2)
for xx_, yy_, t_, c_ in ((300, 0.9, "radio head", BLUE), (85, 0.55, "softmax", "#7a7a7a"), (100, 0.25, "PG", GREEN)):
    D_.text(xx_, yy_, t_, fontsize=5.5, color=c_)
D_.axhline(0.2, color=MUTED, lw=0.5, ls=":")
D_._xl = True; ctx_axis(D_, 48, 48, 0.2, 1024); style(D_, "e")

TC = {32: "#bcd6f4", 64: "#82b4ea", 128: BLUE, 256: "#123f78"}
# ---------------- counting: copies add below threshold, saturate above (mirror of the induction panel)
# the five networks of panel (f), pooled: their receivers fall on both sides of the threshold
_cs = [json.load(open(f))["count"] for f in sorted(glob.glob(os.path.join(DUMP, f"ci_lab_lr{MECH_LR['count']:g}_s*.json")))]
Cq = {k: np.concatenate([np.array(c["Q"][k], float) for c in _cs]) for k in ("q_kk", "q_T", "q_k", "q_wsum")}
Cok = np.concatenate([np.array(c["D"]["ok"], float) for c in _cs])   # one row per query, aligned with the q_ rows
print("[count mechanism] lab, pooled seeds:", len(_cs))
E = fig.add_subplot(gs[0, 1])
# one network that reads the count as a number (trained on 0-15 copies, contexts of 32; radio head started at K = K_c,
# seed 0), at the trained length; its counting layer with the coupling raised, K -> cK, and the output gain lowered, g -> g/c,
# which leaves the layer unchanged below threshold and brings the threshold to fewer copies (dumps/count_sat_s0.json, ksat.py)
SAT = json.load(open(os.path.join(DUMP, "count_sat_s0.json")))
for c2, col in (("1", "#123f78"), ("36", "#82b4ea")):
    r = [x for x in SAT[c2] if 1 <= x["k"] <= 15]; kx = [math.sqrt(x["kk2"]) for x in r]
    E.plot(kx, [x["ws"] for x in r], "o", color=col, ms=2.0, lw=0, zorder=4)
# accuracy against K/K_c, pooled over couplings raised from 1 to 40 times (1-15 copies each), binned
_ax = np.array([math.sqrt(x["kk2"]) for c2 in SAT for x in SAT[c2] if 1 <= x["k"] <= 15])
_ac = np.array([x["acc"] for c2 in SAT for x in SAT[c2] if 1 <= x["k"] <= 15])
_ed = np.logspace(-1, 2, 31); _mid = np.sqrt(_ed[1:] * _ed[:-1])   # 26 Oct: to 100, as the axis
_acc = [(_ac[(_ax >= a) & (_ax < b)].mean() if ((_ax >= a) & (_ax < b)).any() else np.nan) for a, b in zip(_ed[:-1], _ed[1:])]
_ok = ~np.isnan(_acc); E.plot(_mid[_ok], np.array(_acc)[_ok], "-", color=ORANGE, lw=1.1, zorder=2)
xt = np.logspace(-2, 2, 400); E.plot(xt, np.minimum(1, xt ** 2), "--", color=INK, lw=0.7, zorder=3)
E.axvline(1, color=MUTED, lw=0.5, ls=":"); silent_side(E); kk_axis(E); E.set_ylim(-0.03, 1.1); E.set_yticks([0, 0.5, 1])
E.text(0.11, 0.94, "accuracy", color=ORANGE, fontsize=5.5, va="top")
eins = E.inset_axes([0.56, 0.22, 0.42, 0.46])                   # the same, against the number of copies: linear, then saturated
for c2, col in (("1", "#123f78"), ("36", "#82b4ea")):
    r = [x for x in SAT[c2] if x["k"] <= 15]
    eins.plot([x["k"] for x in r], [x["ws"] for x in r], "-", color=col, lw=0.9)
eins.text(15, 0.3, "as trained", color="#123f78", fontsize=5, ha="right"); eins.text(15, 0.86, r"coupling $\times6$", color="#82b4ea", fontsize=4.6, ha="right", va="top")   # 27 Oct: by the inset
eins.set_xlim(0, 15); eins.set_ylim(0, 1.1); eins.set_xticks([0, 15]); eins.set_yticks([0, 1])
eins.tick_params(labelsize=4.5, length=1.5, pad=1); eins.set_xlabel("copies", fontsize=5, labelpad=0); eins.patch.set_alpha(0)
for sp in ("top", "right"): eins.spines[sp].set_visible(False)
E.set_xlabel(r"$K/K_c$, adding copies"); E.set_ylabel(r"$\Sigma_j|w_j|^2$", labelpad=1)
style(E, "b")
F = fig.add_subplot(gs[1, 1])
# weight the counting layer puts on the copies (1-15 copies, random positions), relative to the trained length 32
# (dumps/count_reg_wcopies.json, kregw.py); dashed: softmax 32/T (uniform over the context), radio head
# <w>_[1,T)/<w>_[1,32) from the weight of one copy against its age (dumps/count_reg_len.json, kreglen.py)
WC = json.load(open(os.path.join(DUMP, "count_reg_wcopies.json"))); Tc = [32, 64, 128, 256, 512]
for m, c in (("SM", "#7a7a7a"), ("PG", GREEN), ("KM-k1", BLUE)):
    v = np.array([[WC[k][str(T)] for T in Tc] for k in WC if k.startswith(m + "|")])
    if len(v): range_plot(F, Tc, v, c)
tt = np.geomspace(32, 512, 50); F.plot(tt, 32 / tt, "--", color="#7a7a7a", lw=0.8, zorder=1)
_rl = os.path.join(DUMP, "count_reg_len.json")
if os.path.exists(_rl):
    RL = json.load(open(_rl)); pr = np.array([[RL[k]["pred"][str(T)] for T in Tc] for k in RL if k.startswith("KM-k1|") and "pred" in RL[k]])
    if len(pr): F.plot(Tc, pr.mean(0), "--", color=BLUE, lw=0.8, zorder=5)
ctx_axis(F, 32, 32, 0.0); F.set_ylim(-0.03, 1.08); F.set_ylabel(r"$\Sigma_{\rm copies}|w_j|^2$ (rel.)", labelpad=1)
for xx_, yy_, t_, c_ in ((260, 0.22, "softmax", "#7a7a7a"), (40, 0.12, "PG", GREEN), (120, 0.85, "radio head", BLUE)):
    F.text(xx_, yy_, t_, fontsize=5.5, color=c_)
style(F, "f")

# ---------------- next-token prediction (language): lab dumps if present, else bench
LABLM = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(DUMP_LM, "lab_lm_column_s*.json")))]
LABLM_PPL = [d for d in LABLM if "SM" in d and "KM" in d]
LABLM_KK = [d for d in LABLM if "KoverKc" in d]
if LABLM_PPL and LABLM_KK:
    LM = dict(LABLM_KK[0]); PPL = LABLM_PPL
    print(f"[language] lab: {len(PPL)} seed(s) with perplexities, receivers from {LM.get('pt', {}).get('KM')}")
else:
    LM = json.load(open(os.path.join(DUMP, "lm_column.json"))); PPL = [LM]
    print("[language] lab dumps missing or incomplete (need SM and KM): BENCH lm_column.json")
from matplotlib.colors import LinearSegmentedColormap
EPSC = LinearSegmentedColormap.from_list("eps", [BLUE, PINK])   # 26 Oct: slow ageing, the most locked heads (blue, as listed in a) -> fast (pink)
G_ = fig.add_subplot(gs[0, 3])                                # 25 Oct: the order parameter of every receiver, as (a)
RECV = json.load(open(os.path.join(DUMP_LM, "lab_lm_recv_s0.json")))   # dump_lm_recv.py: seed 0, positions >= 8
_e = np.array(list(RECV["eps"].values())); rng_ = np.random.default_rng(0)
for h, e in sorted(RECV["eps"].items(), key=lambda t: -t[1]):
    kk_, ws_ = np.array(RECV[h]["kk"]), np.array(RECV[h]["ws"]); s_ = rng_.choice(len(kk_), min(len(kk_), 1500), replace=False)
    ce = EPSC(np.log(e / _e.min()) / np.log(_e.max() / _e.min()))
    G_.scatter(kk_[s_], ws_[s_], s=3.5, color=ce, alpha=0.45, lw=0, rasterized=True)
xt = np.logspace(-1, 2, 200); G_.plot(xt, np.minimum(1, xt ** 2), "--", color=INK, lw=0.7, zorder=3)
G_.axvline(1, color=MUTED, lw=0.6, ls=":"); silent_side(G_); kk_axis(G_, 0.1, 100)
G_.set_ylim(-0.03, 1.08); G_.set_yticks([0, 0.5, 1]); G_.set_yticklabels(["0", "0.5", "1"])
G_.set_xlabel(r"$K/K_c$ of each receiver"); G_.set_ylabel(r"$\Sigma_j|w_j|^2$", labelpad=1); style(G_, "d")
# inset (26 Oct): learned directivity n against the ageing rate, every head of the seeds dumped (2 Nov: seeds 0-1, all data); dashed, the bound ln(1/eps)/ln(d/2)
nins = G_.inset_axes([0.56, 0.22, 0.42, 0.46])
_ne, _nn = [], []
for sd, mk in ((0, "o"), (1, "s"), (2, "^")):
    if not os.path.exists(os.path.join(DUMP_LM, f"lab_lm_recv_s{sd}.json")): continue
    _rv = json.load(open(os.path.join(DUMP_LM, f"lab_lm_recv_s{sd}.json"))); eps_, nn_ = _rv["eps"], _rv["n"]
    for h, e in eps_.items():
        _ne.append(e); _nn.append(nn_[h])
        nins.scatter(e, nn_[h], s=3.5, marker=mk, color=EPSC(np.clip(np.log(e / _e.min()) / np.log(_e.max() / _e.min()), 0, 1)), lw=0, zorder=3)
_p = np.polyfit(np.log(_ne), _nn, 1); _rc = np.corrcoef(np.log(_ne), _nn)[0, 1]        # 26 Oct: fit n = a ln(1/eps) + b, not the bound
ge = np.logspace(-2.3, 0, 50); nins.plot(ge, np.polyval(_p, np.log(ge)), "-", color=INK, lw=0.6, zorder=2)
nins.text(0.95, 0.9, f"$r={abs(_rc):.2f}$", transform=nins.transAxes, fontsize=4.8, ha="right", va="top", color=INK); print("[d] directivity fit", _p, _rc)
nins.set_xscale("log"); nins.set_xlim(5e-3, 1); nins.set_ylim(0, 3.8); nins.set_xticks([0.01, 0.1, 1]); nins.set_xticklabels([r"$10^{-2}$", r"$10^{-1}$", r"$10^0$"])
nins.set_yticks([0, 1, 2, 3]); nins.minorticks_off(); nins.tick_params(labelsize=4.5, length=1.5, pad=1)
nins.set_title(r"directivity $n$ vs $\varepsilon$", fontsize=5, pad=1); nins.patch.set_alpha(0)
for sp in ("top", "right"): nins.spines[sp].set_visible(False)
_sil = np.mean([np.mean(np.array(RECV[h]["kk"]) < 1) for h in RECV["eps"]]); print(f"[d] silent receivers {_sil:.3f}")
H_ = fig.add_subplot(gs[1, 3]); TT_ = PPL[0]["T"]
for m, c, lab in (("SM", "#7a7a7a", "softmax"), ("PG", GREEN, "PG"), ("KM", BLUE, "radio head")):
    if not any(m in d for d in PPL): continue
    v = np.array([[d[m][str(T)]["acc"] for T in TT_] for d in PPL if m in d])
    if len(v) == 1:
        H_.plot(TT_, v[0], "o-", color=c, ms=2.4, lw=1.1 if c == BLUE else 0.9, zorder=3 if c == BLUE else 2)
    else:
        range_plot(H_, TT_, v, c)
H_.set_xscale("log", base=2); H_.set_xticks(TT_); H_.set_xticklabels(map(str, TT_)); H_.minorticks_off()
H_.set_ylim(0.17, 0.6); H_.set_yticks([0.2, 0.4, 0.6])   # 27 Oct: from 0.2, as (e)
H_.axvspan(TT_[0] / 1.12, 128 * 1.12, color="#f1f0eb", zorder=0); H_.set_xlim(TT_[0] / 1.12, TT_[-1] * 1.12)
H_.set_ylabel("accuracy", labelpad=1); style(H_, "h")
_y512 = {m: np.mean([d[m]["512"]["acc"] for d in PPL]) for m in ("SM", "KM")}     # labels follow the curves
H_.text(560, _y512["SM"] + 0.05, "softmax", fontsize=5.5, color="#7a7a7a", ha="left"); H_.text(480, _y512["KM"] + 0.04, "radio head", fontsize=5.5, color=BLUE, ha="center")
_pg = [d["PG"]["512"]["acc"] for d in PPL if "PG" in d]
if _pg: H_.text(330, 0.21, "PG", fontsize=5.5, color=GREEN, ha="center")

for ax in fig.axes:
    lo, hi = ax.get_ylim()
    if ax.get_yscale() == "linear" and -0.1 < lo < 0.05 and 0.98 < hi < 1.15:
        ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0", "0.5", "1"])
fig.savefig(os.path.join(HERE, "fig2_physics.pdf"), bbox_inches="tight", pad_inches=0.01); fig.savefig(os.path.join(HERE, "fig2_physics.png"), dpi=250, bbox_inches="tight", pad_inches=0.01)
