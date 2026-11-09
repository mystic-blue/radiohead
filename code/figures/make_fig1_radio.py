"""Fig. 1 (24 Oct 2026, v13), one column, one drawing, no frames: one radio head; embeddings of d = 8.  Stations span the width; waves are as thick as
the gain of their station (what the antenna hears); the rail carries the passband (orange, up) and the attention
|w_j|^2 (bars, down), so that the matching station outside the passband is seen to lose its weight; the receiver sits
in front of the board that carries the rail, its antenna rising above the board; inside it, its state (a unit vector,
sphere) relaxes; the antenna magnified bottom left as a polarizer, the closed form beneath it.
Top: the embeddings of the first two tokens and of the new token i, with time running between them; every token is a
station (a sphere: a high-dimensional state) that carries its value at a frequency falling with age; the new token is
also a station (age 0).  Middle: the rail of ages on which the receiver slides; on it, the passband of the receiver
(orange, centred at the dial) and the gains of the stations (ticks, G_j^2).  The new token sends its dial to the
receiver and its query to the antenna, drawn magnified next to it as a polarizer: a value is a wave in two planes and the
polarizer along q passes G = |<q|v>|.  The receiver sits on the passband; its output o_i (with those of the other heads,
faded) goes down to the MLP.  Left: the closed form of the locked receiver.  No training; writes fig1_radio.{pdf,png}."""
import os, sys
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from theory import BLUE, INK, MUTED, ORANGE, plt
AGE = LinearSegmentedColormap.from_list("age", ["#2f6fc4", "#8b6fb0", "#c9503c"])  # new (blue) -> old (red)
ARR = "-|>,head_length=0.35,head_width=0.16"
col_of = lambda a: AGE(a / 7)
FS, FSS = 7.2, 6.6

# ---- toy context: ages 0 (the new token) to 7; ages 2 and 5 match the query; the dial at 2.1 ----
eps, alpha, K = 1.0, 2.1, 0.85
age = np.arange(0, 8)
G = np.sqrt(np.array([0.05, 0.06, 0.85, 0.15, 0.04, 0.8, 0.07, 0.03]))
D = alpha - eps * age
lo, hi = 1e-9, 10.0
for _ in range(200):
    r = 0.5 * (lo + hi)
    lo, hi = (r, hi) if np.sum(K**2 * G**2 / ((K * r) ** 2 + D**2)) > 1 else (lo, r)
w2 = K**2 * G**2 / ((K * r) ** 2 + D**2)
assert abs(w2.sum() - 1) < 1e-6

XL, YL = (-1.3, 13.05), (2.85, 10.75)
fig = plt.figure(figsize=(3.4, 3.4 * (YL[1] - YL[0]) / (XL[1] - XL[0])))
A = fig.add_axes([0, 0, 1, 1]); A.set_xlim(*XL); A.set_ylim(*YL); A.set_aspect("equal"); A.axis("off")


def arrow(p, q, col=INK, lw=0.6, z=6, ms=4, alpha=1.0):
    A.add_patch(FancyArrowPatch(p, q, arrowstyle=ARR, mutation_scale=ms, lw=lw, color=col, zorder=z, shrinkA=0, shrinkB=0, alpha=alpha))


def hvec(xc, y0, cmap, seed, bold=False, cw=0.32, ch=0.34, n=4, z=5, alpha=1.0):
    rg = np.random.default_rng(seed)
    for c in range(n):
        A.add_patch(Rectangle((xc - n / 2 * cw + cw * c, y0), cw, ch, fc=cmap(0.15 + 0.55 * rg.random()), ec=INK, lw=0.35, zorder=z, alpha=alpha))
    if bold:
        A.add_patch(Rectangle((xc - n / 2 * cw, y0), n * cw, ch, fc="none", ec=INK, lw=0.8, zorder=z + 1, alpha=alpha))


def sphere(x, y, R, col, z=4):
    """A shaded sphere: concentric discs shifted towards a highlight at the top left."""
    base = np.array(to_rgb(col)); n = 12                    # 25 Oct: lighter shading, thin rim
    for k in range(n):
        t = k / (n - 1)
        c = base * (1 - 0.45 * t ** 2) + 0.45 * t ** 2
        A.add_patch(Circle((x - 0.3 * R * t, y + 0.3 * R * t), R * (1 - 0.7 * t), fc=np.clip(c, 0, 1), ec="none", zorder=z + 0.01 * k))
    A.add_patch(Circle((x, y), R, fc="none", ec=base * 0.75, lw=0.35, zorder=z + 0.3))


def antenna_icon(x, y, s_=1.0, z=6):
    A.plot([x, x], [y, y + 0.55 * s_], color=INK, lw=0.8, zorder=z)
    A.plot([x, x - 0.22 * s_], [y + 0.55 * s_, y + 0.82 * s_], color=INK, lw=0.8, zorder=z)
    A.plot([x, x + 0.22 * s_], [y + 0.55 * s_, y + 0.82 * s_], color=INK, lw=0.8, zorder=z)


def dial_icon(x, y, s_=1.0, z=6):
    A.add_patch(Circle((x, y), 0.33 * s_, fc="white", ec=ORANGE, lw=0.7, zorder=z))
    for k in range(7):
        t = np.deg2rad(200 - 36.7 * k)
        A.plot([x + 0.24 * s_ * np.cos(t), x + 0.31 * s_ * np.cos(t)], [y + 0.24 * s_ * np.sin(t), y + 0.31 * s_ * np.sin(t)], color=ORANGE, lw=0.4, zorder=z + 1)
    t = np.deg2rad(115); A.plot([x, x + 0.26 * s_ * np.cos(t)], [y, y + 0.26 * s_ * np.sin(t)], color=ORANGE, lw=1.1, zorder=z + 1)


# ---- geometry ----
SP = 1.6
xa = lambda a_: SP * (7 - a_)                              # age 7 at x = 0, the new token (age 0) at 7 SP
ages8 = np.arange(7, -1, -1); xs = xa(ages8); xi = xs[-1]
YE, ys = 9.75, 8.7                                         # embeddings, stations
YR, HP, HW = 4.25, 1.0, 1.05                                # rail; height of the passband and of a unit weight
rx = xa(alpha); tx = rx                                    # the antenna on the dial line
RW, RH = 2.3, 0.9; RX0, YRec = rx - 0.05, YR + HP + 0.6   # the receiver stands on the passband, right of the dial line
ty = YRec + RH / 2 + 0.12                                  # foot of the antenna's V, on the receiver
CW, CH = 0.32, 0.34

# embeddings (d = 8) of the first two tokens and of the new one; time runs from the second to the new one
for x, sd in ((xs[0], 20), (xs[1], 21)):
    hvec(x, YE, plt.get_cmap("Greys"), sd, cw=CW, ch=CH)
hvec(xi, YE, plt.get_cmap("Greys"), 30, bold=True, cw=CW, ch=CH)
arrow((xs[1] + 0.85, YE + CH / 2), (xi - 0.85, YE + CH / 2), lw=0.5)
A.text((xs[0] + xs[1]) / 2, YE + CH + 0.08, r"embeddings $\boldsymbol{e}_j$", ha="center", va="bottom", fontsize=FSS - 0.8)   # 27 Oct
A.text(xi, YE + CH + 0.08, r"$\boldsymbol{e}_i$", ha="center", va="bottom", fontsize=FSS - 0.8)
A.text((xs[1] + xi) / 2, YE + CH / 2 + 0.12, r"time $t$", ha="center", va="bottom", fontsize=FSS)
A.text(xs[0] - 0.78, YE + CH / 2, r"$j$", ha="right", va="center", fontsize=FS); A.text(xi + 0.78, YE + CH / 2, r"$i$", ha="left", va="center", fontsize=FS)
for x in (xs[0], xs[1], xi):
    arrow((x, YE - 0.04), (x, ys + 0.55), col=BLUE, lw=0.5)
A.text(xs[0] + 0.24, (YE + ys + 0.36) / 2 - 0.08, r"$\boldsymbol{v}_j$", ha="left", va="center", fontsize=FS, color=BLUE)

# stations; each wave as thick as the gain of its station, down to the antenna
thetas = {7: 100, 6: 35, 5: 30, 4: 250, 3: 300, 2: 215, 1: 60, 0: 140}
for x, a in zip(xs, ages8):
    col = col_of(a); f = 1 - 0.11 * a
    P0, P2 = np.array([x, ys - 0.36]), np.array([tx, ty + 0.62])   # 27 Oct: a small gap above the antenna
    P1 = np.array([0.5 * (x + tx), 0.5 * (ys + ty) + 0.1])
    u_ = np.linspace(0, 1, 400)[:, None]
    c_ = (1 - u_) ** 2 * P0 + 2 * u_ * (1 - u_) * P1 + u_ ** 2 * P2
    d_ = np.gradient(c_, axis=0); sl = np.concatenate([[0], np.cumsum(np.linalg.norm(d_, axis=1))[:-1]])
    nrm = np.stack([-d_[:, 1], d_[:, 0]], 1) / np.linalg.norm(d_, axis=1)[:, None]
    amp = 0.07 * np.clip(np.sin(np.pi * sl / sl[-1]), 0, None) ** 0.5
    pts = c_ + nrm * (amp * np.sin(2 * np.pi * sl * (0.6 + 1.6 * f)))[:, None]
    g_ = G[a] ** 2 / (G ** 2).max()
    A.plot(pts[:, 0], pts[:, 1], color=col, lw=0.35 + 1.1 * g_, alpha=0.3 + 0.7 * g_, zorder=3)
    sphere(x, ys, 0.36, col)
    t = np.deg2rad(thetas[a])
    arrow((x, ys), (x + 0.3 * np.cos(t), ys + 0.3 * np.sin(t)), col=INK, lw=0.5, z=5, ms=3)
    A.text(x - 0.45, ys + 0.38, str(a), ha="center", va="center", fontsize=FSS - 0.4, color=col)

# the rail of ages: passband and gains up, attention down (behind the receiver)
xx = np.linspace(xs[0] - 0.5, xi + 0.5, 400); pb = 1 / (1 + ((alpha - eps * (7 - xx / SP)) / (K * r)) ** 2)
A.fill_between(xx, YR, YR + HP * pb, color=ORANGE, alpha=0.1, lw=0, zorder=2.2)
A.plot(xx, YR + HP * pb, color=ORANGE, lw=0.7, alpha=0.8, zorder=2.3)
A.plot([xs[0] - 0.5, xi + 0.5], [YR, YR], color=INK, lw=0.6, zorder=2.5)
for x, a in zip(xs, ages8):
    A.plot([x, x], [YR, YR + HP * G[a] ** 2], color=col_of(a), lw=1.0, zorder=2.4)
    A.plot(x, YR + HP * G[a] ** 2, "o", color=col_of(a), ms=1.6, zorder=2.4)
    A.add_patch(Rectangle((x - 0.3, YR - HW * w2[a]), 0.6, HW * w2[a], fc=col_of(a), ec="none", zorder=2.4))
A.text(xi + 0.55, YR + 0.12, r"$G_{ij}^2$", ha="left", va="bottom", fontsize=FSS)
A.text(xi + 0.55, YR - 0.12, r"$|w_{ij}|^2$", ha="left", va="top", fontsize=FSS)
A.plot([rx, rx], [YR - HW * w2.max(), YR + HP], color=ORANGE, lw=0.5, ls=(0, (1.5, 1.2)), zorder=2.3)
hw = SP * K * r / eps                                      # half width of the passband in ages: K r / eps
A.annotate("", xy=(rx + hw, YR + HP / 2), xytext=(rx, YR + HP / 2), arrowprops=dict(arrowstyle="-|>,head_length=0.3,head_width=0.12", lw=0.5, color=ORANGE, shrinkA=0, shrinkB=0), zorder=2.6)
A.text(rx + hw / 2, YR + HP / 2 + 0.06, r"$Kr$", ha="center", va="bottom", fontsize=FSS, color=ORANGE)

# the receiver, in front of the rail: the antenna on the dial line; inside, its state (a unit vector) relaxes
A.plot([tx, tx], [YRec + RH / 2, ty], color=INK, lw=0.8, zorder=6.5); antenna_icon(tx, ty, s_=0.6, z=6.5)
A.add_patch(FancyBboxPatch((RX0 + 0.1, YRec - RH / 2 - 0.1), RW, RH, boxstyle="round,pad=0.03,rounding_size=0.18",
                           fc="#000000", ec="none", alpha=0.1, zorder=5.5))
A.add_patch(FancyBboxPatch((RX0, YRec - RH / 2), RW, RH, boxstyle="round,pad=0.03,rounding_size=0.18", fc="white", ec=INK, lw=0.7, alpha=0.88, zorder=6))
SX = RX0 + 0.6
sphere(SX, YRec, 0.33, col_of(2), z=6.2)                   # the colour of the station it locks to
tt = np.linspace(0, 1, 60); ang = np.deg2rad(thetas[2] + 120 - 120 * (1 - np.exp(-3.5 * tt)) / (1 - np.exp(-3.5)))   # relaxes to the state of station 2
A.plot(SX + 0.27 * np.cos(ang), YRec + 0.27 * np.sin(ang), color="white", lw=0.5, ls=(0, (1.2, 1)), zorder=7)
arrow((SX, YRec), (SX + 0.27 * np.cos(ang[-1]), YRec + 0.27 * np.sin(ang[-1])), col=INK, lw=0.7, z=7.2)
dial_icon(RX0 + RW - 0.6, YRec, z=7)

# the antenna magnified, on the left: the wave of station 5 (a value, two planes) meets the polarizer along q_i and
# passes with amplitude G_i5 = |<q_i|v_5>|, in the plane of the query
c5 = col_of(5); PY, QX = YR + 1.75, 3.35
PX0, PX1, PA = QX - 3.25, QX - 0.55, 0.55
dv = np.array([-0.38, -0.48])
for corners, al in (([(PX0, PY + PA), (PX1, PY + PA), (PX1, PY - PA), (PX0, PY - PA)], 0.1),
                    ([(PX0 - PA * dv[0], PY - PA * dv[1]), (PX1 - PA * dv[0], PY - PA * dv[1]), (PX1 + PA * dv[0], PY + PA * dv[1]),
                      (PX0 + PA * dv[0], PY + PA * dv[1])], 0.06)):
    A.add_patch(Polygon(corners, closed=True, fc=c5, ec="none", alpha=al, zorder=2.5))
xp = np.linspace(PX0, PX1, 200); ph = 2 * np.pi * (xp - PX0) / 0.8
A.plot(xp, PY + PA * 0.95 * np.sin(ph), color=c5, lw=0.8, zorder=3)
A.plot(xp + PA * 0.75 * np.sin(ph - 1.2) * dv[0], PY + PA * 0.75 * np.sin(ph - 1.2) * dv[1], color=c5, lw=0.8, alpha=0.6, zorder=3)
A.add_patch(Circle((QX, PY), 0.45, fc="white", ec=INK, lw=0.6, zorder=5))
tq = np.deg2rad(62)
A.plot([QX - 0.38 * np.cos(tq), QX + 0.38 * np.cos(tq)], [PY - 0.38 * np.sin(tq), PY + 0.38 * np.sin(tq)], color=BLUE, lw=1.4, zorder=6)
xo = np.linspace(QX + 0.5, QX + 1.9, 80); pho = 2 * np.pi * (xo - PX0) / 0.8; amp_o = 0.42 * G[5]
A.plot(xo + amp_o * np.sin(pho) * np.cos(tq), PY + amp_o * np.sin(pho) * np.sin(tq), color=c5, lw=0.9, zorder=3)
A.text(QX, PY + 0.52, r"$\boldsymbol{q}_i$", ha="center", va="bottom", fontsize=FS, color=BLUE)
for dy_ in (0.45, -0.45):                                   # the polarizer is the antenna, magnified
    A.plot([QX + 0.05, tx - 0.05], [PY + dy_, ty + 0.25 + 0.6 * dy_], color=MUTED, lw=0.35, ls=(0, (2, 1.5)), zorder=2)
A.text((PX0 + PX1) / 2, PY + PA + 0.1, r"$\boldsymbol{v}_j$: two planes", ha="center", va="bottom", fontsize=FSS - 0.3, color=c5)
A.text(QX + 1.3, PY - 0.62, r"$G_{ij}=|\langle\boldsymbol{q}_i|\boldsymbol{v}_j\rangle|$", ha="center", va="top", fontsize=FSS - 0.2)
# the new token carries its query (blue) to the antenna and its dial (orange) to the receiver: two thin lines from the
# middle of its embedding, behind its station, then square turns
YQ = ty + 0.6; YD = YRec + RH / 2 + 0.3; XD = RX0 + RW - 0.6
for dx_, col_, yt, xt, lab in ((-0.1, BLUE, YQ, tx + 0.3, r"$\boldsymbol{q}_i$"), (0.1, ORANGE, YD, XD, r"$\alpha_i$")):
    x_ = xi + dx_
    A.plot([x_, x_], [YE - 0.02, yt], color=col_, lw=0.5, alpha=0.75, zorder=3.5)      # behind the sphere (z = 4)
    if col_ is BLUE:
        arrow((x_, yt), (xt, yt), col=col_, lw=0.5, z=6.4, alpha=0.75)
        A.text((x_ + xt) / 2 + 0.5, yt + 0.08, lab, ha="center", va="bottom", fontsize=FSS, color=col_)
    else:
        A.plot([x_, xt], [yt, yt], color=col_, lw=0.5, alpha=0.75, zorder=6.4)
        arrow((xt, yt), (xt, YRec + RH / 2 + 0.04), col=col_, lw=0.5, z=6.4, alpha=0.75)
        A.text((x_ + xt) / 2, yt + 0.08, lab, ha="center", va="bottom", fontsize=FSS, color=col_)
# output, to the right of the receiver, under the new token
hvec(xi, YRec - CH / 2, plt.get_cmap("Blues"), 40, bold=True, cw=CW, ch=CH)
arrow((RX0 + RW + 0.06, YRec), (xi - 2 * CW - 0.05, YRec))
A.text(xi, YRec - CH / 2 - 0.1, r"$\boldsymbol{o}_i$", ha="center", va="top", fontsize=FS)
fig.savefig(os.path.join(HERE, "fig1_radio.pdf")); fig.savefig(os.path.join(HERE, "fig1_radio.png"), dpi=300)
print("Kr", K * r, "weights", np.round(w2, 3), "top", ty + 0.65)
