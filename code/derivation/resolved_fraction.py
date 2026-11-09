"""27 Oct 2026: fraction of trained receivers whose tokens are resolved, eps >= K_c (and eps > K r at the trained
coupling), per layer and head, for the language models of the letter (KM-same D256 h4 f0.1 seeds 0-2: Fig. 3(d);
KM D128 h1-h8 seed 1: Fig. 2).  Receivers: every position 16..T-1 of 8 validation sequences.
Usage (from code/): python3 derivation/resolved_fraction.py"""
import os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); CODE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(CODE, "figures")); sys.path.insert(0, HERE)
import dump_lm_lab as D                                                          # noqa: E402
from oscfamily.data import tinystories                                           # noqa: E402
import theory as th                                                              # noqa: E402
torch.set_num_threads(2)
val = tinystories._packed()["val"].long()
tok = val[:8, :-1]
RUNS = [(f"KM-same/L2-D256-h4-f0.1/lr0.001-e6-w0.02-dc0.2/s{s}.pt", f"D256 h4 s{s}") for s in range(3)] + \
       [(f"KM/L2-D128-h{h}-f0.1/lr0.001-e6-w0.02-dc0.2/s1.pt", f"D128 h{h}") for h in (8, 4, 2, 1)]
for path, name in RUNS:
    m_ = D.load_lab(os.path.join(CODE, "lab/results/lm", path)); cap = {}
    for L, blk in enumerate(m_.blocks):
        blk.attn.register_forward_pre_hook(lambda mod, args, L=L: cap.__setitem__(L, args[0].detach()))
    with torch.no_grad():
        m_(tok, key_valid=tok != 0)
    for L, blk in enumerate(m_.blocks):
        att = blk.attn; x = cap[L]; B, T, _ = x.shape; H = att.H
        with torch.no_grad():
            unit = lambda y: torch.nn.functional.normalize(att.split(y).float(), dim=-1)
            v = unit(att.W_v(x)); a = unit(att.W_q(x))
            Jv = torch.stack((-v[..., 1::2], v[..., 0::2]), -1).flatten(-2)
            ov2 = (a @ v.transpose(-1, -2)).square() + (a @ Jv.transpose(-1, -2)).square()
            rho = torch.sigmoid(att.raw_rho).view(1, H, 1, 1)
            G2 = (rho + (1 - rho) * ov2.clamp(1e-12, 1).pow(att.log_n.exp().view(1, H, 1, 1))).numpy()
            alpha = att.W_alpha(x).float().transpose(1, 2).numpy()
            eps = att.eps().numpy(); K = att.log_K.exp().numpy()
        row = []
        for hh in np.argsort(eps):
            res = band = lock = n = 0
            for b in range(B):
                for i in range(16, T):
                    j = np.arange(i + 1); j = j[(tok[b, : i + 1] != 0).numpy()]
                    G2j = G2[b, hh, i, j]; Dl = alpha[b, hh, i] - eps[hh] * (i - j)
                    s1 = (G2j / Dl ** 2).sum(); Kc = 1 / np.sqrt(s1)
                    r = th.implicit(np.sqrt(G2j), Dl, K[hh]) if K[hh] > Kc else 0.0
                    n += 1; res += eps[hh] >= Kc; lock += r > 0; band += eps[hh] > K[hh] * r
            row.append(f"eps {eps[hh]:.3g}: resolved {res / n:.2f} eps>Kr {band / n:.2f} locked {lock / n:.2f}")
        print(f"{name} layer {L}: " + " | ".join(row), flush=True)
