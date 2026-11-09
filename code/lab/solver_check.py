"""How many bisection iterations does the implicit width need?  The inputs (K2m, D2, D) of every KM layer are captured
on real batches of trained models (language: the d_model 256 implicit radio head; MQAR: a trained probe), the width is
solved in float64 with 80 iterations as reference, and the float32 solver with n iterations (+ the Newton step) is
compared with it: relative error of mu, of the output of the layer's read (sum |w|^2), and of the model's logits.
    python3 solver_check.py"""
from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import layers as L                                                         # noqa: E402
from layers import Cfg                                                     # noqa: E402
from net import Net, Model                                                 # noqa: E402

BANC = os.path.join(os.path.dirname(HERE), "bench")
sys.path.insert(0, BANC)
CASES = {
    "language d256": ("results/lm/L2/KM-full-D256-h4-ihead-pow-ar1e-08-delta0.001/lr0.001-f0.1-st20p100-w0.02-dc0.2/s0.pt", "lm"),
    "MQAR d64": ("results/mqar/L2/KM-full-D64-ihead-pow-ar1e-08-delta0.001/lr0.001-st100p20/s0.pt", "mqar"),
}
ITERS = (6, 8, 10, 12, 14, 16, 20, 24)


def build(pt):
    s = torch.load(os.path.join(BANC, pt), map_location="cpu", weights_only=False)
    n, c = s["net"], s["cfg"]
    net = Net(task=n["task"], vocab=n["vocab"], n_out=n["n_out"], d_model=n["d_model"], n_heads=n["n_heads"],
              n_layers=n["n_layers"], d_ff=n["d_ff"], dropout=0.0, embed_scale=n["embed_scale"], init_std=0.0,
              padding_idx=n["padding_idx"], pe=n["pe"])
    cfg = Cfg(model="KM", mask_self=c["mask_self"], max_len=c["max_len"], km_delta=c["km_delta"], km_rho=c["an_rho"])
    m = Model(net, cfg)
    m.load_state_dict(s["state"], strict=True)
    return m.eval()


def batch(kind):
    if kind == "lm":
        from tasks import lm
        _, va, _, _, _ = lm(frac=0.001, batch_size=32, max_eval=64)
        tok = next(iter(va))["tokens"].long()[:, :-1]
        return tok, tok != 0
    from tasks import mqar
    _, va, _, _, _ = mqar(batch_size=64)
    return next(iter(va))["x"], None


torch.manual_seed(0)
for name, (pt, kind) in CASES.items():
    m = build(pt)
    tok, kv = batch(kind)
    caught, orig = [], L.solve_mu
    L.solve_mu = lambda K2m, D2, Dn, iters=24: (caught.append((K2m.detach(), D2.detach(), Dn.detach())), orig(K2m, D2, Dn, iters))[1]
    with torch.no_grad():
        ref_logits = m(tok, key_valid=kv)
    print(f"\n{name}: {len(caught)} layers, {caught[0][0].shape[0] * caught[0][0].shape[1] * caught[0][0].shape[2]:,} queries per layer")
    rows = {n: [] for n in ITERS}
    for K2m, D2, Dn in caught:
        mu_ref = orig(K2m.double(), D2.double(), Dn.double(), 80)
        ws_ref = (K2m.double() / (mu_ref.unsqueeze(-1) ** 2 + D2.double())).sum(-1)
        for n in ITERS:
            mu = orig(K2m, D2, Dn, n).double()
            ws = (K2m.double() / (mu.unsqueeze(-1) ** 2 + D2.double())).sum(-1)
            rows[n].append(((mu - mu_ref).abs() / mu_ref).flatten())
            rows[n][-1] = (rows[n][-1], ((ws - ws_ref).abs()).flatten())
    for n in ITERS:
        L.solve_mu = lambda K2m, D2, Dn, iters=24, n=n: orig(K2m, D2, Dn, n)
        with torch.no_grad():
            lg = m(tok, key_valid=kv)
        dl = float((lg - ref_logits).abs().max())
        e_mu = torch.cat([r[0] for r in rows[n]]); e_ws = torch.cat([r[1] for r in rows[n]])
        print(f"  iters {n:2d}: mu rel err median {float(e_mu.median()):.1e} max {float(e_mu.max()):.1e} | "
              f"sum|w|^2 abs err max {float(e_ws.max()):.1e} | logits max change vs 24 iters {dl:.1e}")
    L.solve_mu = orig
