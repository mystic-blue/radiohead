"""Mechanism data of the counting and induction columns of the experiments figure, from the LAB weights (24 Oct 2026).
Same schema as the old bench draft (dumps/ci_draft_lr1e-3.json, scratchpad probe_ci2.py), so that panels (b, c) of
make_fig2_physics.py read it unchanged:

    out[task]["Q"]   one row per answer (induction: the token after the earlier copy of the query) or per copy of the
                     query (count): T, age, w = |w_ij|^2, G (antenna gain), Delta = alpha_i - eps (i - j), mu (width),
                     K, ok (the model's answer is right), kk = K/K_c of the query, wsum = sum_j |w_ij|^2, k (copies);
                     count also one row per query: q_T, q_k, q_wsum, q_kk
    out[task]["D"]   T, ok (count, per query), L<l>_alpha_eps (dial/eps at the query, T = 32), L0_profile (mean |w|^2 of
                     layer 0 against age, T = 32), eps (per layer), layer (the layer the rows describe)

The layer is found from the data (unless --layer_<task> is given): induction, the layer with the largest median weight
on the answer; count, the layer whose total weight sum_j |w_ij|^2 correlates best with the number of copies k
(T = 32).  The receiver quantities are recomputed from the layer parameters and the layer input (the LayerNorm output
of the block, captured with a forward hook), and the recomputed weights are checked against the weights the layer
returns (max |difference| printed and stored in meta).  CPU only, 2 threads.

    python3 dump_ci_lab.py [--lr 0.001] [--seed 0] [--tasks count induction] [--out dumps/ci_lab_lr1e-3_s0.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
LAB = os.path.join(CODE, "lab")
sys.path[:0] = [LAB, os.path.join(CODE, "bench"), CODE]
from layers import Cfg, KM                                  # noqa: E402
from net import Net, Model                                  # noqa: E402
from tasks import TASKS                                     # noqa: E402
import torch.nn.functional as F                             # noqa: E402

torch.set_num_threads(2)


def run_dir(task, label, lr, d_model=64, heads=1, layers=2, epochs=80):
    return os.path.join(LAB, "results", task, label, f"L{layers}-D{d_model}-h{heads}", f"lr{lr:g}-e{epochs}-w0.02-dc0.2")


def load_lab(pt):
    s = torch.load(pt, map_location="cpu", weights_only=False)
    m = Model(Net(**s["net"]), Cfg(**s["cfg"]))
    m.load_state_dict(s["best"]); m.eval()
    return m


def recompute(at: KM, x, key_valid=None):
    """The receiver of a lab KM layer from its parameters and its input x (B,T,d_model): alpha (B,H,T), G2 (B,H,T,T,
    unmasked), Delta (B,H,T,T), mu (B,H,T), K (H), w2 = |w_ij|^2 (B,H,T,T).  Mirrors layers.KM.forward."""
    from layers import allowed, solve_mu, explicit_mu
    cfg, H = at.cfg, at.H
    B, T, _ = x.shape
    unit = lambda y: F.normalize(at.split(y).float(), dim=-1)
    v = unit(at.W_v(x))
    if cfg.km_noant:
        G2 = torch.ones(B, H, T, T)
    else:
        a = unit(at.W_q(x))
        Jv = torch.stack((-v[..., 1::2], v[..., 0::2]), -1).flatten(-2)
        ov2 = (a @ v.transpose(-1, -2)).square() + (a @ Jv.transpose(-1, -2)).square()
        rho = torch.sigmoid(at.raw_rho).view(1, H, 1, 1)
        G2 = rho + (1.0 - rho) * ov2.clamp(1e-12, 1.0).pow(at.log_n.exp().view(1, H, 1, 1))
    alpha = at.W_alpha(x).float().transpose(1, 2)
    pos = torch.arange(T, dtype=torch.float32)
    age = (pos.view(T, 1) - pos.view(1, T)).clamp_min(0.0)
    eps = at.eps()
    Delta = alpha.unsqueeze(-1) - eps.view(1, H, 1, 1) * age
    ok = allowed(T, x.device, cfg.mask_self, key_valid).to(Delta.dtype)
    K = at.log_K.exp()
    K2m = K.view(1, H, 1, 1) ** 2 * G2 * ok
    D2 = Delta.square()
    Dn = cfg.km_delta * K
    mu = explicit_mu(K2m, D2, Dn, cfg.km_explicit) if cfg.km_explicit else solve_mu(K2m, D2, Dn)
    w2 = K2m / (mu.unsqueeze(-1) ** 2 + D2)
    if cfg.km_norm:
        w2 = w2 / w2.sum(-1, keepdim=True).clamp_min(1e-12)
    return dict(alpha=alpha, G2=G2, Delta=Delta, mu=mu, K=K, eps=eps, w2=w2)


def pick_layer(task, m, xin, loader, n_batches=5):
    """The layer that does the task (see the module docstring); prints the score of each layer."""
    score = {l: [] for l in range(len(m.blocks))}
    ks, ws = [], {l: [] for l in range(len(m.blocks))}
    for i, bt in enumerate(loader):
        if i >= n_batches: break
        x, idx = bt["x"], bt["index"]
        with torch.no_grad():
            m(x, index=idx)
        B, T = x.shape; ar = torch.arange(B); q = x[ar, idx]
        ks += ((x == q[:, None]) & (torch.arange(T)[None] < idx[:, None])).sum(-1).tolist()
        for l, blk in enumerate(m.blocks):
            w = blk.last[0][ar, 0, idx]
            if task == "induction":
                score[l] += w[ar, bt["match"] + 1].tolist()
            else:
                ws[l] += w.sum(-1).tolist()
    if task == "induction":
        s = {l: float(np.median(v)) for l, v in score.items()}
        print("  layer choice (median weight on the answer):", s)
    else:
        s = {l: float(np.corrcoef(ks, ws[l])[0, 1]) if np.std(ws[l]) > 0 else -1.0 for l in ws}
        print("  layer choice (corr(total weight, k)):", s)
    return max(s, key=s.get), s


def dump_task(task, pt, layer=None):
    m = load_lab(pt)
    cfg = m.cfg
    assert cfg.model == "KM", pt
    xin = {}
    for l, b in enumerate(m.blocks):
        b.n1.register_forward_hook(lambda mod, i, o, l=l: xin.__setitem__(l, o.detach()))
    m.keep(True)
    _, _, te, extra, _ = TASKS[task]["load"](batch_size=100)
    if layer is None:
        layer, lscore = pick_layer(task, m, xin, te)
    else:
        lscore = None
    print(f"  {task}: layer {layer}", flush=True)
    D = {"T": [], "ok": [], "L0_profile": None, "layer": int(layer)}
    for l in range(len(m.blocks)):
        D[f"L{l}_alpha_eps"] = []
    Q = {k: [] for k in ("T", "age", "w", "G", "Delta", "mu", "K", "k", "wsum", "ok", "kk")}
    if task == "count":
        Q.update({k: [] for k in ("q_k", "q_wsum", "q_kk", "q_T")})
    prof = np.zeros(32); nprof = 0
    maxdiff = maxrel = 0.0
    acc = {}
    for nm, ld in {"T32": te, **extra}.items():
        T_ = int(nm[1:]); oks = []
        for i, bt in enumerate(ld):
            if i >= (20 if T_ <= 128 else 6): break
            x, idx = bt["x"], bt["index"]
            with torch.no_grad():
                lo = m(x, index=idx)
                ok = (lo.argmax(-1) == bt["y"]); oks += ok.tolist()
                B, T = x.shape; ar = torch.arange(B); q = x[ar, idx]
                age = (idx[:, None] - torch.arange(T)[None])
                for l, blk in enumerate(m.blocks):
                    a, dbg = blk.last
                    R = recompute(blk.attn, xin[l])
                    # check: the recomputed weights are the layer's weights (rows of the queries and all rows)
                    dif = (R["w2"] - a).abs()
                    maxdiff = max(maxdiff, float(dif.max()))
                    maxrel = max(maxrel, float((dif / a.abs().clamp_min(1e-6)).max()))
                    eps = float(R["eps"][0]); K = float(R["K"][0])
                    alpha = R["alpha"][ar, 0, idx]
                    if nm == "T32":
                        D[f"L{l}_alpha_eps"] += (alpha / eps).tolist() if eps > 0 else alpha.tolist()
                    if l == 0 and nm == "T32":
                        w = a[ar, 0, idx]
                        sel = (age >= 0) & (age < 32)
                        np.add.at(prof, age[sel].numpy(), w[sel].numpy())
                        nprof += B
                    if l != layer: continue
                    w = a[ar, 0, idx]; G = R["G2"][ar, 0, idx].clamp_min(0).sqrt()
                    mu = R["mu"][ar, 0, idx]
                    Delta = R["Delta"][ar, 0, idx]
                    kk = dbg["KoverKc"][ar, 0, idx]; wsum = w.sum(-1)
                    kq = ((x == q[:, None]) & (torch.arange(T)[None] < idx[:, None])).sum(-1)
                    if task == "induction":
                        bb = ar; jj = bt["match"] + 1
                    else:
                        hit = (x == q[:, None]) & (torch.arange(T)[None] < idx[:, None])
                        bb, jj = hit.nonzero(as_tuple=True)
                        D["T"] += [T_] * B; D["ok"] += ok.int().tolist()
                        Q["q_k"] += kq.tolist(); Q["q_wsum"] += wsum.tolist(); Q["q_kk"] += kk.tolist(); Q["q_T"] += [T_] * B
                    n = len(bb)
                    Q["T"] += [T_] * n; Q["age"] += age[bb, jj].tolist(); Q["w"] += w[bb, jj].tolist()
                    Q["G"] += G[bb, jj].tolist(); Q["Delta"] += Delta[bb, jj].tolist(); Q["mu"] += mu[bb].tolist()
                    Q["K"] += [K] * n; Q["ok"] += ok[bb].int().tolist(); Q["kk"] += kk[bb].tolist()
                    Q["wsum"] += wsum[bb].tolist(); Q["k"] += kq[bb].tolist()
        acc[nm] = float(np.mean(oks))
        print(f"  {task} {nm} acc {acc[nm]:.3f}   (recomputed vs layer weights: max |diff| {maxdiff:.2e}, max rel {maxrel:.2e})", flush=True)
    D["L0_profile"] = (prof / max(nprof, 1)).tolist()
    D["eps"] = [float(b.attn.eps()[0]) for b in m.blocks]
    meta = {"pt": os.path.relpath(pt, CODE), "layer_score": lscore, "acc": acc,
            "check_w2_maxabs": maxdiff, "check_w2_maxrel_floor1e-6": maxrel}
    return {"D": D, "Q": Q, "meta": meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="KM")
    ap.add_argument("--tasks", nargs="+", default=["count", "induction"])
    ap.add_argument("--layer_count", type=int, default=None)
    ap.add_argument("--layer_induction", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_path = a.out or os.path.join(HERE, "dumps", f"ci_lab_lr{a.lr:g}_s{a.seed}.json")
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for task in a.tasks:
        pt = os.path.join(run_dir(task, a.label, a.lr), f"s{a.seed}.pt")
        if not os.path.exists(pt):
            print("missing, skipped:", pt); continue
        t0 = time.time(); print(task, pt, flush=True)
        out[task] = dump_task(task, pt, getattr(a, f"layer_{task}"))
        out[task]["meta"].update(lr=a.lr, seed=a.seed, label=a.label)
        print(f"  {task} done in {time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path + ".tmp", "w") as f:
        json.dump(out, f)
    os.replace(out_path + ".tmp", out_path)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
