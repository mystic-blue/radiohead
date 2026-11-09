"""Language data of the figures from the LAB weights (24 Oct 2026): the lab analogue of dump_lm_column.py (bench) and
of the scratchpad lm_profile.py, same schemas.

    dumps/lab_lm_column_s<seed>.json   T; <key>[str(T)] = {"all", "beyond128"} perplexity and {"acc", "acc_beyond128"} next-token accuracy on the concatenated
                                       validation stream cut into contexts of T tokens (T = 64 ... 1024);
                                       KoverKc[L<l>h<h>] = K/K_c of every receiver (positions >= 8, validation chunks)
                                       and eps[L<l>h<h>] (radio head only); labels[key] = the lab label
    dumps/lab_lm_profile_s<seed>.json  <key> = mean attention weight against the age, (layers, heads, 127)

<key> is "SM" for softmax and "KM" for any radio-head label (default KM-same, identical head init).  The files are
updated in place, so the labels can be dumped one at a time.  Weights: code/lab/results/lm/<label>/<arch>/<sched>/
s<seed>.pt (the validation-best state).  CPU only, 2 threads.

    python3 dump_lm_lab.py --seed 0 [--labels KM-same SM] [--arch L2-D256-h4-f0.1] [--sched lr0.001-e6-w0.02-dc0.2]
                           [--no_ppl] [--smoke]
--smoke: a tiny random lab model (d_model 32, 2 heads), a few hundred tokens; nothing written to dumps/.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
LAB = os.path.join(CODE, "lab")
sys.path[:0] = [LAB, CODE]
from layers import Cfg                                       # noqa: E402
from net import Net, Model                                   # noqa: E402

torch.set_num_threads(2)
TS = [64, 128, 256, 512, 1024]


def key_of(label):
    return "SM" if label == "SM" else ("KM" if label.startswith("KM") else label)


def load_lab(pt, pos_len=max(TS)):
    """PG has one learned anchor per position: positions beyond the trained ones get fresh anchors from the same
    initialization (as in lab/eval_long.py)."""
    s = torch.load(pt, map_location="cpu", weights_only=False)
    m = Model(Net(**s["net"]), Cfg(**{**s["cfg"], "pos_len": max(s["cfg"].get("pos_len", 0), pos_len)}))
    st = dict(s["best"])
    for k, v in list(st.items()):
        if k.endswith("attn.anchors"):
            new = m.state_dict()[k].clone(); new[:, : v.shape[1]] = v; st[k] = new
    m.load_state_dict(st); m.eval()
    return m


def perplexity(m, stream, Ts, ntok):
    res = {}
    for T in Ts:
        n = ntok // T; x = stream[: n * (T + 1)].view(n, T + 1)
        nll = torch.zeros(T); hit = torch.zeros(T); bs = max(1, 4096 // T)
        for i in range(0, n, bs):
            b = x[i: i + bs]
            logits = m(b[:, :-1]).float()
            nll += F.cross_entropy(logits.transpose(1, 2), b[:, 1:], reduction="none").sum(0)
            hit += (logits.argmax(-1) == b[:, 1:]).float().sum(0)
        nll /= n; hit /= n
        res[str(T)] = {"all": math.exp(nll.mean().item()), "beyond128": math.exp(nll[128:].mean().item()) if T > 128 else None,
                      "acc": hit.mean().item(), "acc_beyond128": hit[128:].mean().item() if T > 128 else None}
        print(f"    T {T}: {res[str(T)]}", flush=True)
    return res


def receivers(m, val, n_chunks, chunk=64):
    """K/K_c of every receiver (positions >= 8) of every head, and eps per head (radio head)."""
    m.keep(True); kk = {}
    for i in range(0, n_chunks, chunk):
        tok = val[i: i + chunk, :-1]; m(tok, key_valid=tok != 0); valid = (tok != 0)[:, 8:]
        for li, blk in enumerate(m.blocks):
            x_ = blk.last[1]["KoverKc"][:, :, 8:]
            for h in range(x_.shape[1]):
                kk.setdefault(f"L{li + 1}h{h + 1}", []).extend(x_[:, h][valid].tolist())
    m.keep(False)
    eps = {f"L{li + 1}h{h + 1}": float(blk.attn.eps()[h]) for li, blk in enumerate(m.blocks) for h in range(blk.attn.H)}
    return kk, eps


def profile(m, val, n_chunks, chunk=128):
    """Mean attention weight against the age (query - key), (layers, heads, T)."""
    m.keep(True)
    T = val.shape[1] - 1; L, H = len(m.blocks), m.blocks[0].attn.H
    prof = np.zeros((L, H, T)); cnt = np.zeros(T)
    for i in range(0, n_chunks, chunk):
        inp = val[i:i + chunk, :-1]
        m(inp, key_valid=inp != 0)
        valid = (inp != 0).numpy()
        for l, blk in enumerate(m.blocks):
            a = blk.last[0].float().numpy()
            for dist in range(T):
                diag = np.diagonal(a, offset=-dist, axis1=2, axis2=3)
                ok = valid[:, dist:] & valid[:, :T - dist]
                prof[l, :, dist] += (diag * ok[:, None]).sum((0, 2))
                if l == 0: cnt[dist] += ok.sum()
    m.keep(False)
    prof /= np.maximum(cnt, 1)
    for l in range(L):
        for h in range(H):
            pr = prof[l, h]; d = np.arange(T); sel = (d >= 8) & (d <= min(100, T - 1)) & (pr > 0)
            slope = np.polyfit(np.log(d[sel]), np.log(pr[sel]), 1)[0] if sel.sum() > 2 else float("nan")
            print(f"    L{l} h{h}: w(1) {pr[1]:.3f} w(8) {pr[min(8, T - 1)]:.4f} log-log slope 8-100 {slope:.2f}", flush=True)
    return prof


def update(path, fn):
    d = json.load(open(path)) if os.path.exists(path) else {}
    fn(d)
    with open(path + ".tmp", "w") as f:
        json.dump(d, f)
    os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--labels", "--label", nargs="+", default=["KM-same", "SM"])
    ap.add_argument("--arch", default="L2-D256-h4-f0.1")
    ap.add_argument("--sched", default="lr0.001-e6-w0.02-dc0.2")
    ap.add_argument("--ntok", type=int, default=1 << 16)
    ap.add_argument("--no_ppl", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--results", default=os.path.join(LAB, "results", "lm"))   # tests only
    ap.add_argument("--dumps", default=os.path.join(HERE, "dumps"))            # tests only
    ap.add_argument("--n_chunks", type=int, default=0)                         # tests only: fewer validation chunks
    a = ap.parse_args()
    if a.smoke:
        val = torch.randint(1, 500, (8, 65)); val[:, -5:] = 0              # padding at the end, as the real chunks
        stream = val[val != 0]
        Ts, ntok, nk, npf = [16, 64, 128], 400, 8, 8
    else:
        from oscfamily.data import tinystories
        val = tinystories._packed()["val"].long()
        stream = val[val != 0]                                              # BOS ... EOS of every story, back to back
        Ts, ntok, nk, npf = TS, a.ntok, 256, 512
        if a.n_chunks:
            nk = npf = a.n_chunks
    col = os.path.join(a.dumps, f"lab_lm_column_s{a.seed}.json")
    prf = os.path.join(a.dumps, f"lab_lm_profile_s{a.seed}.json")
    for label in a.labels:
        k = key_of(label)
        if a.smoke:
            torch.manual_seed(0)
            model = "SM" if label == "SM" else "KM"
            m = Model(Net(task="lm", vocab=500, n_out=500, d_model=32, n_heads=2, n_layers=2, d_ff=64),
                      Cfg(model=model, max_len=64, km_same=label.endswith("same"))).eval()
            pt = "(random init)"
        else:
            pt = os.path.join(a.results, label, a.arch, a.sched, f"s{a.seed}.pt")
            if not os.path.exists(pt):
                print(f"{label} s{a.seed}: no weights yet, skipped ({pt})"); continue
            m = load_lab(pt)
        t0 = time.time(); print(f"{label} s{a.seed} -> {k}  {pt}", flush=True)
        with torch.no_grad():
            ppl = None if a.no_ppl else perplexity(m, stream, Ts, ntok)
            kk = receivers(m, val, nk) if k == "KM" else None
            pr = profile(m, val, npf)
        print(f"  {time.time() - t0:.0f}s", flush=True)
        if a.smoke:
            print("  smoke ok:", {"ppl": ppl, "receivers": None if kk is None else {h: len(v) for h, v in kk[0].items()},
                                  "eps": None if kk is None else kk[1], "profile": pr.shape})
            continue

        def put_col(d, k=k, label=label, ppl=ppl, kk=kk):
            d["T"] = Ts
            d.setdefault("labels", {})[k] = label
            d.setdefault("pt", {})[k] = os.path.relpath(pt, CODE)
            if ppl is not None:
                d[k] = ppl
            if kk is not None:
                d["KoverKc"], d["eps"] = kk

        def put_prof(d, k=k, label=label, pr=pr):
            d[k] = pr.tolist(); d.setdefault("labels", {})[k] = label

        os.makedirs(os.path.dirname(col), exist_ok=True)
        update(col, put_col); update(prf, put_prof)
        print("  wrote", col, "and", prf, flush=True)


if __name__ == "__main__":
    main()
