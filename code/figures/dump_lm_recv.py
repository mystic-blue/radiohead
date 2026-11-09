"""Receivers of a trained radio head for Fig. 3(d) (2 Nov 2026; replaces the lost scratchpad lmrecv.py and the directivity
dump).  For every head: K/K_c and the total weight sum_j |w_j|^2 of every receiver (positions >= 8, the first 256
validation chunks, as dump_lm_lab.receivers), its ageing rate eps, coupling K and directivity n.

    dumps_f1/lab_lm_recv_s<seed>.json  <L<l>h<h>> = {"kk", "ws"}; eps, K, n = {<L<l>h<h>>: value}
    dumps_f1/lab_lm_n_s<seed>.json     <L<l>h<h>> = n (the schema of the old dumps/lab_lm_n_s*.json)

    python3 dump_lm_recv.py --seed 0 [--arch L2-D256-h4-f1] [--sched lr0.001-e1-w0.02-dc0.2] [--dumps dumps_f1]
CPU only, 2 threads."""
from __future__ import annotations

import argparse
import json
import os

from dump_lm_lab import LAB, load_lab, torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="KM-same")
    ap.add_argument("--arch", default="L2-D256-h4-f1")
    ap.add_argument("--sched", default="lr0.001-e1-w0.02-dc0.2")
    ap.add_argument("--n_chunks", type=int, default=256)
    ap.add_argument("--dumps", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "dumps_f1"))
    a = ap.parse_args()
    from oscfamily.data import tinystories
    val = tinystories._packed()["val"].long()
    m = load_lab(os.path.join(LAB, "results", "lm", a.label, a.arch, a.sched, f"s{a.seed}.pt"))
    out, chunk = {}, 64
    m.keep(True)
    with torch.no_grad():
        for i in range(0, a.n_chunks, chunk):
            tok = val[i: i + chunk, :-1]; m(tok, key_valid=tok != 0); valid = (tok != 0)[:, 8:]
            for li, blk in enumerate(m.blocks):
                dbg = blk.last[1]
                kk, ws = dbg["KoverKc"][:, :, 8:], dbg["wsum"][:, :, 8:]
                for h in range(kk.shape[1]):
                    d = out.setdefault(f"L{li + 1}h{h + 1}", {"kk": [], "ws": []})
                    d["kk"].extend(kk[:, h][valid].tolist()); d["ws"].extend(ws[:, h][valid].tolist())
    m.keep(False)
    heads = [(f"L{li + 1}h{h + 1}", blk.attn, h) for li, blk in enumerate(m.blocks) for h in range(blk.attn.H)]
    out["eps"] = {k: float(at.eps()[h]) for k, at, h in heads}
    out["K"] = {k: float(at.log_K.exp()[h]) for k, at, h in heads}
    out["n"] = {k: float(at.log_n.exp()[h]) for k, at, h in heads}
    os.makedirs(a.dumps, exist_ok=True)
    json.dump(out, open(os.path.join(a.dumps, f"lab_lm_recv_s{a.seed}.json"), "w"))
    json.dump(out["n"], open(os.path.join(a.dumps, f"lab_lm_n_s{a.seed}.json"), "w"))
    print(f"s{a.seed}: {len(out['eps'])} heads, {len(out['L1h1']['kk'])} receivers each; eps {out['eps']}; n {out['n']}")


if __name__ == "__main__":
    main()
