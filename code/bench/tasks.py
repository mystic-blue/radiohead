"""Les tasques de la síntesi.  Les dades de les etapes 2-3 es generen amb el seu codi (mateixes llavors de dades);
MQAR-S és nova.

    lm         TinyStories (vocabulari i validació de Pasqualetti & Guo), el criteri final
    mqar       record associatiu amb moltes consultes i silenci: parells (clau, valor) i després consultes; una consulta
               present demana el valor de la seva clau, una absent demana la classe nul·la.  És la primitiva que separa
               l'atenció softmax de les alternatives (Zoology, Arora et al. 2023) i hi afegeix l'abstenció.  Proves a
               2x, 4x i 8x parells.
    count      quantes còpies anteriors de la consulta (0..3): el que una lectura normalitzada no pot dir
    induction  el protocol de la tesi (A B ... A -> B), per replicar el resultat d'M2 a l'etapa 2
"""
from __future__ import annotations

import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))         # code/, for the oscfamily data loaders
from oscfamily.data import synthetic, tinystories          # noqa: E402
from oscfamily.data.common import Loader, FreshLoader      # noqa: E402

KEYS, VALS = 512, 64


def _mqar(n, n_pairs, n_q, absent, rng):
    """Fitxes: claus 1..KEYS, valors KEYS+1..KEYS+VALS.  y = -100 fora de les consultes; valor 0..VALS-1; nul = VALS."""
    T = 2 * n_pairs + n_q
    X = torch.zeros(n, T, dtype=torch.long)
    Y = torch.full((n, T), -100, dtype=torch.long)
    P = torch.zeros(n, T, dtype=torch.bool)
    keys = list(range(1, KEYS + 1))
    for i in range(n):
        ks = rng.sample(keys, n_pairs + n_q)
        ctx, outside = ks[:n_pairs], ks[n_pairs:]
        vals = [rng.randrange(VALS) for _ in range(n_pairs)]
        seq = []
        for k, v in zip(ctx, vals):
            seq += [k, KEYS + 1 + v]
        order = rng.sample(range(n_pairs), min(n_q, n_pairs))
        for qi in range(n_q):
            t = 2 * n_pairs + qi
            if rng.random() >= absent and qi < len(order):
                j = order[qi]
                seq.append(ctx[j]); Y[i, t] = vals[j]; P[i, t] = True
            else:
                seq.append(outside[qi]); Y[i, t] = VALS
        X[i] = torch.tensor(seq)
    return {"x": X, "y": Y, "present": P}


def mqar(n_pairs=16, n_q=16, absent=0.2, n_train=20000, n_val=2000, n_test=2000, batch_size=128, seed=0, data_seed=0):
    rng = random.Random(data_seed)
    make = lambda n, r, npairs=n_pairs: _mqar(n, npairs, n_q, absent, r)
    va, te = make(n_val, rng), make(n_test, rng)
    long_rng = random.Random(10_000 + data_seed)
    extra = {f"P{k * n_pairs}": Loader(make(n_test // 2, long_rng, k * n_pairs), max(8, batch_size // k), False)
             for k in (2, 4, 8) if k * n_pairs + n_q <= KEYS}   # no més claus de les que hi ha
    train = FreshLoader(lambda r: make(n_train, r), batch_size, seed, tag=f"mqar-{data_seed}-{n_pairs}-{n_q}-{absent}")
    meta = {"vocab": KEYS + VALS + 1, "n_out": VALS + 1, "max_len": 2 * n_pairs + n_q, "kind": "seq"}
    return train, Loader(va, batch_size, False), Loader(te, batch_size, False), extra, meta


def count(batch_size=128, seed=0, data_seed=0, kmax=3):
    tr, va, te, meta = synthetic.count(batch_size=batch_size, seed=seed, data_seed=data_seed, fresh=1, kmax=kmax)
    extra = meta.pop("_extra")
    return tr, va, te, extra, {"vocab": meta["vocab_size"], "n_out": meta["num_classes"],
                               "max_len": meta["max_seq_len"], "kind": "cls"}


def induction(batch_size=128, seed=0, data_seed=0):
    tr, va, te, meta = synthetic.induction(batch_size=batch_size, seed=seed, data_seed=data_seed, fresh=1)   # 11 oct: dades noves cada època, com mqar i count
    extra = meta.pop("_extra")
    long_rng = random.Random(20_000 + data_seed)
    for L in (512, 1024):
        extra[f"T{L}"] = Loader(synthetic._induction(500, 50, L, long_rng), 25, False)
    return tr, va, te, extra, {"vocab": meta["vocab_size"], "n_out": meta["num_classes"],
                               "max_len": meta["max_seq_len"], "kind": "cls"}


def lm(frac=0.02, batch_size=256, seed=0, max_eval=0):
    tr, va, te, meta = tinystories.load(frac=frac, batch_size=batch_size, seed=seed, max_eval=max_eval)
    return tr, va, te, {}, {"vocab": meta["vocab_size"], "n_out": meta["vocab_size"], "max_len": 128, "kind": "lm",
                            "n_train": meta["n_train"]}


PROBE = dict(d_model=64, n_heads=1, d_ff=128, dropout=0.0, embed_scale=True, init_std=0.02, padding_idx=None)
LMNET = dict(d_model=128, n_heads=4, d_ff=512, dropout=0.1, embed_scale=False, init_std=0.0, padding_idx=0)

TASKS = {
    "lm":        dict(load=lm,        net=LMNET, epochs=6,  lr=5e-3, batch=256, mask_self=False, warmup=0.02, rs_blocks=4),
    "mqar":      dict(load=mqar,      net=PROBE, epochs=30, lr=1e-3, batch=128, mask_self=True,  warmup=0.0),
    "count":     dict(load=count,     net=PROBE, epochs=60, lr=1e-3, batch=128, mask_self=True,  warmup=0.0),
    "induction": dict(load=induction, net=PROBE, epochs=30, lr=1e-3, batch=128, mask_self=True,  warmup=0.0),
}
