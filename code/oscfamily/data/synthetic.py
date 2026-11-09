"""The two synthetic tasks.  Each isolates one operation of the probe.

induction   (the delay) the last token repeats one earlier token; the label is the token
            that FOLLOWED that earlier copy.  A reading that lands on the match returns the
            query itself: only a reading displaced one step forward is right.  Generated as
            in Khadraoui's thesis.  Extra test sets at longer sequences measure whether the
            learned operation survives beyond the training length.
absence     (the threshold) records (key, value) and a query key.  Present: the answer is
            the record's value.  Absent: the answer is the HISTOGRAM of the values in the
            context, which is exactly what a uniform reading returns.  Keys are two
            attributes (a, b) and the embedding of a token is the sum of its field
            embeddings, so a bilinear score is additive in the attributes: a record that
            shares one attribute with an absent query always scores above one that shares
            none, and a softmax row cannot go flat on it.  Every absent query shares at
            least one attribute with some record.  Each record is ONE token, so no delay is
            needed anywhere in this task.
fallback    (v8, the delay and the threshold) induction in which a fraction of the queries has no earlier
            copy; then the answer is the most frequent token of the context (unique in every sequence,
            present or absent), which only a flat reading returns.
abstention  (v8, the threshold) as absence, but an absent key has its own answer, a null class:
            the model must say that the key is not there.  T records, each ONE token (a, b, v);
            present: the label is v; absent: the query changes one attribute of a record, so it
            always shares the other one with it (a near miss), and the label is the null class.
            majority = 1 (the variant used, 25 Sep 2026): an absent key is answered with the most
            frequent value of the context (unique in every sequence), which only a flat reading
            returns: a softmax row always favours the near miss.
"""
from __future__ import annotations

import random
import torch

from .common import Loader, FreshLoader


def _induction(n, vocab, T, rng, offset=1):
    """offset: the answer is the token `offset` positions after the earlier copy of the query
    (1: the standard task; negative: before it)."""
    X, Y = [], []
    lo, hi = max(1, -offset), T - 2 - max(offset, 0)       # the answer stays inside the context
    for _ in range(n):
        q = rng.randrange(1, 1 + vocab)
        j = rng.randrange(lo, hi + 1) if offset != 1 else rng.randrange(1, T - 2)
        seq = []
        for _t in range(T - 1):
            r = rng.randrange(1, 1 + vocab)
            while r == q:
                r = rng.randrange(1, 1 + vocab)
            seq.append(r)
        seq[j] = q
        lab = seq[j + offset]
        seq.append(q)
        X.append(seq); Y.append(lab - 1)
    X = torch.tensor(X, dtype=torch.long)
    match = (X[:, :-1] == X[:, -1:]).float().argmax(-1)
    return {"x": X, "y": torch.tensor(Y), "index": torch.full((n,), T - 1, dtype=torch.long), "match": match}


def _silence(n, vocab, T, absent, rng, offset=1):
    """Induction with silence (1 Oct 2026, final): with probability 1 - absent the last token repeats exactly one
    earlier token and the label is the token that followed it (A B ... A -> B); with probability absent the last
    token has not appeared before and the label is the null class (index vocab): nothing to read.
    offset k (1 Oct 2026): the label is the token k positions after the earlier copy (k = 1: the task above, with
    the same draws; negative: before it); the answer always lies inside the context."""
    X, Y, P = [], [], []
    for _ in range(n):
        q = rng.randrange(1, 1 + vocab)
        seq = []
        for _t in range(T - 1):
            r = rng.randrange(1, 1 + vocab)
            while r == q:
                r = rng.randrange(1, 1 + vocab)
            seq.append(r)
        present = rng.random() >= absent
        if present:
            j = rng.randrange(1, T - 2) if offset == 1 else rng.randrange(max(1, -offset), T - 1 - max(offset, 0))
            seq[j] = q
            lab = seq[j + offset] - 1
        else:
            lab = vocab
        seq.append(q)
        X.append(seq); Y.append(lab); P.append(present)
    X = torch.tensor(X, dtype=torch.long)
    match = (X[:, :-1] == X[:, -1:]).float().argmax(-1)
    return {"x": X, "y": torch.tensor(Y), "index": torch.full((n,), T - 1, dtype=torch.long), "match": match,
            "present": torch.tensor(P), "null": torch.full((n,), vocab, dtype=torch.long)}


def silence(n_train=20000, n_val=2000, n_test=2000, vocab=50, T=32, absent=0.2, batch_size=128, seed=0,
            data_seed=0, fresh=0, offset=1):
    """Induction with silence: validation and test sets fixed by data_seed; fresh = 1: new training sequences at
    every epoch; offset: the answer k positions from the earlier copy."""
    rng = random.Random(data_seed)
    T, absent, offset = int(T), float(absent), int(offset)
    make = lambda n, r: _silence(n, vocab, T, absent, r, offset)
    tr, va, te = (make(n, rng) for n in (n_train, n_val, n_test))
    meta = {"vocab_size": vocab + 1, "num_classes": vocab + 1, "max_seq_len": T}
    train = (FreshLoader(lambda r: make(n_train, r), batch_size, seed,
                         tag=f"silence-{data_seed}-T{T}-a{absent}" + ("" if offset == 1 else f"-k{offset}"))
             if fresh else Loader(tr, batch_size, True, seed))
    return train, Loader(va, batch_size, False), Loader(te, batch_size, False), meta


def _distance(n, vocab, T, absent, rng, dmax=None):
    """Distance with silence (4 Oct 2026): the draws of _silence (offset 1); present: the label is the DISTANCE from the
    query back to its earlier copy, T - 1 - j in [2, T - 2] (the token read is the query itself: the content of the
    reading says nothing, only where it was read); absent: the null class 0.  Without a positional encoding a reading
    that returns only the value read cannot answer; one whose output carries the distance can (values turned by it)."""
    X, Y, P, M = [], [], [], []
    for _ in range(n):
        q = rng.randrange(1, 1 + vocab)
        seq = []
        for _t in range(T - 1):
            r = rng.randrange(1, 1 + vocab)
            while r == q:
                r = rng.randrange(1, 1 + vocab)
            seq.append(r)
        present = rng.random() >= absent
        j = rng.randrange(1 if dmax is None else max(1, T - 1 - dmax), T - 2)   # dmax: the copy within dmax tokens
        if present:
            seq[j] = q
        seq.append(q)
        X.append(seq); Y.append(T - 1 - j if present else 0); P.append(present); M.append(j if present else 0)
    return {"x": torch.tensor(X, dtype=torch.long), "y": torch.tensor(Y), "index": torch.full((n,), T - 1, dtype=torch.long),
            "match": torch.tensor(M), "present": torch.tensor(P), "null": torch.zeros(n, dtype=torch.long)}


def distance(n_train=20000, n_val=2000, n_test=2000, vocab=50, T=32, absent=0.2, batch_size=128, seed=0,
             data_seed=0, fresh=0):
    """Distance to the earlier copy of the query, with a null class when there is none; classes 0 (null), 2..T-2."""
    rng = random.Random(data_seed)
    T, absent = int(T), float(absent)
    make = lambda n, r: _distance(n, vocab, T, absent, r)
    tr, va, te = (make(n, rng) for n in (n_train, n_val, n_test))
    meta = {"vocab_size": vocab + 1, "num_classes": T - 1, "max_seq_len": T}
    long_rng = random.Random(10_000 + data_seed)      # 4 Oct 2026: longer contexts, the copy within the trained range
    meta["_extra"] = {f"T{L}": Loader(_distance(n_test, vocab, L, absent, long_rng, dmax=T - 2), batch_size, False)
                      for L in (64, 128)}
    train = (FreshLoader(lambda r: make(n_train, r), batch_size, seed, tag=f"distance-{data_seed}-T{T}-a{absent}")
             if fresh else Loader(tr, batch_size, True, seed))
    return train, Loader(va, batch_size, False), Loader(te, batch_size, False), meta


def _count(n, vocab, T, kmax, rng):
    """Count (4 Oct 2026): the last token is the query q; it appears k times before it, k uniform in 0..kmax, at
    distinct random positions; the label is k.  A normalized reading of k identical values is the same for every
    k >= 1: only the weight of the copies against the background, or the dephasing of the turned values (copies at
    different distances), tells k."""
    X, Y, P, M = [], [], [], []
    for _ in range(n):
        q = rng.randrange(1, 1 + vocab)
        seq = []
        for _t in range(T - 1):
            r = rng.randrange(1, 1 + vocab)
            while r == q:
                r = rng.randrange(1, 1 + vocab)
            seq.append(r)
        k = rng.randrange(0, kmax + 1)
        pos = sorted(rng.sample(range(T - 1), k))
        for j in pos:
            seq[j] = q
        seq.append(q)
        X.append(seq); Y.append(k); P.append(k > 0); M.append(pos[-1] if k else 0)
    return {"x": torch.tensor(X, dtype=torch.long), "y": torch.tensor(Y), "index": torch.full((n,), T - 1, dtype=torch.long),
            "match": torch.tensor(M), "present": torch.tensor(P), "null": torch.zeros(n, dtype=torch.long)}


def count(n_train=20000, n_val=2000, n_test=2000, vocab=50, T=32, kmax=3, batch_size=128, seed=0, data_seed=0, fresh=0):
    """How many earlier copies of the query (0..kmax); extra tests at longer contexts (64, 128)."""
    rng = random.Random(data_seed)
    T, kmax = int(T), int(kmax)
    make = lambda n, r: _count(n, vocab, T, kmax, r)
    tr, va, te = (make(n, rng) for n in (n_train, n_val, n_test))
    meta = {"vocab_size": vocab + 1, "num_classes": kmax + 1, "max_seq_len": T}
    long_rng = random.Random(10_000 + data_seed)
    meta["_extra"] = {f"T{L}": Loader(_count(n_test, vocab, L, kmax, long_rng), batch_size, False) for L in (64, 128, 256, 512)}
    train = (FreshLoader(lambda r: make(n_train, r), batch_size, seed, tag=f"count-{data_seed}-T{T}-k{kmax}")
             if fresh else Loader(tr, batch_size, True, seed))
    return train, Loader(va, batch_size, False), Loader(te, batch_size, False), meta


def _lookback(n, vocab, T, k, rng):
    """Lookback (4 Oct 2026): random tokens; the label at the last position is the token k positions back.  Pure
    relative position: content says nothing about where to read."""
    X = torch.tensor([[rng.randrange(1, 1 + vocab) for _ in range(T)] for _ in range(n)], dtype=torch.long)
    Y = X[:, T - 1 - k] - 1
    return {"x": X, "y": Y, "index": torch.full((n,), T - 1, dtype=torch.long)}


def lookback(n_train=20000, n_val=2000, n_test=2000, vocab=50, T=32, k=1, batch_size=128, seed=0, data_seed=0, fresh=0):
    """Read the token k positions back; extra tests at longer contexts (64, 128)."""
    rng = random.Random(data_seed)
    T, k = int(T), int(k)
    make = lambda n, r: _lookback(n, vocab, T, k, r)
    tr, va, te = (make(n, rng) for n in (n_train, n_val, n_test))
    meta = {"vocab_size": vocab + 1, "num_classes": vocab, "max_seq_len": T}
    long_rng = random.Random(10_000 + data_seed)
    meta["_extra"] = {f"T{L}": Loader(_lookback(n_test, vocab, L, k, long_rng), batch_size, False) for L in (64, 128)}
    train = (FreshLoader(lambda r: make(n_train, r), batch_size, seed, tag=f"lookback-{data_seed}-T{T}-k{k}")
             if fresh else Loader(tr, batch_size, True, seed))
    return train, Loader(va, batch_size, False), Loader(te, batch_size, False), meta


INDUCTION_TEST_LENGTHS = (64, 128, 256)


def induction(n_train=20000, n_val=2000, n_test=2000, vocab=50, T=32, batch_size=128, seed=0, data_seed=0,
              n_long=2000, fresh=0, offset=1):
    """fresh = 1: new training sequences at every epoch (n_train per epoch); validation and
    test sets are fixed by data_seed."""
    rng = random.Random(data_seed)
    offset = int(offset)
    tr, va, te = (_induction(n, vocab, T, rng, offset) for n in (n_train, n_val, n_test))
    meta = {"vocab_size": vocab + 1, "num_classes": vocab, "max_seq_len": T}
    long_rng = random.Random(10_000 + data_seed)
    meta["_extra"] = {f"T{L}": Loader(_induction(n_long, vocab, L, long_rng, offset), batch_size, False)
                      for L in INDUCTION_TEST_LENGTHS}
    train = (FreshLoader(lambda r: _induction(n_train, vocab, T, r, offset), batch_size, seed,
                         tag=f"induction-{data_seed}" + ("" if offset == 1 else f"-k{offset}"))
             if fresh else Loader(tr, batch_size, True, seed))
    return train, Loader(va, batch_size, False), Loader(te, batch_size, False), meta


def _absence(n, A, Bn, V, R, absent_frac, rng):
    X = torch.zeros(n, R + 1, 3, dtype=torch.long)
    Y = torch.zeros(n, V)
    P = torch.zeros(n, dtype=torch.bool)
    keys = [(a, b) for a in range(1, A + 1) for b in range(1, Bn + 1)]
    for i in range(n):
        ks = rng.sample(keys, R)
        vals = [rng.randrange(1, V + 1) for _ in range(R)]
        for r, ((a, b), v) in enumerate(zip(ks, vals)):
            X[i, r] = torch.tensor([a, b, v])
        if rng.random() >= absent_frac:
            j = rng.randrange(R)
            q = ks[j]
            Y[i, vals[j] - 1] = 1.0
            P[i] = True
        else:
            inside = set(ks)
            while True:
                q = rng.choice(keys)
                if q not in inside and any(q[0] == a or q[1] == b for a, b in ks):
                    break
            for v in vals:
                Y[i, v - 1] += 1.0 / R
        X[i, R] = torch.tensor([q[0], q[1], V + 1])        # the query: its key and a query marker
    return {"x": X, "y": Y, "index": torch.full((n,), R, dtype=torch.long), "present": P}


def absence(n_train=20000, n_val=4000, n_test=8000, n_a=8, n_b=8, n_vals=8, records=12, absent_frac=0.5,
            batch_size=128, seed=0, data_seed=0):
    rng = random.Random(data_seed)
    tr, va, te = (_absence(n, n_a, n_b, n_vals, records, absent_frac, rng) for n in (n_train, n_val, n_test))
    meta = {"field_sizes": (n_a + 1, n_b + 1, n_vals + 2), "num_classes": n_vals, "max_seq_len": records + 1}
    return Loader(tr, batch_size, True, seed), Loader(va, batch_size, False), Loader(te, batch_size, False), meta




def _values_unique_mode(T, V, rng):
    while True:
        vals = [rng.randrange(1, V + 1) for _ in range(T)]
        c = sorted((vals.count(v) for v in set(vals)), reverse=True)
        if len(c) == 1 or c[0] > c[1]:
            return vals


def _abstention(n, A, Bn, V, T, present_frac, rng, majority=0):
    X = torch.zeros(n, T + 1, 3, dtype=torch.long)
    Y = torch.zeros(n, dtype=torch.long)
    P = torch.zeros(n, dtype=torch.bool)
    M = torch.zeros(n, dtype=torch.long)          # the record the query matches (present) or nearly (absent)
    keys = [(a, b) for a in range(1, A + 1) for b in range(1, Bn + 1)]
    for i in range(n):
        ks = rng.sample(keys, T)
        vals = _values_unique_mode(T, V, rng) if majority else [rng.randrange(1, V + 1) for _ in range(T)]
        for r, ((a, b), v) in enumerate(zip(ks, vals)):
            X[i, r] = torch.tensor([a, b, v])
        j = rng.randrange(T)
        if rng.random() < present_frac:
            q, Y[i], P[i] = ks[j], vals[j] - 1, True
        else:
            inside = set(ks)
            while True:                            # a near miss of record j: one attribute changed
                a, b = ks[j]
                q = (rng.randrange(1, A + 1), b) if rng.random() < 0.5 else (a, rng.randrange(1, Bn + 1))
                if q not in inside:
                    break
            Y[i] = max(set(vals), key=vals.count) - 1 if majority else V   # the majority value, or the null class
        M[i] = j
        X[i, T] = torch.tensor([q[0], q[1], V + 1])   # the query: its key and a query marker
    out = {"x": X, "y": Y, "index": torch.full((n,), T, dtype=torch.long), "present": P, "match": M}
    if majority:
        out["majority"] = torch.ones(n, dtype=torch.bool)      # tells the evaluation there is no null class
    return out


def abstention(n_train=20000, n_val=2000, n_test=2000, n_a=8, n_b=8, n_vals=8, T=16, present_frac=0.5,
               batch_size=128, seed=0, data_seed=0, fresh=0, majority=0):
    """fresh = 1: new training sequences at every epoch; validation and test fixed by data_seed.
    majority = 1: absent keys are answered with the majority value (no null class)."""
    rng = random.Random(data_seed)
    majority = int(majority)
    make = lambda n, r: _abstention(n, n_a, n_b, n_vals, int(T), present_frac, r, majority)
    tr, va, te = (make(n, rng) for n in (n_train, n_val, n_test))
    meta = {"field_sizes": (n_a + 1, n_b + 1, n_vals + 2), "num_classes": n_vals + (0 if majority else 1),
            "max_seq_len": int(T) + 1}
    train = (FreshLoader(lambda r: make(n_train, r), batch_size, seed,
                         tag=f"abstention-{data_seed}-T{int(T)}" + ("-maj" if majority else ""))
             if fresh else Loader(tr, batch_size, True, seed))
    return train, Loader(va, batch_size, False), Loader(te, batch_size, False), meta


def _fallback(n, vocab, T, absent_frac, rng):
    X, Y, P, M = [], [], [], []
    for _ in range(n):
        q = rng.randrange(1, 1 + vocab)
        present = rng.random() >= absent_frac
        while True:
            ctx = []
            for _t in range(T - 1):
                r = rng.randrange(1, 1 + vocab)
                while r == q:
                    r = rng.randrange(1, 1 + vocab)
                ctx.append(r)
            j = rng.randrange(1, T - 2)
            if present:
                ctx[j] = q
            c = sorted(((ctx.count(v), v) for v in set(ctx)), reverse=True)
            if c[0][0] > c[1][0]:                  # a unique most frequent token
                break
        lab = ctx[j + 1] if present else c[0][1]
        X.append(ctx + [q]); Y.append(lab - 1); P.append(present); M.append(j if present else 0)
    return {"x": torch.tensor(X, dtype=torch.long), "y": torch.tensor(Y), "index": torch.full((n,), T - 1, dtype=torch.long),
            "present": torch.tensor(P), "match": torch.tensor(M), "majority": torch.ones(n, dtype=torch.bool)}


def fallback(n_train=20000, n_val=2000, n_test=2000, vocab=50, T=32, absent=0.5, batch_size=128, seed=0,
             data_seed=0, fresh=0):
    rng = random.Random(data_seed)
    T = int(T)
    make = lambda n, r: _fallback(n, vocab, T, float(absent), r)
    tr, va, te = (make(n, rng) for n in (n_train, n_val, n_test))
    meta = {"vocab_size": vocab + 1, "num_classes": vocab, "max_seq_len": T}
    train = (FreshLoader(lambda r: make(n_train, r), batch_size, seed, tag=f"fallback-{data_seed}-T{T}-a{absent}")
             if fresh else Loader(tr, batch_size, True, seed))
    return train, Loader(va, batch_size, False), Loader(te, batch_size, False), meta
