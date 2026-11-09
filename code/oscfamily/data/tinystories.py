"""TinyStories in the tokenisation of Pasqualetti & Guo (word level, vocabulary 8004,
chunks of at most 128 tokens with BOS/EOS, padding id 0).

Validation is THEIR shipped split (data/tinystories_eval), so perplexities are on the
same footing as the published ones.  That split is used for model selection and
reporting exactly as in their protocol ('val'); in addition a TEST split is carved
here from the END of the training chunks (never trained on), so that anything selected
on validation (readout exponent, epoch) can be reported on data it was not selected on.

The 4.7 M training chunks are packed once into an int16 tensor (about 1.2 GB) and
cached.  `frac` < 1 takes a fixed, seed-independent subset of the training chunks
(every k-th chunk), which is what the sweeps use; frac = 1 is the published protocol.
"""
from __future__ import annotations

import os
import pickle

import torch

from .. import paths
from .common import Loader

MAX_LEN = 128
N_TEST = 40_000            # chunks held out from the tail of the training file


def _pack(chunks):
    n = len(chunks)
    out = torch.zeros(n, MAX_LEN, dtype=torch.int16)
    for i, c in enumerate(chunks):
        out[i, : len(c)] = torch.tensor(c, dtype=torch.int16)
    return out


def _packed():
    os.makedirs(paths.CACHE, exist_ok=True)
    f = os.path.join(paths.CACHE, "tinystories_packed.pt")
    if os.path.exists(f):
        return torch.load(f, weights_only=False)
    print("[tinystories] packing the training chunks (once, ~2 min)...", flush=True)
    with open(paths.TS_TRAIN, "rb") as fh:
        train = pickle.load(fh)
    with open(os.path.join(paths.TS_EVAL, "val_chunks.pkl"), "rb") as fh:
        val = pickle.load(fh)
    vd = torch.load(os.path.join(paths.TS_EVAL, "vocab.pt"), weights_only=False, map_location="cpu")
    blob = {"train": _pack(train[:-N_TEST]), "test": _pack(train[-N_TEST:]), "val": _pack(val),
            "vocab_size": len(vd["vocab"])}
    torch.save(blob, f)
    return blob


def load(frac: float = 1.0, batch_size: int = 256, seed: int = 0, smoke: bool = False, max_eval: int = 0):
    """max_eval > 0: validation and test cut to their first max_eval chunks (rehearsals only)."""
    b = _packed()
    train = b["train"]
    if frac < 1.0:
        step = int(round(1.0 / frac))
        train = train[::step]
    val, test = b["val"], b["test"]
    if max_eval:
        val, test = val[:max_eval], test[:max_eval]
    if smoke:
        train, val, test = train[:2048], val[:512], test[:512]
    meta = {"vocab_size": b["vocab_size"], "max_seq_len": MAX_LEN, "n_train": len(train),
            "n_val": len(val), "n_test": len(test), "frac": frac}
    mk = lambda t, sh: Loader({"tokens": t}, batch_size, sh, seed)
    return mk(train, True), mk(val, False), mk(test, False), meta
