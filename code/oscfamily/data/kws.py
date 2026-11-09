"""Keyword spotting: Google Speech Commands v0.02, 40 log-mel bins x 49 frames.

`load`    the ten core words, from the feature cache built by the thesis code.
"""
from __future__ import annotations

import os

import torch

from .. import paths
from .common import Loader


def load(batch_size=64, seed=0, smoke=False):
    if not os.path.exists(paths.KWS_CACHE):
        raise FileNotFoundError(f"{paths.KWS_CACHE}: build it with codes.data.load_kws in the thesis repository")
    b = torch.load(paths.KWS_CACHE, weights_only=False)
    tr, va, te = ({"x": b["X" + s], "y": b["y" + s]} for s in ("tr", "va", "te"))
    if smoke:
        tr, va, te = ({k: v[:1024] for k, v in d.items()} for d in (tr, va, te))
    meta = {"n_feats": 40, "num_classes": 10, "max_seq_len": 49}
    return Loader(tr, batch_size, True, seed), Loader(va, batch_size, False), Loader(te, batch_size, False), meta
