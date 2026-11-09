"""In-memory batching.  Every dataset is a dict of equally long CPU tensors; a batch is
a slice of them.  Shuffling uses a generator seeded per (seed, epoch), so the data
order is a function of the seed alone and a resumed run sees the same batches."""
from __future__ import annotations

import math
import random
from typing import Dict

import torch


class Loader:
    def __init__(self, tensors: Dict[str, torch.Tensor], batch_size: int, shuffle: bool, seed: int = 0):
        self.t = tensors
        self.n = len(next(iter(tensors.values())))
        self.bs, self.shuffle, self.seed, self.epoch = batch_size, shuffle, seed, 0

    def __len__(self):
        return max(1, math.ceil(self.n / self.bs))

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator().manual_seed(1_000_003 * self.seed + self.epoch)
            idx = torch.randperm(self.n, generator=g)
        else:
            idx = torch.arange(self.n)
        for i in range(0, self.n, self.bs):
            sl = idx[i:i + self.bs]
            yield {k: v[sl] for k, v in self.t.items()}


class FreshLoader(Loader):
    """A training loader that draws new examples at every epoch: make(rng) -> tensors, with
    rng seeded by (tag, seed, epoch), so the data are a function of the seed alone."""

    def __init__(self, make, batch_size: int, seed: int = 0, tag: str = ""):
        self.make, self.tag = make, tag
        super().__init__(make(random.Random(f"{tag}-{seed}-0")), batch_size, True, seed)

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self.t = self.make(random.Random(f"{self.tag}-{self.seed}-{epoch}"))
