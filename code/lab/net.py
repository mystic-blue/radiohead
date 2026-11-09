"""One skeleton for every attention layer (same as bench/models.py Model): token embedding (+ the sinusoidal input code
for PG, as published), pre-LayerNorm blocks with a GELU feed-forward, final LayerNorm and a linear readout."""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn

from layers import LAYERS, Cfg, sinusoidal


@dataclass
class Net:
    task: str = "lm"              # lm | cls (one position read) | seq (many labelled positions)
    vocab: int = 8004
    n_out: int = 8004
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1
    embed_scale: bool = False
    init_std: float = 0.0         # > 0: normal init of every Linear/Embedding (synthetic tasks)
    padding_idx: object = 0
    pe: str = "none"              # none | sinusoidal

    def to_dict(self):
        return asdict(self)


class Block(nn.Module):
    def __init__(self, net: Net, cfg: Cfg):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(net.d_model), nn.LayerNorm(net.d_model)
        self.attn = LAYERS[cfg.model](cfg, net.d_model, net.n_heads)
        self.ffn = nn.Sequential(nn.Linear(net.d_model, net.d_ff), nn.GELU(), nn.Dropout(net.dropout),
                                 nn.Linear(net.d_ff, net.d_model), nn.Dropout(net.dropout))
        self.keep, self.last = False, None

    def forward(self, x, key_valid=None):
        self.attn.need_weights = self.keep
        out, a, dbg = self.attn(self.n1(x), key_valid)
        if self.keep:
            self.last = (a, dbg)
        x = x + out
        return x + self.ffn(self.n2(x))


class Model(nn.Module):
    def __init__(self, net: Net, cfg: Cfg):
        super().__init__()
        self.net, self.cfg = net, cfg
        self.embed = nn.Embedding(net.vocab, net.d_model, padding_idx=net.padding_idx)
        if net.pe == "sinusoidal":
            self.register_buffer("pe", sinusoidal(8192, net.d_model), persistent=False)
        self.drop = nn.Dropout(net.dropout)
        self.blocks = nn.ModuleList([Block(net, cfg) for _ in range(net.n_layers)])
        self.norm = nn.LayerNorm(net.d_model)
        self.head = nn.Linear(net.d_model, net.n_out)
        if net.init_std > 0:
            for mod in self.modules():
                if isinstance(mod, (nn.Linear, nn.Embedding)):
                    nn.init.normal_(mod.weight, std=net.init_std)
                    if getattr(mod, "bias", None) is not None:
                        nn.init.zeros_(mod.bias)
            for mod in self.modules():
                if hasattr(mod, "reset_mechanism"):
                    mod.reset_mechanism()

    def keep(self, on):
        for b in self.blocks:
            b.keep = on
            if not on:
                b.last = None

    def forward(self, x, key_valid=None, index=None):
        h = self.embed(x)
        if self.net.embed_scale:
            h = h * math.sqrt(self.net.d_model)
        if self.net.pe == "sinusoidal":
            h = h + self.pe[: x.size(1)]
        h = self.drop(h)
        for b in self.blocks:
            h = b(h, key_valid)
        h = self.norm(h)
        if self.net.task == "cls":
            h = h[torch.arange(h.size(0), device=h.device), index]
        return self.head(h)

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def attn_params(self):
        return sum(p.numel() for n, p in self.named_parameters() if ".attn." in n and p.requires_grad)
