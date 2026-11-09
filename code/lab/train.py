"""One training run: build, train, select on validation, report test, write a JSON and the weights.

    python3 train.py --task mqar --model KM --seed 0 [--lr 1e-3] [--epochs 30] [--stop_after 12] ...

Protocol (the same for every model): AdamW (weight decay 1e-4 on matrices), gradient clipping 1, learning rate warmed
up linearly over `warmup` of the steps, constant, then decayed linearly to 0 over the last `decay` of the steps of
`epochs` epochs.  `stop_after N` stops cleanly after epoch N of that schedule (the curve up to N is the same as in the
full run).  The epoch with the best validation (lm: loss; otherwise accuracy, ties by loss) gives the weights on which
test and every extra set are measured.  Nothing is ever chosen on test.

A run whose loss becomes non-finite stops there; its result is the best validation epoch before it (recorded as
diverged_at).  Robustness: a checkpoint (model, optimizer, schedule, random state, history) is written after every epoch and a killed
run resumes from it; the result is written atomically at the end.  Only one run may use the GPU at a time (lock file).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from layers import Cfg                                                    # noqa: E402
from net import Net, Model                                                # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(HERE), "bench"))
from tasks import TASKS                                                   # noqa: E402  (data: same loaders as bench)

RES = os.path.join(HERE, "results")
GPU_LOCK = os.path.join(RES, ".gpu.lock")


# ------------------------------------------------------------------ names

def label(a):
    """KM variants get their own model label: KM (implicit), KMe2 (explicit, 2 steps), KMnorm, KMnoage, KMnoant."""
    if a.model != "KM":
        return a.model
    s = "KM" + (f"e{a.explicit}" if a.explicit else "")
    return s + "".join(f"-{c}" for c in ("norm", "noage", "noant", "same") if getattr(a, c)) + (f"-k{a.k0:g}" if a.k0 != 1.5 else "")


def result_path(a):
    arch = f"L{a.layers}-D{a.d_model}-h{a.heads}"
    sched = f"lr{a.lr:g}-e{a.epochs}" + (f"-stop{a.stop_after}" if a.stop_after else "") + f"-w{a.warmup:g}-dc{a.decay:g}"
    data = (f"-f{a.frac:g}" if a.task == "lm" else "") + (f"-p{a.pairs}" if a.pairs else "") + (f"-k{a.kmax}" if a.kmax else "") + ("-reg" if a.reg else "")
    return os.path.join(RES, a.task, label(a), arch + data, sched, f"s{a.seed}.json")


# ------------------------------------------------------------------ steps

def forward(model, batch, kind, dev):
    if kind == "lm":
        tok = batch["tokens"].to(dev).long()
        inp, tgt = tok[:, :-1], tok[:, 1:]
        logits = model(inp, key_valid=inp != 0).float()
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=0, reduction="none")
        return ce.view_as(tgt), tgt != 0, logits, tgt
    x, y = batch["x"].to(dev), batch["y"].to(dev)
    if kind == "reg":                                         # count as a number: squared error, hit = within 0.5
        out = model(x, index=batch["index"].to(dev)).float().squeeze(-1)
        return (out - y.float()) ** 2, torch.ones_like(y, dtype=torch.bool), out, y
    if kind == "cls":
        logits = model(x, index=batch["index"].to(dev)).float()
        return F.cross_entropy(logits, y, reduction="none"), torch.ones_like(y, dtype=torch.bool), logits, y
    logits = model(x).float()
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100, reduction="none")
    return ce.view_as(y), y != -100, logits, y


@torch.no_grad()
def evaluate(model, loader, kind, dev):
    model.eval()
    L = n = c = cp = npres = ca = nabs = 0.0
    for b in loader:
        ce, m, logits, y = forward(model, b, kind, dev)
        hit = (((logits - y).abs() < 0.5) if kind == "reg" else (logits.argmax(-1) == y)) & m
        L += float((ce * m).sum()); n += float(m.sum()); c += float(hit.sum())
        if kind == "seq" and "present" in b:
            pr = b["present"].to(dev) & m
            cp += float((hit & pr).sum()); npres += float(pr.sum())
            ca += float((hit & ~pr).sum()); nabs += float((m & ~pr).sum())
    out = {"loss": L / max(n, 1), "acc": c / max(n, 1)}
    if kind == "lm":
        out["ppl"] = math.exp(out["loss"])
    if kind == "seq" and npres + nabs:
        out["acc_present"] = cp / max(npres, 1); out["acc_absent"] = ca / max(nabs, 1)
    return out


def better(new, old, kind):
    if old is None:
        return True
    if kind == "lm":
        return new["loss"] < old["loss"]
    return (new["acc"], -new["loss"]) > (old["acc"], -old["loss"])


# ------------------------------------------------------------------ GPU lock

def take_gpu():
    os.makedirs(RES, exist_ok=True)
    if os.path.exists(GPU_LOCK):
        pid = int(open(GPU_LOCK).read().strip() or 0)
        try:
            os.kill(pid, 0)
            raise SystemExit(f"GPU in use by pid {pid} ({GPU_LOCK}); never two GPU runs at once")
        except ProcessLookupError:
            pass                                           # stale lock of a dead run
    with open(GPU_LOCK, "w") as f:
        f.write(str(os.getpid()))


def drop_gpu():
    if os.path.exists(GPU_LOCK) and open(GPU_LOCK).read().strip() == str(os.getpid()):
        os.remove(GPU_LOCK)


# ------------------------------------------------------------------ one run

def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--model", required=True, choices=["SM", "ALIBI", "ALIBIL", "GATED", "SIGMOID", "PG", "KM"])
    ap.add_argument("--explicit", type=int, default=0)      # KM: Newton steps of the explicit width (0 = implicit)
    ap.add_argument("--norm", action="store_true")          # KM control: no silent branch
    ap.add_argument("--noage", action="store_true")         # KM control: no ageing (no position)
    ap.add_argument("--noant", action="store_true")         # KM control: no antenna (no content)
    ap.add_argument("--same", action="store_true")          # KM: identical initial ageing rate in every head
    ap.add_argument("--k0", type=float, default=1.5)        # KM: initial K/K_c (< 1 starts in the silent phase)
    ap.add_argument("--d_model", type=int, default=None)
    ap.add_argument("--heads", type=int, default=None)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=None)     # length of the schedule
    ap.add_argument("--stop_after", type=int, default=0)    # stop cleanly after this epoch (0: run the whole schedule)
    ap.add_argument("--warmup", type=float, default=0.02)
    ap.add_argument("--decay", type=float, default=0.2)
    ap.add_argument("--frac", type=float, default=0.1)      # lm: fraction of TinyStories
    ap.add_argument("--pairs", type=int, default=None)      # mqar: key-value pairs (default 16)
    ap.add_argument("--reg", action="store_true")             # count: read the count as a number (one output, squared error)
    ap.add_argument("--kmax", type=int, default=None)       # count: largest number of earlier copies (default 3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")         # a few steps, nothing written
    a = ap.parse_args(argv)
    if a.model != "KM":
        assert not (a.explicit or a.norm or a.noage or a.noant or a.same or a.k0 != 1.5), "KM options on a non-KM model"
    T = TASKS[a.task]
    a.d_model = a.d_model or T["net"]["d_model"]
    a.heads = a.heads or T["net"]["n_heads"]
    a.epochs = a.epochs or T["epochs"]
    return a


def main(argv=None):
    a = parse(argv)
    T = TASKS[a.task]
    out = result_path(a)
    if os.path.exists(out) and not a.smoke:
        print("done already:", out); return json.load(open(out))
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device or ("mps" if a.task == "lm" else "cpu"))
    if dev.type == "mps" and not a.smoke:
        take_gpu()
    try:
        return _run(a, T, out, dev)
    finally:
        if dev.type == "mps":
            drop_gpu()


def _run(a, T, out, dev):
    torch.manual_seed(a.seed)
    if a.task == "lm":
        tr, va, te, extra, meta = T["load"](seed=a.seed, frac=a.frac, max_eval=(512 if a.smoke else 0))
    else:
        tr, va, te, extra, meta = T["load"](seed=a.seed, **({"n_pairs": a.pairs} if a.pairs else {}), **({"kmax": a.kmax} if a.kmax else {}))
    nk = dict(T["net"])
    nk.update(d_model=a.d_model, n_heads=a.heads, d_ff=2 * a.d_model)
    if a.reg:
        assert a.task == "count", "--reg is for count"
        meta = {**meta, "kind": "reg", "n_out": 1}
    net = Net(task="cls" if meta["kind"] == "reg" else meta["kind"], vocab=meta["vocab"], n_out=meta["n_out"], n_layers=a.layers,
              pe="sinusoidal" if a.model == "PG" else "none", **nk)
    longest = max([meta["max_len"]] + [int(next(iter(ld))["x"].shape[1]) for ld in extra.values()] if a.task != "lm" else [meta["max_len"]])
    cfg = Cfg(model=a.model, mask_self=T["mask_self"], max_len=meta["max_len"], pos_len=longest,
              km_explicit=a.explicit, km_norm=a.norm, km_noage=a.noage, km_noant=a.noant, km_same=a.same, km_k0=a.k0)
    model = Model(net, cfg).to(dev)
    mats = [p for p in model.parameters() if p.ndim >= 2 and p.requires_grad]
    vecs = [p for p in model.parameters() if p.ndim < 2 and p.requires_grad]
    opt = torch.optim.AdamW([{"params": mats, "weight_decay": 1e-4}, {"params": vecs, "weight_decay": 0.0}], lr=a.lr)
    total = a.epochs * len(tr)
    warm, dec = max(1, int(a.warmup * total)), int(a.decay * total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm, max(0.0, (total - s) / dec) if dec else 1.0))
    kind = meta["kind"]
    last_ep = a.stop_after or a.epochs
    if a.smoke:
        last_ep = 1
    ckpt = out[:-5] + ".ckpt.pt"
    hist, best, best_state, start = [], None, None, 1
    if os.path.exists(ckpt) and not a.smoke:
        c = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"])
        torch.set_rng_state(c["rng"])
        hist, best, best_state, start = c["hist"], c["best"], c["best_state"], c["epoch"] + 1
        print(f"resuming after epoch {c['epoch']}", flush=True)
    print(f"[{a.task} {label(a)} L{a.layers} D{a.d_model} h{a.heads} lr{a.lr:g} s{a.seed}] params {model.n_params():,} "
          f"(attention {model.attn_params():,}) {len(tr)} steps x {a.epochs} epochs, stop after {last_ep} | {dev}", flush=True)
    t0 = time.time()
    diverged = None
    for ep in range(start, last_ep + 1):
        model.train(); tr.set_epoch(ep); te0 = time.time(); Ls = Ns = 0.0
        for bi, b in enumerate(tr):
            ce, m, _, _ = forward(model, b, kind, dev)
            loss = (ce * m).sum() / m.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            Ls += float(loss.detach()) * float(m.sum()); Ns += float(m.sum())
            if a.smoke and bi >= 20:
                break
        if not math.isfinite(Ls):                          # diverged: keep the best validation epoch so far, say so
            print(f"  non-finite loss at epoch {ep}: stopping; the result is the best validation epoch before it", flush=True)
            diverged = ep
            break
        v = evaluate(model, va, kind, dev)
        row = {"epoch": ep, "train_loss": Ls / max(Ns, 1), "lr": sched.get_last_lr()[0], "secs": round(time.time() - te0, 1),
               **{"val_" + k: x for k, x in v.items()}}
        hist.append(row)
        if better(v, best["val"] if best else None, kind):
            best = {"epoch": ep, "val": v}
            best_state = {k: x.detach().cpu().clone() for k, x in model.state_dict().items()}
        print(f"  ep {ep}/{a.epochs} train {row['train_loss']:.4f} val " + " ".join(f"{k} {x:.4f}" for k, x in v.items())
              + f" ({row['secs']}s)", flush=True)
        if dev.type == "mps":
            torch.mps.empty_cache()
        if not a.smoke:
            os.makedirs(os.path.dirname(out), exist_ok=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "rng": torch.get_rng_state(), "hist": hist, "best": best, "best_state": best_state, "epoch": ep},
                       ckpt + ".tmp")
            os.replace(ckpt + ".tmp", ckpt)
    final_state = {k: x.detach().cpu().clone() for k, x in model.state_dict().items()}
    if best_state is None:
        raise FloatingPointError("diverged before the first validation")
    model.load_state_dict(best_state)
    best["test"] = evaluate(model, te, kind, dev)
    ext = {}
    for name, ld in extra.items():
        try:
            ext[name] = evaluate(model, ld, kind, dev)
        except RuntimeError as e:                          # PG cannot read beyond its anchors
            ext[name] = {"error": str(e)[:200]}
    res = {"args": vars(a), "label": label(a), "cfg": cfg.to_dict(), "net": net.to_dict(), "meta": meta,
           "params": model.n_params(), "attn_params": model.attn_params(), "history": hist, "best_epoch": best["epoch"],
           "val": best["val"], "test": best["test"], "extra": ext, "diverged_at": diverged, "secs": round(time.time() - t0, 1), "device": str(dev),
           "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(f"  -> best epoch {best['epoch']}, test " + " ".join(f"{k} {x:.4f}" for k, x in best["test"].items())
          + "".join(f" | {n_} {e.get('acc', e.get('ppl', float('nan'))):.3f}" for n_, e in ext.items()), flush=True)
    if a.smoke:
        return res
    torch.save({"best": best_state, "final": final_state, "cfg": cfg.to_dict(), "net": net.to_dict()}, out[:-5] + ".pt")
    with open(out + ".tmp", "w") as f:
        json.dump(res, f, indent=1)
    os.replace(out + ".tmp", out)
    os.remove(ckpt)
    return res


if __name__ == "__main__":
    main()
