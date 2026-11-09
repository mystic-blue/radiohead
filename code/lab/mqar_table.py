"""MQAR results so far: one row per (model, width, lr, seed); the learning rate of each model is the one with the best
validation (accuracy, ties by loss) at d_h 64, seed 0 -- never test.   python3 mqar_table.py [--all]"""
from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["SM", "GATED", "SIGMOID", "PG", "KM", "KMe2", "KM-norm", "KM-noage", "KM-noant"]


def load():
    rows = []
    for f in glob.glob(os.path.join(HERE, "results", "mqar", "*", "*", "*", "s*.json")):
        r = json.load(open(f))
        a = r["args"]
        rows.append(dict(model=r["label"], d=a["d_model"] // a["heads"], lr=a["lr"], seed=a["seed"], val=r["val"],
                         test=r["test"], extra=r["extra"], best=r["best_epoch"], epochs=len(r["history"])))
    return rows


def chosen_lr(rows):
    out = {}
    for m in ORDER:
        c = [r for r in rows if r["model"] == m and r["d"] == 64 and r["seed"] == 0]
        if c:
            out[m] = max(c, key=lambda r: (r["val"]["acc"], -r["val"]["loss"]))["lr"]
    return out


if __name__ == "__main__":
    rows = load()
    lr = chosen_lr(rows)
    show = rows if "--all" in sys.argv else [r for r in rows if lr.get(r["model"]) == r["lr"]]
    print(f"{'model':9s} {'d_h':>4s} {'lr':>7s} {'seed':>4s} {'best':>4s} | {'val':>5s} {'present':>7s} {'absent':>6s} | "
          f"{'P32':>5s} {'P64':>5s} {'P128':>5s}")
    for r in sorted(show, key=lambda r: (ORDER.index(r["model"]) if r["model"] in ORDER else 99, r["d"], r["lr"], r["seed"])):
        t, e = r["test"], r["extra"]
        mark = "*" if lr.get(r["model"]) == r["lr"] else " "
        print(f"{r['model']:9s} {r['d']:4d} {r['lr']:7g}{mark}{r['seed']:4d} {r['best']:4d} | {r['val']['acc']:5.3f} "
              f"{t.get('acc_present', float('nan')):7.3f} {t.get('acc_absent', float('nan')):6.3f} | "
              + " ".join(f"{e.get(k, {}).get('acc', float('nan')):5.2f}" for k in ("P32", "P64", "P128")))
    print("\n* = learning rate chosen on validation (d_h 64, seed 0)")
