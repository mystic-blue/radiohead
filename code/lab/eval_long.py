"""MQAR at longer test contexts than train.py measures (24 Oct 2026): 256 and 496 pairs (T = 528 and 1008; 496 + 16
queries use all 512 keys).  Loads the best-validation weights of each finished run and writes
results/mqar_long/<label>-D<d>-lr<lr>-s<seed>.json.  PG has one learned anchor per position: positions beyond those
built at training get fresh anchors from the same initialization, as unseen positions already had.
python3 eval_long.py --models SM PG KM --d 64"""
import argparse, glob, json, os, random, sys
import torch
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "bench"))
from layers import Cfg                                         # noqa: E402
from net import Net, Model                                     # noqa: E402
from tasks import _mqar                                        # noqa: E402
from oscfamily.data.common import Loader                       # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=["SM", "PG", "KM"])
ap.add_argument("--d", type=int, default=64)
ap.add_argument("--pairs", nargs="+", type=int, default=[256, 496])
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
torch.set_num_threads(a.threads)
sets = {P: Loader(_mqar(a.n, P, 16, 0.2, random.Random(20_000 + P)), max(4, 2048 // (2 * P + 16)), False) for P in a.pairs}
out_dir = os.path.join(HERE, "results", "mqar_long"); os.makedirs(out_dir, exist_ok=True)
for lab in a.models:
    for f in sorted(glob.glob(os.path.join(HERE, "results", "mqar", lab, f"L2-D{a.d}-h1", "*", "s*.pt"))):
        if f.endswith(".ckpt.pt"):
            continue
        lr, seed = f.split(os.sep)[-2].split("-")[0][2:], int(os.path.basename(f)[1:-3])
        dst = os.path.join(out_dir, f"{lab}-D{a.d}-lr{lr}-s{seed}.json")
        if os.path.exists(dst):
            continue
        s = torch.load(f, map_location="cpu", weights_only=False)
        st = dict(s["best"])
        cfg = Cfg(**{**s["cfg"], "pos_len": max(s["cfg"].get("pos_len", 0), 2 * max(a.pairs) + 16)})
        m = Model(Net(**s["net"]), cfg)
        for k, v in list(st.items()):
            if k.endswith("attn.anchors"):                     # PG: keep the trained anchors, fresh ones beyond
                new = m.state_dict()[k].clone(); new[:, : v.shape[1]] = v; st[k] = new
        m.load_state_dict(st); m.eval()
        res = {}
        with torch.no_grad():
            for P, ld in sets.items():
                c = n = cp = np_ = ca = na = 0
                for b in ld:
                    lo = m(b["x"]); y = b["y"]; q = y != -100; hit = (lo.argmax(-1) == y) & q; pr = b["present"] & q
                    c += int(hit.sum()); n += int(q.sum()); cp += int((hit & pr).sum()); np_ += int(pr.sum())
                    ca += int((hit & q & ~pr).sum()); na += int((q & ~pr).sum())
                res[f"P{P}"] = {"acc": c / n, "acc_present": cp / max(np_, 1), "acc_absent": ca / max(na, 1), "T": 2 * P + 16}
        json.dump({"label": lab, "d": a.d, "lr": float(lr), "seed": seed, "extra": res}, open(dst, "w"), indent=1)
        print(lab, a.d, lr, seed, {k: round(v["acc"], 3) for k, v in res.items()}, flush=True)
