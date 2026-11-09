#!/bin/bash
# 6 Nov 2026: phase A (lr sweep, seed 0) of ALIBIL, then phase B (seeds 1-4 at the lr chosen on validation).
cd "$(dirname "$0")"
for i in 1 2 3; do python3 cua.py plans/alibil_A_lr.txt > logs/cua_alibil_A_$i.out 2>&1 & done; wait
python3 - <<'PY'
import json, glob
lines = ["# ALIBIL seeds 1-4 at the lr chosen on validation at seed 0 (written by alibil_driver.sh)"]
for task, arch, extra in [("mqar", "L2-D64-h1", ""), ("count", "L2-D64-h1-k15-reg", " --kmax 15 --reg"), ("induction", "L2-D64-h1", "")]:
    best = max(glob.glob(f"results/{task}/ALIBIL/{arch}/*/s0.json"), key=lambda f: (json.load(open(f))["val"]["acc"], -json.load(open(f))["val"]["loss"]))
    lr = best.split("/")[4].split("-")[0][2:]
    lines.append(f"# {task}: lr {lr}")
    lines += [f"--task {task} --model ALIBIL{extra} --epochs 80 --lr {lr} --seed {s}" for s in (1, 2, 3, 4)]
open("plans/alibil_B_seeds.txt", "w").write("\n".join(lines) + "\n")
PY
for i in 1 2 3; do python3 cua.py plans/alibil_B_seeds.txt > logs/cua_alibil_B_$i.out 2>&1 & done; wait
