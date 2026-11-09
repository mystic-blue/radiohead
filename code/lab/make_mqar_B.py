"""Writes plans/mqar_B_width.txt from the learning rates chosen in phase A (validation, d_h 64, seed 0): every model at
d_h = 16, 32, 64, 128 (one head, two layers), seeds 0, 1, 2, 80 epochs; seed 0 of every width first."""
import os
from mqar_table import ORDER, chosen_lr, load

ARGS = {"SM": "SM", "GATED": "GATED", "SIGMOID": "SIGMOID", "PG": "PG", "KM": "KM", "KMe2": "KM --explicit 2",
        "KM-norm": "KM --norm", "KM-noage": "KM --noage", "KM-noant": "KM --noant"}
lr = chosen_lr(load())
missing = [m for m in ORDER if m not in lr]
assert not missing, f"phase A not finished for {missing}"
lines = ["# MQAR fase B: amplada x seeds al lr triat a la fase A " + str(lr)]
for seed in (0, 1, 2):
    for d in (16, 32, 64, 128):
        for m in ORDER:
            lines.append(f"--task mqar --model {ARGS[m]} --d_model {d} --heads 1 --epochs 80 --lr {lr[m]:g} --seed {seed}")
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "plans", "mqar_B_width.txt"), "w").write("\n".join(lines) + "\n")
print(lines[0]); print(len(lines) - 1, "runs")
