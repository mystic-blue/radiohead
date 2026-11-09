# lab: the clean code for the letter's experiments (from 19 Oct 2026)

Replaces `../bench/` for every new run; `bench/` stays as the record of the runs made so far.

| file | what |
|---|---|
| `layers.py` | SM (softmax + RoPE), GATED (Qiu et al. 2025, G1 elementwise), SIGMOID (Ramapuram et al. 2025: ALiBi, b = -log T, QK-norm), PG (Pasqualetti & Guo, as published), KM (radio head: implicit / explicit; controls `norm`, `noage`, `noant`) |
| `net.py` | the common skeleton (pre-LN blocks, GELU FFN, d_ff = 2 d_model) |
| `train.py` | one run: checkpoint every epoch (a killed run resumes exactly), `--stop_after N`, selection on validation, test and extra sets on the selected weights, JSON + weights (best and final) |
| `cua.py` | queue: `python3 cua.py plans/<plan>.txt`; re-reads the plan before each job; stops on a failure |
| `tests.py` | equality with bench (KM, PG, SM), fast path = explicit path, GATED(zero gate) = SM/2, ALiBi slopes, causality of every model, the controls, finite gradients |
| `solver_check.py` | bisection iterations needed by the implicit width (20 kept; float32 precision from 14 on trained models) |
| `speed.py` | ms per training step of each model (MPS) |

Rules: one GPU run at a time (train.py takes `results/.gpu.lock`); nothing chosen on test; data loaders are those of
`bench/tasks.py` (data depend on the seed alone).

Labels: KM (implicit), KMe2 (explicit, 2 Newton steps), KM-norm (no silent branch), KM-noage (no position),
KM-noant (no content).  Results: `results/<task>/<label>/L<layers>-D<d_model>-h<heads>[-f<frac>][-p<pairs>]/lr..-e..[-stopN]-w..-dc../s<seed>.json`.
