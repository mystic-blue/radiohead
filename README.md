# Everything in its right place: attention is a radio head

Code and data for the letter by L. Arola-Fernández, A. Khadraoui and A. Arenas (2026) and its Supplemental Material.

The radio head replaces softmax attention by the stationary state of a receiver, a Lohe/Kuramoto oscillator that listens
to past tokens whose frequency drifts with their age. Its weights are

    w_ij = K G_ij / (K r_i + i Delta_ij),

with a locked phase (weights normalized in power) and a silent phase (weights summing to less than one) separated by a
threshold K_c set by the context.

## Layout

| folder | what |
|---|---|
| `code/lab/` | the layers and the training code used for every run of the letter (see `code/lab/README.md`) |
| `code/lab/layers.py` | radio head (`KM`), softmax + RoPE (`SM`), linear biases (`ALIBI`, `ALIBIL` with learned slopes), sigmoid (`SIGMOID`), gated softmax (`GATED`), Pasqualetti and Guo (`PG`) |
| `code/lab/plans/` | the run queues (one line per run, arguments of `train.py`) |
| `code/lab/results/` | one JSON per run: arguments, history, validation and test metrics, longer contexts (weights not included) |
| `code/bench/` | data loaders of the tasks (`tasks.py`) and the reference implementation used by `tests.py` |
| `code/oscfamily/` | data utilities (synthetic tasks, TinyStories) |
| `code/derivation/` | integration of the receiver dynamics against the closed form (Fig. 2a-c) |
| `code/figures/` | figure scripts and the data they read (`dumps*/`) |
| `code/third_party/` | the published attention of Pasqualetti and Guo (MIT) and of the BSc thesis, used only for equivalence tests |

## Quick start

    pip install -r requirements.txt
    cd code/lab
    python3 tests.py                                     # checks of every layer; ends with "TOT BÉ"
    python3 train.py --task mqar --model KM --d_model 64 --heads 1 --layers 2 --lr 1e-3 --epochs 80 --seed 0
    python3 cua.py plans/<plan>.txt                      # a whole queue

Run `python3 train.py --help` for all options. Results are written to
`code/lab/results/<task>/<label>/<arch>/<schedule>/s<seed>.json`.

## Figures

    cd code/figures
    python3 make_fig1_radio.py
    F1=1 python3 make_fig2_combined.py                   # Fig. 2 (full-data language models)
    python3 make_fig2_physics.py                         # Fig. 3

`dumps/mqar_where.json` is stored compressed: run `gunzip -k code/figures/dumps/mqar_where.json.gz` first.

## Data

The synthetic tasks (recall, counting, induction) are generated on the fly from the seed. The language experiments use
TinyStories (Eldan and Li, 2023) with the vocabulary and validation split of Pasqualetti and Guo; place them as described
in `code/oscfamily/paths.py` (`data/tinystories_eval/`, `data/tinystories_raw/`).

## Citation

L. Arola-Fernández, A. Khadraoui and A. Arenas, Everything in its right place: attention is a radio head (2026).
