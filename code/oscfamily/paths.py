"""Where things live (radiohead/current/kuramotor since 14 Oct 2026; originally PRL_attention_synchrony/):

    05_data/                   corpora and feature caches
    04_results/runs/           the run store (one JSON per run, checkpoints, physics/)
    01_draft/                  the article
    03_code/third_party/       Pasqualetti & Guo's released attention (equivalence test only)

Every location can be overridden with an environment variable.
"""
import os

PKG = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PKG)                                    # 03_code/oscfamily/
PROJECT = os.path.dirname(ROOT)                                # radiohead/current/kuramotor/  (ROOT = code/)

DATA = os.environ.get("OSC_DATA", os.path.join(PROJECT, "data"))
CACHE = os.environ.get("OSC_CACHE", DATA)                      # tinystories_packed.pt lives here
TS_EVAL = os.path.join(DATA, "tinystories_eval")               # Pasqualetti & Guo's vocab + validation split
TS_TRAIN = os.path.join(DATA, "tinystories_raw", "train_chunks.pkl")   # only needed to rebuild the packed cache
KWS_CACHE = os.path.join(DATA, "kws_40x49_h320_global.pt")      # Speech Commands, 10 words, 40 log-mel x 49

PASQ = os.environ.get("OSC_PASQ", os.path.join(ROOT, "third_party", "pasqualetti"))
KHAD = os.path.join(ROOT, "third_party", "khadraoui")    # the thesis' published attention (unmodified)

RUNS = os.environ.get("OSC_RUNS", os.path.join(PROJECT, "04_results", "runs"))
RUNS_V8 = os.path.join(PROJECT, "04_results", "runs_v8")      # the v8 plans (clean restart, 25 Sep 2026)
RUNS_FINAL = os.path.join(PROJECT, "04_results", "runs_final")  # the final runs (1 Oct 2026), plans final_*
DRAFT = os.path.join(PROJECT, "01_draft")
