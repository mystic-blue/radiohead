"""A queue of runs.  python3 cua.py plans/<plan>.txt

Each non-empty line of the plan (not starting with #) holds the arguments of one train.py run.  The worker takes the
first line whose result does not exist and that no live worker holds (lock = a file with the worker's pid), runs it,
and re-reads the plan before every job, so lines can be added, removed or reordered while it runs.  Several CPU workers
may share a plan; train.py itself refuses a second GPU run.  A killed run resumes from its last epoch when the queue is
started again.  Logs: logs/<plan>/<line>.log.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train                                                              # noqa: E402

LOCKS = os.path.join(HERE, "results", ".locks")


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def take(name):
    os.makedirs(LOCKS, exist_ok=True)
    f = os.path.join(LOCKS, name)
    if os.path.exists(f):
        pid = int(open(f).read().strip() or 0)
        if pid and alive(pid):
            return None
        os.remove(f)                                       # stale: its worker is dead
    try:
        fd = os.open(f, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    return f


def main(plan):
    logdir = os.path.join(HERE, "logs", os.path.splitext(os.path.basename(plan))[0])
    os.makedirs(logdir, exist_ok=True)
    while True:
        jobs = [l.split() for l in open(plan) if l.strip() and not l.lstrip().startswith("#")]
        job = None
        for args in jobs:
            out = train.result_path(train.parse(args))
            if os.path.exists(out):
                continue
            name = "_".join(x.lstrip("-") for x in args)
            lock = take(name)
            if lock:
                job = (args, name, lock)
                break
        if job is None:
            print(f"[{time.strftime('%H:%M:%S')}] queue empty", flush=True)
            return
        args, name, lock = job
        print(f"[{time.strftime('%H:%M:%S')}] {' '.join(args)}", flush=True)
        t0 = time.time()
        with open(os.path.join(logdir, name + ".log"), "a") as f:
            f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(args)}\n"); f.flush()
            r = subprocess.run([sys.executable, os.path.join(HERE, "train.py"), *args], stdout=f, stderr=subprocess.STDOUT, cwd=HERE)
        os.remove(lock)
        print(f"    {'done' if r.returncode == 0 else f'FAILED ({r.returncode})'} in {time.time() - t0:.0f}s", flush=True)
        if r.returncode != 0:
            print(f"    log: {os.path.join(logdir, name + '.log')}", flush=True)
            return                                         # stop on failure: look before going on


if __name__ == "__main__":
    main(sys.argv[1])
