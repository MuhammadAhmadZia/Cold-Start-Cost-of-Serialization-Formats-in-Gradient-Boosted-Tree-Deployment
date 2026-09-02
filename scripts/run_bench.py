"""
run_bench.py - the main cold-start benchmark.

Reads results/manifest.csv and, for every artifact by batch size by repeat,
launches probe.py as a separate operating system process. Each launch is one
cold start. The orchestrator also records total process wall time from the
outside, which captures interpreter start-up that the probe cannot observe
from within itself.

Results append to results/raw_main.jsonl after every measurement, so the run is
resumable: stop with Ctrl-C, run the same command again, and completed cells
are skipped. A smoke-test run (--quick) writes to results/raw_smoketest.jsonl
instead, keeping the two designs in separate files.

Run:  python scripts/run_bench.py --quick     (about 3 minutes)
      python scripts/run_bench.py             (about 40 minutes)
"""

import argparse
import json
import os
import subprocess
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data")
PROBE = os.path.join(HERE, "probe.py")


def interpreter_floor(env, n=5):
    """Wall time of a Python process that does nothing: the unavoidable floor."""
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        subprocess.run([sys.executable, "-c", "pass"], env=env, check=True)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def completed(path):
    keys = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
                keys.add((r["artifact"], r["loader"], r["batch"], r["repeat"]))
            except (ValueError, KeyError):
                continue
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="short smoke-test run, written to a separate file")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    raw = os.path.join(RES, "raw_smoketest.jsonl" if args.quick
                       else "raw_main.jsonl")
    batches = [1, 32] if args.quick else [1, 32, 1024]
    repeats = 2 if args.quick else args.repeats

    man = pd.read_csv(os.path.join(RES, "manifest.csv"))
    man = man[man.status == "ok"].reset_index(drop=True)

    env = dict(os.environ, CS_THREADS=str(args.threads))
    floor = interpreter_floor(env)
    print(f"bare interpreter start-up floor: {floor*1000:.1f} ms")
    print(f"writing to {os.path.basename(raw)}\n")

    have = completed(raw)
    todo = [(r, b, k) for _, r in man.iterrows() for b in batches
            for k in range(repeats)
            if (os.path.basename(r.artifact), r.loader, b, k) not in have]
    print(f"{len(todo)} measurements to run "
          f"({len(man)} artifacts x {len(batches)} batches x {repeats} repeats)\n")

    t0_all = time.perf_counter()
    with open(raw, "a") as out:
        for i, (row, batch, k) in enumerate(todo, 1):
            cmd = [sys.executable, PROBE,
                   "--loader", row.loader,
                   "--artifact", row.artifact,
                   "--data", os.path.join(DATA, f"{row.dataset}_X.npy"),
                   "--batch", str(batch),
                   "--reps", str(args.reps)]
            t0 = time.perf_counter()
            try:
                p = subprocess.run(cmd, env=env, capture_output=True,
                                   text=True, timeout=args.timeout)
            except subprocess.TimeoutExpired:
                print(f"  timeout: {row.loader} batch={batch}")
                continue
            wall = time.perf_counter() - t0

            if p.returncode != 0:
                print(f"  error: {row.loader} batch={batch}: "
                      f"{p.stderr.strip().splitlines()[-1][:110]}")
                continue

            rec = json.loads(p.stdout.strip().splitlines()[-1])
            rec.update(dataset=row.dataset, lib=row.lib, fmt=row.fmt,
                       n_trees=int(row.n_trees), bytes=int(row.bytes),
                       repeat=k, t_process_wall_s=wall,
                       interpreter_floor_s=floor,
                       experiment="smoketest" if args.quick else "main")
            out.write(json.dumps(rec) + "\n")
            out.flush()

            if i % 25 == 0 or i == len(todo):
                rate = (time.perf_counter() - t0_all) / i
                print(f"  {i}/{len(todo)}   "
                      f"eta {(len(todo)-i)*rate/60:5.1f} min", flush=True)

    print(f"\nwrote {raw}")


if __name__ == "__main__":
    main()
