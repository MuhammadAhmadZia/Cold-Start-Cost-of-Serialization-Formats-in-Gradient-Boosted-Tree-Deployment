"""
run_batch_sweep.py - locate the batch size at which the ONNX path stops being
the faster option for each library.

The main benchmark (run_bench.py) covers three batch sizes, which is enough to
show that a crossover exists but not enough to say where it falls. This script
sweeps batch size in powers of two from 1 to 2048 over the native and ONNX
paths, using the largest model of each dataset.

It reuses the artifacts written by prepare.py, so no retraining is needed.
Output goes to results/raw_batchsweep.jsonl. Each experiment writes to its own
file so that measurements taken under different designs are never pooled.

Run:  python scripts/run_batch_sweep.py
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
RAW = os.path.join(RES, "raw_batchsweep.jsonl")

BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
FORMATS = ["native", "onnx"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5,
                    help="cold starts per cell; each is a fresh process")
    ap.add_argument("--reps", type=int, default=100,
                    help="steady-state predictions inside each process")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    man = pd.read_csv(os.path.join(RES, "manifest.csv"))
    man = man[man.status == "ok"]

    # The crossover is a property of the batch and runtime interaction, so the
    # sweep uses the largest model of each dataset rather than every size.
    biggest = man.groupby("dataset").n_trees.max().to_dict()
    sel = man[man.fmt.isin(FORMATS)
              & man.apply(lambda r: r.n_trees == biggest[r.dataset],
                          axis=1)].reset_index(drop=True)

    missing = [a for a in sel.artifact if not os.path.exists(a)]
    if missing:
        sys.exit(f"{len(missing)} artifacts are not on disk. "
                 f"Run scripts/prepare.py first.")

    print(f"{len(sel)} artifacts in the sweep:")
    for _, r in sel.iterrows():
        print(f"   {r.dataset:8s} {r.lib:4s} {r.fmt:7s} "
              f"t{r.n_trees:<5d} {r.bytes/1e6:7.2f} MB")

    have = set()
    if os.path.exists(RAW):
        for line in open(RAW):
            try:
                r = json.loads(line)
                have.add((r["artifact"], r["batch"], r["repeat"]))
            except (ValueError, KeyError):
                continue

    todo = [(r, b, k) for _, r in sel.iterrows() for b in BATCHES
            for k in range(args.repeats)
            if (os.path.basename(r.artifact), b, k) not in have]
    print(f"\n{len(todo)} measurements to run, {len(have)} already recorded\n")

    env = dict(os.environ, CS_THREADS=str(args.threads))
    t0_all = time.perf_counter()
    with open(RAW, "a") as out:
        for i, (row, batch, k) in enumerate(todo, 1):
            cmd = [sys.executable, PROBE,
                   "--loader", row.loader,
                   "--artifact", row.artifact,
                   "--data", os.path.join(DATA, f"{row.dataset}_X.npy"),
                   "--batch", str(batch),
                   "--reps", str(args.reps)]
            try:
                p = subprocess.run(cmd, env=env, capture_output=True,
                                   text=True, timeout=args.timeout)
            except subprocess.TimeoutExpired:
                print(f"  timeout: {row.loader} batch={batch}")
                continue
            if p.returncode != 0:
                print(f"  error: {row.loader} batch={batch}: "
                      f"{p.stderr.strip().splitlines()[-1][:110]}")
                continue

            rec = json.loads(p.stdout.strip().splitlines()[-1])
            rec.update(dataset=row.dataset, lib=row.lib, fmt=row.fmt,
                       n_trees=int(row.n_trees), bytes=int(row.bytes),
                       repeat=k, experiment="batch_sweep")
            out.write(json.dumps(rec) + "\n")
            out.flush()

            if i % 50 == 0 or i == len(todo):
                rate = (time.perf_counter() - t0_all) / i
                print(f"  {i}/{len(todo)}   "
                      f"eta {(len(todo)-i)*rate/60:5.1f} min", flush=True)

    print(f"\nwrote {RAW}")


if __name__ == "__main__":
    main()
