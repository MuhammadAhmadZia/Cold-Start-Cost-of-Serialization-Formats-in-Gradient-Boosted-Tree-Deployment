"""
run_coldcache.py - measure cold start with the operating system page cache
evicted, instead of warm.

The main benchmark measures process-level cold starts: a fresh process, but the
artifact file and the library shared objects are already resident in the page
cache. A serverless container pulled onto a fresh host pays more than that.
This script varies how much of the cache is evicted before each launch, so the
difference can be measured rather than assumed.

Eviction levels, from least to most aggressive:

  warm        no eviction. Reproduces the main benchmark, and acts as the
              control arm so the levels are comparable within one session.
  artifact    evicts only the model file, using posix_fadvise(DONTNEED).
              Needs no special privileges and runs on a hosted notebook.
  libs        evicts the installed package trees for the model library, NumPy
              and ONNX Runtime, as well as the artifact. Also needs no
              privileges.
  full        drops the entire system page cache via /proc/sys/vm/drop_caches.
              Needs root, so this arm only runs on a machine you control.

Results go to results/env_<label>/raw_coldcache.jsonl with the eviction level
recorded on every measurement.

Run:  python scripts/run_coldcache.py --env local --levels warm,artifact,libs
      sudo -E python scripts/run_coldcache.py --env local --levels warm,full
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

LIB_PACKAGES = {
    "lgb": ["lightgbm"],
    "xgb": ["xgboost"],
    "cat": ["catboost"],
}
ALWAYS_EVICT = ["numpy", "joblib"]
ONNX_PACKAGES = ["onnxruntime"]


# ---------------------------------------------------------------------------
# Cache eviction
# ---------------------------------------------------------------------------
def evict_file(path):
    """Drop one file from the page cache. No privileges needed."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return 0
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return 1
    except (OSError, AttributeError):
        return 0
    finally:
        os.close(fd)


def evict_tree(root_dir, suffixes=(".so", ".pyd", ".py", ".dylib")):
    """Drop a whole installed package from the page cache."""
    n = 0
    for dirpath, _, names in os.walk(root_dir):
        for name in names:
            if name.endswith(suffixes):
                n += evict_file(os.path.join(dirpath, name))
    return n


def package_dirs(names):
    """Locate installed packages without importing them into this process."""
    out = []
    for name in names:
        try:
            r = subprocess.run(
                [sys.executable, "-c",
                 f"import importlib.util,os;"
                 f"s=importlib.util.find_spec('{name}');"
                 f"print(os.path.dirname(s.origin) if s and s.origin else '')"],
                capture_output=True, text=True, timeout=60)
            d = r.stdout.strip()
            if d and os.path.isdir(d):
                out.append(d)
        except (OSError, subprocess.SubprocessError):
            continue
    return out


def drop_all_caches():
    """Drop the entire system page cache. Requires root."""
    os.sync()
    with open("/proc/sys/vm/drop_caches", "w") as f:
        f.write("3\n")


def apply_eviction(level, artifact, lib, cache):
    if level == "warm":
        return
    os.sync()
    if level == "full":
        drop_all_caches()
        return
    evict_file(artifact)
    if level == "libs":
        names = LIB_PACKAGES.get(lib, []) + ALWAYS_EVICT + ONNX_PACKAGES
        key = tuple(sorted(names))
        if key not in cache:
            cache[key] = package_dirs(names)
        for d in cache[key]:
            evict_tree(d)


def check_level_available(level):
    if level == "full":
        if os.geteuid() != 0:
            return "needs root; rerun that level with sudo -E"
        if not os.access("/proc/sys/vm/drop_caches", os.W_OK):
            return "/proc/sys/vm/drop_caches is not writable here"
    if level in ("artifact", "libs") and not hasattr(os, "posix_fadvise"):
        return "posix_fadvise is unavailable on this platform"
    return None


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, help="platform label")
    ap.add_argument("--levels", default="warm,artifact,libs",
                    help="comma separated: warm, artifact, libs, full")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--reps", type=int, default=20,
                    help="steady-state predictions; kept low because this "
                         "experiment is about start-up, not throughput")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--formats", default="native,onnx",
                    help="comma separated; use all four only if you have time")
    args = ap.parse_args()

    levels = [x.strip() for x in args.levels.split(",") if x.strip()]
    for lv in levels:
        if lv not in ("warm", "artifact", "libs", "full"):
            sys.exit(f"unknown eviction level: {lv}")
        why = check_level_available(lv)
        if why:
            sys.exit(f"cannot run level '{lv}': {why}")

    out_dir = os.path.join(RES, f"env_{args.env}")
    os.makedirs(out_dir, exist_ok=True)
    raw = os.path.join(out_dir, "raw_coldcache.jsonl")

    man = pd.read_csv(os.path.join(RES, "manifest.csv"))
    man = man[(man.status == "ok")
              & man.fmt.isin([f.strip() for f in args.formats.split(",")])]
    man = man.reset_index(drop=True)

    missing = [a for a in man.artifact if not os.path.exists(a)]
    if missing:
        sys.exit(f"{len(missing)} artifacts are not on disk. "
                 f"Run scripts/prepare.py first.")

    have = set()
    if os.path.exists(raw):
        for line in open(raw):
            try:
                r = json.loads(line)
                have.add((r["artifact"], r["eviction"], r["repeat"]))
            except (ValueError, KeyError):
                continue

    # Interleave the levels rather than running each level to completion, so
    # that machine drift over the session cannot be mistaken for an effect of
    # eviction.
    todo = [(r, lv, k)
            for k in range(args.repeats)
            for lv in levels
            for _, r in man.iterrows()
            if (os.path.basename(r.artifact), lv, k) not in have]

    print(f"environment : {args.env}")
    print(f"levels      : {', '.join(levels)}")
    print(f"artifacts   : {len(man)}   batch size {args.batch}")
    print(f"to run      : {len(todo)} measurements "
          f"({len(have)} already recorded)\n")

    env = dict(os.environ, CS_THREADS=str(args.threads))
    pkg_cache = {}
    t0_all = time.perf_counter()

    with open(raw, "a") as out:
        for i, (row, level, k) in enumerate(todo, 1):
            apply_eviction(level, row.artifact, row.lib, pkg_cache)

            cmd = [sys.executable, PROBE,
                   "--loader", row.loader,
                   "--artifact", row.artifact,
                   "--data", os.path.join(DATA, f"{row.dataset}_X.npy"),
                   "--batch", str(args.batch),
                   "--reps", str(args.reps)]
            t0 = time.perf_counter()
            try:
                p = subprocess.run(cmd, env=env, capture_output=True,
                                   text=True, timeout=args.timeout)
            except subprocess.TimeoutExpired:
                print(f"  timeout: {row.loader} level={level}")
                continue
            wall = time.perf_counter() - t0

            if p.returncode != 0:
                print(f"  error: {row.loader} level={level}: "
                      f"{p.stderr.strip().splitlines()[-1][:110]}")
                continue

            rec = json.loads(p.stdout.strip().splitlines()[-1])
            rec.update(dataset=row.dataset, lib=row.lib, fmt=row.fmt,
                       n_trees=int(row.n_trees), bytes=int(row.bytes),
                       repeat=k, eviction=level, t_process_wall_s=wall,
                       experiment="coldcache", env_label=args.env)
            out.write(json.dumps(rec) + "\n")
            out.flush()

            if i % 25 == 0 or i == len(todo):
                rate = (time.perf_counter() - t0_all) / i
                print(f"  {i}/{len(todo)}   "
                      f"eta {(len(todo)-i)*rate/60:5.1f} min", flush=True)

    print(f"\nwrote {raw}")
    print("\nNote: the interpreter start-up floor recorded by the main "
          "benchmark does not apply here, because a dropped cache also slows "
          "interpreter start-up. Use t_process_wall_s for the cold-cache "
          "comparison.")


if __name__ == "__main__":
    main()
