"""
probe.py - Stage 2 inner loop. ONE measurement, in ONE fresh process.

This file must never be imported by the orchestrator. It is launched as a
subprocess so that every measurement sees a genuinely cold Python interpreter:
nothing imported, no thread pools spun up, no model in memory. That isolation
is the whole point of the experiment.

It prints a single JSON object to stdout and nothing else.

Run (normally called by run_bench.py):
    python probe.py --loader lgb_native --artifact artifacts/x.lgbtxt \
                    --data data/breast_X.npy --batch 1 --reps 200
"""

import argparse
import json
import os
import sys
import time

# Thread count must be fixed BEFORE numpy / OpenMP are imported, otherwise the
# runtime grabs every core and the timings stop being comparable.
_THREADS = os.environ.get("CS_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = _THREADS

CLOCK = time.perf_counter


def peak_rss_mb():
    """Peak resident set size of THIS process, in megabytes.

    Read from /proc/self/status rather than from getrusage, because some
    container runtimes report a cgroup-wide figure through getrusage that is
    identical for every process and therefore carries no signal.
    """
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float("nan")


# ---------------------------------------------------------------------------
# Loaders. Each returns (predict_fn, t_import_seconds, t_load_seconds).
# The import is timed separately from the deserialize on purpose: that split
# is the finding.
# ---------------------------------------------------------------------------
def load_lgb(path, fmt):
    t0 = CLOCK()
    import lightgbm as lgb
    t_imp = CLOCK() - t0

    t0 = CLOCK()
    if fmt == "native":
        booster = lgb.Booster(model_file=path)
        fn = booster.predict
    else:
        obj = _unpickle(path, fmt)
        fn = obj.predict
    return fn, t_imp, CLOCK() - t0


def load_xgb(path, fmt):
    t0 = CLOCK()
    import xgboost as xgb
    t_imp = CLOCK() - t0

    t0 = CLOCK()
    if fmt == "native":
        booster = xgb.Booster()
        booster.load_model(path)
        fn = booster.inplace_predict          # realistic serving path
    else:
        obj = _unpickle(path, fmt)
        fn = obj.predict
    return fn, t_imp, CLOCK() - t0


def load_cat(path, fmt):
    t0 = CLOCK()
    import catboost
    t_imp = CLOCK() - t0

    t0 = CLOCK()
    if fmt == "native":
        m = catboost.CatBoost()
        m.load_model(path)
        fn = m.predict
    else:
        obj = _unpickle(path, fmt)
        fn = obj.predict
    return fn, t_imp, CLOCK() - t0


def load_onnx(path, fmt):
    t0 = CLOCK()
    import onnxruntime as ort
    t_imp = CLOCK() - t0

    t0 = CLOCK()
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(_THREADS)
    so.inter_op_num_threads = int(_THREADS)
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    t_load = CLOCK() - t0

    def fn(X):
        return sess.run(None, {name: X})

    return fn, t_imp, t_load


def _unpickle(path, fmt):
    if fmt == "joblib":
        import joblib
        return joblib.load(path)
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def get_loader(loader: str):
    lib, fmt = loader.split("_", 1)
    if fmt == "onnx":
        return lambda p: load_onnx(p, fmt)
    return {"lgb": lambda p: load_lgb(p, fmt),
            "xgb": lambda p: load_xgb(p, fmt),
            "cat": lambda p: load_cat(p, fmt)}[lib]


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loader", required=True)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--reps", type=int, default=100)
    args = ap.parse_args()

    t_proc0 = CLOCK()

    # numpy is timed on its own: every backend pays it, so it belongs in the
    # fixed-overhead column rather than being charged to any one format.
    t0 = CLOCK()
    import numpy as np
    t_numpy = CLOCK() - t0

    dtype = np.float32 if args.loader.endswith("onnx") else np.float64
    X_all = np.load(args.data).astype(dtype)
    X = np.ascontiguousarray(X_all[: args.batch])
    if X.shape[0] < args.batch:                      # tile if probe set is small
        reps_needed = int(np.ceil(args.batch / X_all.shape[0]))
        X = np.ascontiguousarray(
            np.tile(X_all, (reps_needed, 1))[: args.batch].astype(dtype))

    predict, t_import, t_load = get_loader(args.loader)(args.artifact)

    # First prediction. Includes lazy allocation, thread-pool spin-up and any
    # one-off kernel setup. This is what an end user's first request pays.
    t0 = CLOCK()
    predict(X)
    t_first = CLOCK() - t0

    # Steady state.
    times = []
    for _ in range(args.reps):
        t0 = CLOCK()
        predict(X)
        times.append(CLOCK() - t0)
    times.sort()
    n = len(times)

    rss_mb = peak_rss_mb()

    print(json.dumps(dict(
        loader=args.loader,
        artifact=os.path.basename(args.artifact),
        batch=args.batch,
        reps=args.reps,
        threads=int(_THREADS),
        t_numpy_s=t_numpy,
        t_import_s=t_import,
        t_load_s=t_load,
        t_first_pred_s=t_first,
        t_steady_p50_s=times[n // 2],
        t_steady_p95_s=times[min(n - 1, int(0.95 * n))],
        t_steady_min_s=times[0],
        t_in_process_s=CLOCK() - t_proc0,
        peak_rss_mb=rss_mb,
    )))


if __name__ == "__main__":
    main()
