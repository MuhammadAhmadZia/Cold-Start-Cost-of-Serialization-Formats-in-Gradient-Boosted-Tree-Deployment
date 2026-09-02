"""
prepare.py - Stage 1: train models and export them in every serialization format.

Produces:
  artifacts/<tag>.<ext>   one file per (dataset, library, n_trees, format)
  data/<dataset>_X.npy    a held-out feature matrix used later for prediction
  results/manifest.csv    one row per artifact, with export cost and file size

Run:  python prepare.py            (full)
      python prepare.py --quick    (tiny smoke-test version, ~1 minute)
"""

import argparse
import json
import os
import pickle
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import (
    fetch_california_housing,
    fetch_covtype,
    load_breast_cancer,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
DATA = os.path.join(HERE, "data")
RES = os.path.join(HERE, "results")
for d in (ART, DATA, RES):
    os.makedirs(d, exist_ok=True)

SEED = 42


# ----------------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------------
def get_datasets(quick: bool):
    """Return {name: (X_train, y_train, X_probe, task, n_classes)}."""
    out = {}

    # 1. Tiny binary problem. Its job is to isolate the FIXED overhead:
    #    with a model this small, whatever time remains is pure import tax.
    d = load_breast_cancer()
    Xtr, Xte, ytr, yte = train_test_split(
        d.data, d.target, test_size=0.3, random_state=SEED, stratify=d.target
    )
    out["breast"] = (Xtr, ytr, Xte, "binary", 2)

    if quick:
        return out

    # 2. Medium regression problem.
    d = fetch_california_housing()
    Xtr, Xte, ytr, yte = train_test_split(
        d.data, d.target, test_size=0.3, random_state=SEED
    )
    out["calif"] = (Xtr, ytr, Xte, "regression", 1)

    # 3. Large multiclass problem. 7 classes means the artifact holds
    #    7x as many trees, which is where formats start to separate.
    d = fetch_covtype()
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(d.data), size=50_000, replace=False)
    X, y = d.data[idx], d.target[idx] - 1
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )
    out["covtype"] = (Xtr, ytr, Xte, "multiclass", 7)

    return out


def tree_grid(dataset: str, quick: bool):
    if quick:
        return [50]
    if dataset == "covtype":
        return [100, 500]          # 7 classes, so 500 -> 3500 actual trees
    return [100, 500, 2000]


# ----------------------------------------------------------------------------
# Model construction
# ----------------------------------------------------------------------------
def build_model(lib: str, task: str, n_trees: int, n_classes: int):
    if lib == "lgb":
        import lightgbm as lgb
        kw = dict(n_estimators=n_trees, num_leaves=31, learning_rate=0.1,
                  random_state=SEED, n_jobs=-1, verbose=-1)
        return lgb.LGBMRegressor(**kw) if task == "regression" else lgb.LGBMClassifier(**kw)

    if lib == "xgb":
        import xgboost as xgb
        kw = dict(n_estimators=n_trees, max_depth=6, learning_rate=0.1,
                  random_state=SEED, n_jobs=-1, tree_method="hist", verbosity=0)
        if task == "regression":
            return xgb.XGBRegressor(**kw)
        if task == "multiclass":
            return xgb.XGBClassifier(objective="multi:softprob",
                                     num_class=n_classes, **kw)
        return xgb.XGBClassifier(**kw)

    if lib == "cat":
        from catboost import CatBoostClassifier, CatBoostRegressor
        kw = dict(iterations=n_trees, depth=6, learning_rate=0.1,
                  random_seed=SEED, verbose=0, allow_writing_files=False)
        return CatBoostRegressor(**kw) if task == "regression" else CatBoostClassifier(**kw)

    raise ValueError(lib)


# ----------------------------------------------------------------------------
# Export paths. Each returns (path, seconds_to_export) or raises.
# ----------------------------------------------------------------------------
def export_pickle(model, tag):
    p = os.path.join(ART, f"{tag}.pkl")
    t0 = time.perf_counter()
    with open(p, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    return p, time.perf_counter() - t0


def export_joblib(model, tag):
    p = os.path.join(ART, f"{tag}.joblib")
    t0 = time.perf_counter()
    joblib.dump(model, p, compress=0)
    return p, time.perf_counter() - t0


def export_native(model, tag, lib):
    if lib == "lgb":
        p = os.path.join(ART, f"{tag}.lgbtxt")
        t0 = time.perf_counter()
        model.booster_.save_model(p)
    elif lib == "xgb":
        p = os.path.join(ART, f"{tag}.ubj")
        t0 = time.perf_counter()
        model.get_booster().save_model(p)
    else:
        p = os.path.join(ART, f"{tag}.cbm")
        t0 = time.perf_counter()
        model.save_model(p)
    return p, time.perf_counter() - t0


def export_onnx(model, tag, lib, n_features):
    p = os.path.join(ART, f"{tag}.onnx")
    t0 = time.perf_counter()

    if lib == "cat":
        model.save_model(p, format="onnx")
    else:
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType
        itypes = [("input", FloatTensorType([None, n_features]))]
        # zipmap=False keeps the classifier output as a plain array instead of
        # a list of dicts. Leaving it on adds a large, purely cosmetic cost at
        # inference time and would confound the comparison.
        if lib == "lgb":
            onx = onnxmltools.convert_lightgbm(model, initial_types=itypes,
                                               target_opset=13, zipmap=False)
        else:
            try:
                onx = onnxmltools.convert_xgboost(
                    model, initial_types=itypes, target_opset=13,
                    options={id(model): {"zipmap": False}})
            except TypeError:
                onx = onnxmltools.convert_xgboost(model, initial_types=itypes,
                                                  target_opset=13)
        with open(p, "wb") as f:
            f.write(onx.SerializeToString())

    return p, time.perf_counter() - t0


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="one tiny dataset, one model size - use this first")
    ap.add_argument("--libs", default="lgb,xgb,cat")
    args = ap.parse_args()

    libs = args.libs.split(",")
    datasets = get_datasets(args.quick)
    rows = []

    for dname, (Xtr, ytr, Xprobe, task, n_classes) in datasets.items():
        np.save(os.path.join(DATA, f"{dname}_X.npy"),
                np.ascontiguousarray(Xprobe, dtype=np.float64))
        n_feat = Xtr.shape[1]

        for lib in libs:
            for n_trees in tree_grid(dname, args.quick):
                tag = f"{dname}__{lib}__t{n_trees}"
                model = build_model(lib, task, n_trees, n_classes)

                t0 = time.perf_counter()
                model.fit(Xtr, ytr)
                t_train = time.perf_counter() - t0
                print(f"[train] {tag:34s} {t_train:7.2f}s", flush=True)

                exporters = {
                    "pickle": lambda: export_pickle(model, tag),
                    "joblib": lambda: export_joblib(model, tag),
                    "native": lambda: export_native(model, tag, lib),
                    "onnx":   lambda: export_onnx(model, tag, lib, n_feat),
                }

                for fmt, fn in exporters.items():
                    try:
                        path, t_exp = fn()
                        size = os.path.getsize(path)
                        status, err = "ok", ""
                    except Exception as e:                      # noqa: BLE001
                        path, t_exp, size = "", float("nan"), -1
                        status, err = "failed", f"{type(e).__name__}: {e}"[:200]

                    rows.append(dict(
                        dataset=dname, lib=lib, fmt=fmt, n_trees=n_trees,
                        task=task, n_classes=n_classes, n_features=n_feat,
                        loader=f"{lib}_{fmt}", artifact=path,
                        bytes=size, export_s=t_exp, train_s=t_train,
                        status=status, error=err,
                    ))
                    flag = "ok " if status == "ok" else "FAIL"
                    print(f"   [{flag}] {fmt:7s} {size/1e6:8.3f} MB  "
                          f"export {t_exp:6.3f}s {err}", flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(RES, "manifest.csv")
    df.to_csv(out, index=False)
    n_ok = (df.status == "ok").sum()
    print(f"\nmanifest -> {out}   ({n_ok}/{len(df)} artifacts exported)")


if __name__ == "__main__":
    main()
