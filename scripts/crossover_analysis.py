"""crossover_analysis.py - the batch size at which ONNX stops paying off.

The crossover is the smallest batch size at which the ONNX path is
significantly slower, meaning the lower bound of the bootstrap 95 percent
interval for the ONNX to native steady state latency ratio lies above one, and
at which it stays significantly slower at every larger batch tested.

A significance requirement is used rather than simple interpolation of the
point where the ratio crosses one. Interpolation reports a confident crossover
wherever the ratio happens to wander across one by chance, which is what
happens in cells where the two paths perform comparably.

Every batch sweep found under results/ is analysed, so the flat file for
Environment A and results/env_B/raw_batchsweep.jsonl for Environment B.

Run:  python scripts/crossover_analysis.py
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

LIBNAME = {"lgb": "LightGBM", "xgb": "XGBoost", "cat": "CatBoost"}
DSNAME = {"breast": "Breast Cancer", "calif": "California Housing",
          "covtype": "Covertype"}
PALETTE = {"lgb": "#e1b12c", "xgb": "#487eb0", "cat": "#44bd32"}


def load():
    paths = [os.path.join(RES, "raw_batchsweep.jsonl")]
    paths += sorted(glob.glob(os.path.join(RES, "env_*",
                                           "raw_batchsweep.jsonl")))
    frames = []
    for p in paths:
        if not os.path.exists(p):
            continue
        d = pd.read_json(p, lines=True)
        if "env_label" not in d.columns:
            d["env_label"] = os.path.basename(os.path.dirname(p))[4:] or "A"
        frames.append(d)
    if not frames:
        raise SystemExit("no raw_batchsweep.jsonl found under results/")
    return pd.concat(frames, ignore_index=True)


def analyse(sw, env, rng, n_boot=4000):
    sub = sw[sw.env_label == env]
    batches = sorted(sub.batch.unique())

    def boot(ds, lib, b):
        o = sub[(sub.dataset == ds) & (sub.lib == lib) & (sub.batch == b)
                & (sub.fmt == "onnx")].t_steady_p50_s.values
        v = sub[(sub.dataset == ds) & (sub.lib == lib) & (sub.batch == b)
                & (sub.fmt == "native")].t_steady_p50_s.values
        if len(o) == 0 or len(v) == 0:
            return np.nan, np.nan, np.nan
        r = [np.median(rng.choice(o, len(o))) / np.median(rng.choice(v, len(v)))
             for _ in range(n_boot)]
        return tuple(np.percentile(r, [2.5, 50, 97.5]))

    records, rows = [], []
    for ds in DSNAME:
        for lib in LIBNAME:
            cis = {b: boot(ds, lib, b) for b in batches}
            for b, (lo, md, hi) in cis.items():
                records.append(dict(env_label=env, dataset=ds, lib=lib,
                                    batch=b, ratio_lo=lo, ratio=md,
                                    ratio_hi=hi, onnx_slower=lo > 1))
            sig = [cis[b][0] > 1 for b in batches]
            first = next((batches[i] for i in range(len(batches))
                          if all(sig[i:])), None)
            lo, md, hi = cis[batches[-1]]
            rows.append(dict(env_label=env, dataset=DSNAME[ds],
                             lib=LIBNAME[lib],
                             crossover_batch=first if first else np.inf,
                             ratio_at_max=md, ci_lo=lo, ci_hi=hi,
                             max_batch=batches[-1]))
    return pd.DataFrame(records), pd.DataFrame(rows)


def figure(curves, env="A"):
    sub = curves[curves.env_label == env]
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.2), sharey=True)
    for ax, ds in zip(axes, DSNAME):
        for lib in LIBNAME:
            s = sub[(sub.dataset == ds) & (sub.lib == lib)].sort_values("batch")
            ax.plot(s.batch, s.ratio, marker="o", ms=4, lw=1.8,
                    color=PALETTE[lib], label=LIBNAME[lib])
            ax.fill_between(s.batch, s.ratio_lo, s.ratio_hi,
                            color=PALETTE[lib], alpha=0.18, lw=0)
        ax.axhline(1.0, color="#2d3436", lw=1, ls="--")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(DSNAME[ds], fontsize=10)
        ax.set_xlabel("batch size")
        ax.grid(alpha=0.22, which="both")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("ONNX to native latency ratio")
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out = os.path.join(FIG, "fig5_crossover_ci.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {out}")


def main():
    sw = load()
    rng = np.random.default_rng(0)
    all_curves, all_cross = [], []

    for env in sorted(sw.env_label.unique()):
        curves, cross = analyse(sw, env, rng)
        all_curves.append(curves)
        all_cross.append(cross)
        print(f"=== Environment {env}: crossover, bootstrap 95 percent "
              f"interval ===")
        print(cross.drop(columns=["env_label"]).round(2).to_string(index=False))
        print()

    curves = pd.concat(all_curves, ignore_index=True)
    cross = pd.concat(all_cross, ignore_index=True)
    curves.to_csv(os.path.join(RES, "crossover_curves.csv"), index=False)
    cross.to_csv(os.path.join(RES, "crossover_bootstrap.csv"), index=False)
    cross[cross.env_label == "A"].drop(columns=["env_label"]).to_csv(
        os.path.join(RES, "table6_crossover.csv"), index=False)

    figure(curves, env="A")
    print(f"curves -> {os.path.join(RES, 'crossover_curves.csv')}")
    print(f"crossover -> {os.path.join(RES, 'crossover_bootstrap.csv')}")


if __name__ == "__main__":
    main()
